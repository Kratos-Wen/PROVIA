"""Evaluate PROVIA on fully predicted and segment-support completion protocols."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from provia.data.completion_stream import CompletionInferenceStream, CompletionStream
from provia.observer.completion_process import (
    MarkedCompletionProcess,
    decode_completion_posterior_medians,
)
from provia.observer.factors import EVENT_FACTORS
from provia.training.train_single_actor_observer import _emit
from provia.training.common import _jeffreys_prior
from provia.model.interval_transport import IntervalContinuationLanguage


def _read_language(path: Path) -> IntervalContinuationLanguage:
    with open(path, encoding="utf-8") as handle:
        return IntervalContinuationLanguage.from_dict(json.load(handle))


def _priors(metadata: Mapping[str, object]) -> dict[str, torch.Tensor]:
    raw = metadata.get("event_balanced_factor_counts")
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint lacks factor counts")
    return {
        name: _jeffreys_prior(
            torch.tensor(raw[name], dtype=torch.float64),  # type: ignore[arg-type]
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        for name in EVENT_FACTORS
    }


def _binary_metrics(
    target: np.ndarray, score: np.ndarray, threshold: float = 0.5
) -> dict[str, float | int]:
    target = np.asarray(target, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if target.ndim != 1 or score.shape != target.shape or len(np.unique(target)) != 2:
        raise ValueError("binary metrics require aligned positive and negative samples")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("decision threshold must lie in [0,1]")
    prediction = score >= threshold
    positive = target.astype(bool)
    true_positive = int(np.sum(prediction & positive))
    false_positive = int(np.sum(prediction & ~positive))
    true_negative = int(np.sum(~prediction & ~positive))
    false_negative = int(np.sum(~prediction & positive))
    return {
        "support": int(len(target)),
        "positives": int(target.sum()),
        "prevalence": float(target.mean()),
        "average_precision": float(average_precision_score(target, score)),
        "auroc": float(roc_auc_score(target, score)),
        "operating_threshold": float(threshold),
        "accuracy_at_operating_point": float(np.mean(prediction == positive)),
        "precision_at_operating_point": float(
            true_positive / max(true_positive + false_positive, 1)
        ),
        "recall_at_operating_point": float(
            true_positive / max(true_positive + false_negative, 1)
        ),
        "f1_at_operating_point": float(f1_score(target, prediction, zero_division=0)),
        "mean_class_f1_at_operating_point": float(
            0.5
            * (
                f1_score(target, prediction, pos_label=0, zero_division=0)
                + f1_score(target, prediction, pos_label=1, zero_division=0)
            )
        ),
        "confusion": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        },
    }


def _cell_assignment(
    target_indices: np.ndarray,
    predicted_indices: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    if not len(target_indices):
        raise ValueError("completion cells require target events")
    boundaries = np.empty(len(target_indices) + 1, dtype=np.float64)
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf
    if len(target_indices) > 1:
        boundaries[1:-1] = 0.5 * (
            timestamps[target_indices[:-1]] + timestamps[target_indices[1:]]
        )
    return np.searchsorted(
        boundaries, timestamps[predicted_indices], side="right"
    ) - 1


def _fit_mode_dynamics(
    streams: Sequence[CompletionStream],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    initial_counts = np.zeros(3, dtype=np.int64)
    transition_counts = np.zeros((3, 3), dtype=np.int64)
    mode_counts = np.zeros(3, dtype=np.int64)
    for stream in streams:
        previous = None
        for mode in stream.correctness.numpy().tolist():
            mode = int(mode)
            mode_counts[mode] += 1
            if previous is None:
                initial_counts[mode] += 1
            else:
                transition_counts[previous, mode] += 1
            previous = mode
    initial = initial_counts.astype(np.float64) + 0.5
    initial /= initial.sum()
    transition = transition_counts.astype(np.float64) + 0.5
    transition /= transition.sum(axis=-1, keepdims=True)
    return transition, initial, {
        "mode_order": ["normal", "error", "recovery"],
        "mode_counts": mode_counts.tolist(),
        "initial_counts": initial_counts.tolist(),
        "transition_counts": transition_counts.tolist(),
        "unobserved_modes_retained_by_prior": [
            name
            for name, count in zip(("normal", "error", "recovery"), mode_counts)
            if count == 0
        ],
    }


@torch.no_grad()
def _decode(
    model: MarkedCompletionProcess,
    stream: CompletionStream | CompletionInferenceStream,
    priors: Mapping[str, torch.Tensor],
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, tuple[float, ...]]]]:
    evidence = _emit(model, stream, device)
    probability = model.event_probability(
        evidence.intensity_per_second,
        stream.timestamps_sec.to(device=device, dtype=torch.float32),
    )
    events = decode_completion_posterior_medians(
        probability.cpu(),
        {name: values.cpu() for name, values in evidence.factor_logits.items()},
        priors,
    )
    return (
        np.asarray([event.onset_index for event in events], dtype=np.int64),
        [
            {
                name: tuple(float(np.exp(value)) for value in event.factor_log_probability[name])
                for name in EVENT_FACTORS
            }
            for event in events
        ],
    )


def _fit_deletion_probability(
    model: MarkedCompletionProcess,
    streams: Sequence[CompletionStream],
    priors: Mapping[str, torch.Tensor],
    device: torch.device,
) -> tuple[float, dict[str, object]]:
    total = 0
    covered = 0
    predicted = 0
    for stream in streams:
        indices, _ = _decode(model, stream, priors, device)
        assignment = _cell_assignment(
            stream.event_indices.numpy(), indices, stream.timestamps_sec.numpy()
        )
        total += len(stream.event_indices)
        covered += len(np.unique(assignment)) if len(assignment) else 0
        predicted += len(indices)
    probability = (total - covered + 0.5) / (total + 1.0)
    return float(probability), {
        "target_events": total,
        "covered_events": covered,
        "predicted_events": predicted,
        "missed_event_probability": float(probability),
        "validation_labels_used": False,
    }


def _pool_predicted_to_targets(
    stream: CompletionStream,
    predicted_indices: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assignment = _cell_assignment(
        stream.event_indices.numpy(), predicted_indices, stream.timestamps_sec.numpy()
    )
    pooled = np.zeros(len(stream.event_indices), dtype=np.float64)
    selected_index = np.full(len(stream.event_indices), -1, dtype=np.int64)
    for target in range(len(pooled)):
        candidates = np.flatnonzero(assignment == target)
        if len(candidates):
            best = candidates[int(np.argmax(scores[candidates]))]
            pooled[target] = scores[best]
            selected_index[target] = predicted_indices[best]
    return pooled, selected_index, assignment


def _category_metrics(
    target: np.ndarray,
    score: np.ndarray,
    categories: Sequence[Sequence[str]],
    threshold: float,
) -> dict[str, object]:
    names = sorted({name for values in categories for name in values})
    result = {}
    for name in names:
        positive = np.asarray([name in values for values in categories])
        keep = (target == 0) | positive
        if positive.sum() and np.any(target[keep] == 0):
            result[name] = _binary_metrics(
                positive[keep].astype(np.int64), score[keep], threshold
            )
    return result
