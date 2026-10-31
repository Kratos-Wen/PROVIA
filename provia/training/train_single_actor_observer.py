"""Train the completion-only reduction of the causal marked event observer.

The optimization uses only event times and action marks from the training
partition.  Boundary and mark parameters are disjoint, use independent AdamW
optimizers, and are restored at independent validation proper-loss minima.
Correctness/error labels are never read by the optimizer or checkpoint rule.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from provia.data.completion_stream import (
    CompletionInferenceStream,
    CompletionStream,
    load_completion_streams,
)
from provia.data.io import read_jsonl
from provia.observer.completion_process import (
    MarkedCompletionProcess,
    decode_completion_posterior_medians,
)
from provia.observer.factors import EVENT_FACTORS
from provia.training.common import (
    _append_jsonl,
    _atomic_json,
    _cache_provenance,
    _jeffreys_prior,
    _partitions,
    _prior_corrected_nll,
    _seed_everything,
)
from provia.model.intents import CompositionalIntentSpace


EVENT_PREFIXES = ("event_map.", "event_temporal.", "intensity_head.")
SEMANTIC_PREFIXES = ("semantic_map.", "semantic_temporal.", "factor_heads.")
CONTEXT_MODEL_DIM = 192


def _component_state(
    model: MarkedCompletionProcess, prefixes: Sequence[str]
) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith(tuple(prefixes))
    }


def _restore_components(
    model: MarkedCompletionProcess,
    event_state: Mapping[str, Tensor],
    semantic_state: Mapping[str, Tensor],
) -> None:
    state = model.state_dict()
    expected_event = {name for name in state if name.startswith(EVENT_PREFIXES)}
    expected_semantic = {
        name for name in state if name.startswith(SEMANTIC_PREFIXES)
    }
    if set(event_state) != expected_event or set(semantic_state) != expected_semantic:
        raise ValueError("completion-process component checkpoint is incomplete")
    state.update(event_state)
    state.update(semantic_state)
    model.load_state_dict(state, strict=True)


def _base_rate(streams: Sequence[CompletionStream]) -> tuple[float, int, float]:
    event_count = sum(len(stream.event_indices) for stream in streams)
    exposure = sum(
        float(stream.timestamps_sec[-1] - stream.timestamps_sec[0])
        for stream in streams
        if len(stream.timestamps_sec) > 1
    )
    if event_count <= 0 or exposure <= 0:
        raise ValueError("completion-rate estimation needs events and positive exposure")
    return (event_count + 0.5) / exposure, event_count, exposure


def _completion_dilations(
    streams: Sequence[CompletionStream],
) -> tuple[int, ...]:
    """Smallest dyadic causal field covering every training inter-event gap."""

    maximum_gap = 1
    for stream in streams:
        indices = [0, *(int(value) for value in stream.event_indices.tolist())]
        maximum_gap = max(
            maximum_gap,
            max((right - left for left, right in zip(indices, indices[1:])), default=1),
        )
    result = []
    dilation = 1
    receptive_field = 1
    while receptive_field < maximum_gap + 1:
        result.append(dilation)
        receptive_field += 2 * dilation
        dilation *= 2
    return tuple(result or (1,))


def _factor_counts(
    streams: Sequence[CompletionStream], labels: Mapping[str, Sequence[str]]
) -> dict[str, Tensor]:
    counts = {
        name: torch.zeros(len(labels[name]), dtype=torch.float64)
        for name in EVENT_FACTORS
    }
    for stream in streams:
        for name in EVENT_FACTORS:
            counts[name] += torch.bincount(
                stream.factor_targets[name], minlength=len(labels[name])
            ).to(torch.float64)
    return counts


def _emit(
    model: MarkedCompletionProcess,
    stream: CompletionStream | CompletionInferenceStream,
    device: torch.device,
):
    return model(
        stream.global_features.to(device),
        torch.stack(
            (stream.change_features, torch.zeros_like(stream.change_features)),
            dim=1,
        ).to(device),
        stream.timestamps_sec.to(device=device, dtype=torch.float32),
    )


def _event_nll(
    intensity: Tensor, timestamps: Tensor, event_indices: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    if len(timestamps) < 2:
        raise ValueError("completion likelihood requires at least two frames")
    delta = timestamps[1:] - timestamps[:-1]
    if torch.any(delta <= 0):
        raise ValueError("completion timestamps must be strictly increasing")
    integrated = intensity[1:] * delta.to(intensity)
    target = torch.zeros(len(timestamps) - 1, dtype=torch.bool, device=intensity.device)
    valid_indices = event_indices[event_indices > 0] - 1
    if len(valid_indices) != len(event_indices):
        raise ValueError("a completion at the first cached frame is left-censored")
    target[valid_indices.to(device=intensity.device)] = True
    event_probability = (-torch.expm1(-integrated)).clamp_min(
        torch.finfo(integrated.dtype).tiny
    )
    log_event = torch.log(event_probability)
    log_probability = torch.where(target, log_event, -integrated)
    probability = torch.cat(
        (torch.zeros_like(intensity[:1]), event_probability), dim=0
    )
    return -log_probability.mean(), probability, target


def _semantic_nll(
    factor_logits: Mapping[str, Tensor],
    stream: CompletionStream,
    factor_counts: Mapping[str, Tensor],
    device: torch.device,
) -> tuple[Tensor, dict[str, Tensor]]:
    indices = stream.event_indices.to(device=device, dtype=torch.long)
    values = {
        name: _prior_corrected_nll(
            factor_logits[name].index_select(0, indices),
            stream.factor_targets[name].to(device=device, dtype=torch.long),
            factor_counts[name],
        )
        for name in EVENT_FACTORS
    }
    return torch.stack(tuple(values.values())).sum(), values


def _sequence_detection(
    target_indices: np.ndarray,
    predicted_indices: np.ndarray,
    timestamps: np.ndarray,
) -> dict[str, float | int]:
    """Threshold-free sequence coverage using target Voronoi cells."""

    if not len(target_indices):
        raise ValueError("sequence detection requires target completions")
    boundaries = np.empty(len(target_indices) + 1, dtype=np.float64)
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf
    if len(target_indices) > 1:
        boundaries[1:-1] = 0.5 * (
            timestamps[target_indices[:-1]] + timestamps[target_indices[1:]]
        )
    assigned = np.searchsorted(
        boundaries, timestamps[predicted_indices], side="right"
    ) - 1
    occupied = np.unique(assigned) if len(assigned) else np.empty(0, dtype=np.int64)
    delays = []
    for target in occupied:
        candidates = predicted_indices[assigned == target]
        nearest = candidates[
            np.abs(timestamps[candidates] - timestamps[target_indices[target]]).argmin()
        ]
        delays.append(float(timestamps[nearest] - timestamps[target_indices[target]]))
    true_positive = len(occupied)
    return {
        "target_events": int(len(target_indices)),
        "predicted_events": int(len(predicted_indices)),
        "covered_target_events": int(true_positive),
        "precision": float(true_positive / max(len(predicted_indices), 1)),
        "recall": float(true_positive / len(target_indices)),
        "f1": float(
            2 * true_positive / max(len(target_indices) + len(predicted_indices), 1)
        ),
        "mean_signed_delay_sec": float(np.mean(delays)) if delays else 0.0,
        "mean_absolute_delay_sec": float(np.mean(np.abs(delays))) if delays else 0.0,
    }


@torch.no_grad()
def _evaluate(
    model: MarkedCompletionProcess,
    streams: Sequence[CompletionStream],
    factor_counts: Mapping[str, Tensor],
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    event_losses = []
    semantic_losses = {name: [] for name in EVENT_FACTORS}
    event_targets = []
    event_scores = []
    factor_correct = {name: 0 for name in EVENT_FACTORS}
    factor_total = 0
    sequence_rows = []
    priors = {
        name: _jeffreys_prior(
            factor_counts[name], device=torch.device("cpu"), dtype=torch.float32
        )
        for name in EVENT_FACTORS
    }
    for stream in streams:
        evidence = _emit(model, stream, device)
        event_loss, probability, target = _event_nll(
            evidence.intensity_per_second,
            stream.timestamps_sec.to(device=device, dtype=torch.float32),
            stream.event_indices.to(device),
        )
        _, semantic = _semantic_nll(
            evidence.factor_logits, stream, factor_counts, device
        )
        event_losses.append(float(event_loss))
        for name in EVENT_FACTORS:
            semantic_losses[name].append(float(semantic[name]))
            posterior = (
                evidence.factor_logits[name].index_select(
                    0, stream.event_indices.to(device)
                )
                + priors[name].to(device).log()
            )
            factor_correct[name] += int(
                (
                    posterior.argmax(dim=-1)
                    == stream.factor_targets[name].to(device)
                ).sum()
            )
        factor_total += len(stream.event_indices)
        event_targets.append(target.cpu().numpy())
        event_scores.append(probability[1:].cpu().numpy())
        decoded = decode_completion_posterior_medians(
            probability.cpu(),
            {name: values.cpu() for name, values in evidence.factor_logits.items()},
            priors,
        )
        sequence_rows.append(
            _sequence_detection(
                stream.event_indices.numpy(),
                np.asarray([item.onset_index for item in decoded], dtype=np.int64),
                stream.timestamps_sec.numpy(),
            )
        )
    target = np.concatenate(event_targets).astype(np.int64)
    score = np.concatenate(event_scores)
    return {
        "event_time_nll": float(np.mean(event_losses)),
        "mean_semantic_nll": float(
            np.mean(
                [np.mean(semantic_losses[name]) for name in EVENT_FACTORS]
            )
        ),
        "component_semantic_nll": {
            name: float(np.mean(values)) for name, values in semantic_losses.items()
        },
        "event_time_auprc": float(average_precision_score(target, score)),
        "event_time_prevalence": float(target.mean()),
        "event_factor_accuracy": {
            name: factor_correct[name] / max(factor_total, 1)
            for name in EVENT_FACTORS
        },
        "posterior_median_sequence": {
            key: float(np.mean([float(row[key]) for row in sequence_rows]))
            for key in ("precision", "recall", "f1", "mean_signed_delay_sec", "mean_absolute_delay_sec")
        }
        | {
            "target_events": int(sum(int(row["target_events"]) for row in sequence_rows)),
            "predicted_events": int(sum(int(row["predicted_events"]) for row in sequence_rows)),
            "covered_target_events": int(
                sum(int(row["covered_target_events"]) for row in sequence_rows)
            ),
        },
        "frames": int(len(target)),
        "events": int(factor_total),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--event-learning-rate", type=float, required=True)
    parser.add_argument("--semantic-learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--early-stopping-epochs", type=int, default=6)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (
        args.event_learning_rate <= 0
        or args.semantic_learning_rate <= 0
        or args.weight_decay < 0
        or args.epochs <= 0
        or args.early_stopping_epochs <= 0
        or args.gradient_clip <= 0
    ):
        parser.error("optimization values are invalid")
    _seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    episodes = tuple(read_jsonl(args.manifest))
    train_native, validation_native = _partitions(episodes, args.fold)
    intent_space = CompositionalIntentSpace.from_segments(
        segment for episode in train_native for segment in episode.segments
    )
    train = load_completion_streams(train_native, args.feature_root, intent_space)
    validation = load_completion_streams(
        validation_native, args.feature_root, intent_space
    )
    labels = {name: intent_space.labels[name] for name in EVENT_FACTORS}
    factor_counts = _factor_counts(train, labels)
    base_rate, base_events, exposure_sec = _base_rate(train)
    dilations = _completion_dilations(train)
    model = MarkedCompletionProcess(
        int(train[0].global_features.shape[-1]),
        labels,
        base_rate,
        model_dim=CONTEXT_MODEL_DIM,
        dilations=dilations,
    ).to(device)
    event_parameters = [
        value
        for name, value in model.named_parameters()
        if name.startswith(EVENT_PREFIXES)
    ]
    semantic_parameters = [
        value
        for name, value in model.named_parameters()
        if name.startswith(SEMANTIC_PREFIXES)
    ]
    if (
        not event_parameters
        or not semantic_parameters
        or len(event_parameters) + len(semantic_parameters)
        != sum(1 for _ in model.parameters())
    ):
        raise ValueError("completion parameters are not exactly factorized")
    event_optimizer = torch.optim.AdamW(
        event_parameters,
        lr=args.event_learning_rate,
        weight_decay=args.weight_decay,
    )
    semantic_optimizer = torch.optim.AdamW(
        semantic_parameters,
        lr=args.semantic_learning_rate,
        weight_decay=args.weight_decay,
    )

    cache = _cache_provenance(args.feature_root, args.fold)
    if cache.get("role_mode") != "single_actor":
        raise ValueError("completion training requires a single-actor feature cache")
    metadata = {
        "schema": "marked_completion_process_training",
        "observer_architecture": "completion_process_observer",
        "method": "PROVIA",
        "active_roles": ["left"],
        "inactive_roles": ["right_structural_null"],
        "decoder": "causal posterior median of the next completion time",
        "model_dim": CONTEXT_MODEL_DIM,
        "dilations": list(dilations),
        "temporal_kernel_size": 3,
        "dropout": 0.1,
        "receptive_field_frames": 1 + 2 * sum(dilations),
        "anomaly_labels_used_for_optimization": False,
        "anomaly_metrics_used_for_selection": False,
        "test_split_accessed": False,
        "fold": args.fold,
        "seed": args.seed,
        "train_episodes": len(train),
        "validation_episodes": len(validation),
        "train_frames": sum(len(stream.frame_ids) for stream in train),
        "validation_frames": sum(len(stream.frame_ids) for stream in validation),
        "train_events": base_events,
        "validation_events": sum(len(stream.event_indices) for stream in validation),
        "base_rate_per_second": base_rate,
        "training_exposure_sec": exposure_sec,
        "event_balanced_factor_counts": {
            name: factor_counts[name].tolist() for name in EVENT_FACTORS
        },
        "parameters": {
            "total": sum(value.numel() for value in model.parameters()),
            "trainable": sum(value.numel() for value in model.parameters() if value.requires_grad),
        },
        "cache": cache,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    _atomic_json(args.output_dir / "run.json", metadata)
    _atomic_json(args.output_dir / "intent_space.json", intent_space.to_dict())

    initial = _evaluate(model, validation, factor_counts, device)
    best_event_value = float(initial["event_time_nll"])
    best_semantic_value = float(initial["mean_semantic_nll"])
    best_event_epoch = 0
    best_semantic_epoch = 0
    best_event = _component_state(model, EVENT_PREFIXES)
    best_semantic = _component_state(model, SEMANTIC_PREFIXES)
    event_stale = semantic_stale = 0
    _append_jsonl(args.output_dir / "metrics.jsonl", {"epoch": 0, "validation": initial})

    generator = torch.Generator().manual_seed(args.seed)
    started = time.monotonic()
    epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator)
        train_event = []
        train_semantic = []
        event_norm = []
        semantic_norm = []
        for stream_index in order.tolist():
            stream = train[stream_index]
            event_optimizer.zero_grad(set_to_none=True)
            semantic_optimizer.zero_grad(set_to_none=True)
            evidence = _emit(model, stream, device)
            event_loss, _, _ = _event_nll(
                evidence.intensity_per_second,
                stream.timestamps_sec.to(device=device, dtype=torch.float32),
                stream.event_indices.to(device),
            )
            semantic_loss, _ = _semantic_nll(
                evidence.factor_logits, stream, factor_counts, device
            )
            event_loss.backward()
            semantic_loss.backward()
            event_norm.append(float(clip_grad_norm_(event_parameters, args.gradient_clip)))
            semantic_norm.append(
                float(clip_grad_norm_(semantic_parameters, args.gradient_clip))
            )
            event_optimizer.step()
            semantic_optimizer.step()
            train_event.append(float(event_loss.detach()))
            train_semantic.append(float(semantic_loss.detach()))
        validation_result = _evaluate(model, validation, factor_counts, device)
        event_value = float(validation_result["event_time_nll"])
        semantic_value = float(validation_result["mean_semantic_nll"])
        if event_value < best_event_value - 1e-12:
            best_event_value = event_value
            best_event_epoch = epoch
            best_event = _component_state(model, EVENT_PREFIXES)
            event_stale = 0
        else:
            event_stale += 1
        if semantic_value < best_semantic_value - 1e-12:
            best_semantic_value = semantic_value
            best_semantic_epoch = epoch
            best_semantic = _component_state(model, SEMANTIC_PREFIXES)
            semantic_stale = 0
        else:
            semantic_stale += 1
        _append_jsonl(
            args.output_dir / "metrics.jsonl",
            {
                "epoch": epoch,
                "elapsed_sec": time.monotonic() - started,
                "train_event_time_nll": float(np.mean(train_event)),
                "train_semantic_nll_sum": float(np.mean(train_semantic)),
                "gradient_norm": {
                    "event": float(np.mean(event_norm)),
                    "semantics": float(np.mean(semantic_norm)),
                },
                "validation": validation_result,
                "best_event_epoch": best_event_epoch,
                "best_semantic_epoch": best_semantic_epoch,
            },
        )
        if (
            event_stale >= args.early_stopping_epochs
            and semantic_stale >= args.early_stopping_epochs
        ):
            break

    _restore_components(model, best_event, best_semantic)
    final_validation = _evaluate(model, validation, factor_counts, device)
    checkpoint_metadata = {
        **metadata,
        "best_event_epoch": best_event_epoch,
        "best_event_time_nll": best_event_value,
        "best_semantic_epoch": best_semantic_epoch,
        "best_mean_semantic_nll": best_semantic_value,
    }
    torch.save(
        {"model": model.state_dict(), "metadata": checkpoint_metadata},
        args.output_dir / "best.pt",
    )
    _atomic_json(args.output_dir / "validation.json", final_validation)
    _atomic_json(
        args.output_dir / "complete.json",
        {
            "status": "complete",
            "epochs_completed": epoch,
            "best_event_epoch": best_event_epoch,
            "best_event_time_nll": best_event_value,
            "best_semantic_epoch": best_semantic_epoch,
            "best_mean_semantic_nll": best_semantic_value,
            "elapsed_sec": time.monotonic() - started,
            "test_split_accessed": False,
        },
    )
    print(json.dumps(final_validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
