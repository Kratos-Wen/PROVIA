"""Event support metrics of the step observer."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from provia.automaton.decoded_events import DecodedEvent
from provia.data.role_stream import CompetitiveRoleStream
from provia.observer.factors import EVENT_FACTORS


def _iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start) + 1)
    union = max(a_end, b_end) - min(a_start, b_start) + 1
    return intersection / max(union, 1)


def _matches(
    predictions: Sequence[DecodedEvent],
    stream: CompetitiveRoleStream,
    threshold: float,
) -> list[tuple[int, int, float]]:
    candidates = sorted(
        (
            (
                prediction_index,
                target_index,
                _iou(
                    prediction.start_index,
                    prediction.end_index,
                    target.start,
                    target.end,
                ),
            )
            for prediction_index, prediction in enumerate(predictions)
            for target_index, target in enumerate(stream.intervals)
            if prediction.role == target.role
        ),
        key=lambda value: (-value[2], value[0], value[1]),
    )
    used_predictions: set[int] = set()
    used_targets: set[int] = set()
    result = []
    for prediction, target, overlap in candidates:
        if overlap < threshold:
            break
        if prediction in used_predictions or target in used_targets:
            continue
        used_predictions.add(prediction)
        used_targets.add(target)
        result.append((prediction, target, overlap))
    return result


def _summarize(
    decoded: Sequence[tuple[CompetitiveRoleStream, Sequence[DecodedEvent]]]
) -> dict[str, object]:
    thresholds = (0.1, 0.25, 0.5)
    counts = {
        threshold: {"tp": 0, "predicted": 0, "target": 0}
        for threshold in thresholds
    }
    boundary_error = {name: [] for name in ("start", "onset", "end")}
    factor_correct = {name: 0 for name in EVENT_FACTORS}
    factor_total = 0
    for stream, predictions in decoded:
        for threshold in thresholds:
            matched = _matches(predictions, stream, threshold)
            counts[threshold]["tp"] += len(matched)
            counts[threshold]["predicted"] += len(predictions)
            counts[threshold]["target"] += len(stream.intervals)
        for prediction_index, target_index, _ in _matches(predictions, stream, 0.1):
            prediction = predictions[prediction_index]
            target = stream.intervals[target_index]
            for name, predicted_index, target_index_value in (
                ("start", prediction.start_index, target.start),
                ("onset", prediction.onset_index, target.onset),
                ("end", prediction.end_index, target.end),
            ):
                boundary_error[name].append(
                    abs(
                        float(stream.timestamps_sec[predicted_index])
                        - float(stream.timestamps_sec[target_index_value])
                    )
                )
            factor_total += 1
            for name in EVENT_FACTORS:
                predicted = int(
                    torch.tensor(
                        prediction.factor_log_probability[name]
                    ).argmax()
                )
                target_value = int(stream.factor_targets[name][target.onset, target.role])
                factor_correct[name] += int(predicted == target_value)
    segment = {}
    for threshold, value in counts.items():
        precision = value["tp"] / max(value["predicted"], 1)
        recall = value["tp"] / max(value["target"], 1)
        segment[f"F1@{int(100 * threshold)}"] = {
            **value,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        }
    return {
        "segment": segment,
        "matched_boundary_absolute_error_sec_at_iou_10": {
            name: {
                "median": float(np.median(values)) if values else None,
                "mean": float(np.mean(values)) if values else None,
                "support": len(values),
            }
            for name, values in boundary_error.items()
        },
        "matched_event_factor_accuracy_at_iou_10": {
            name: factor_correct[name] / max(factor_total, 1)
            for name in EVENT_FACTORS
        },
        "matched_events_for_semantics": factor_total,
    }
