"""Operating points, delays and checkpoint metrics of the detection scores.

The commitment score is emitted at the frozen observer's predicted interaction
onset.  When an end observation is available, a second score refines the same
event hypothesis at its predicted end.  In the completion model this is also
the sole checkpoint at which a normal event may update the accepted procedure
frontier.  The free-response curve counts each non-error candidate warning as
one false alarm and each native error interval at most once.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from provia.data.streaming_events import CORRECTNESS_INDEX
from provia.training.completion_metrics import _binary_metrics


DEFAULT_FALSE_ALARM_BUDGETS = (0.1, 0.5, 1.0, 2.0)


def _threshold_points(
    false_alarm_score: np.ndarray,
    error_event_score: np.ndarray,
    duration_minutes: float,
) -> tuple[tuple[float, float, float], ...]:
    """Enumerate the exact event-recall/false-alarm operating curve."""

    false_alarm_score = np.asarray(false_alarm_score, dtype=np.float64)
    error_event_score = np.asarray(error_event_score, dtype=np.float64)
    if (
        false_alarm_score.ndim != 1
        or error_event_score.ndim != 1
        or not len(error_event_score)
        or duration_minutes <= 0
        or not np.isfinite(false_alarm_score).all()
        or not np.isfinite(error_event_score).all()
    ):
        raise ValueError("free-response inputs must be finite non-empty vectors")
    thresholds = np.unique(
        np.concatenate(
            (
                np.asarray((np.inf, -np.inf), dtype=np.float64),
                false_alarm_score,
                error_event_score,
            )
        )
    )[::-1]
    return tuple(
        (
            float(np.sum(false_alarm_score >= threshold) / duration_minutes),
            float(np.mean(error_event_score >= threshold)),
            float(threshold),
        )
        for threshold in thresholds
    )


def _operating_points(
    points: Sequence[tuple[float, float, float]],
    budgets: Sequence[float] = DEFAULT_FALSE_ALARM_BUDGETS,
) -> dict[str, dict[str, float]]:
    result = {}
    for budget in budgets:
        admissible = [point for point in points if point[0] <= budget]
        if not admissible:
            raise RuntimeError("the infinite threshold must satisfy every budget")
        false_alarms, recall, threshold = max(
            admissible, key=lambda item: (item[1], -item[0], item[2])
        )
        result[str(budget)] = {
            "event_recall": recall,
            "false_alarms_per_minute": false_alarms,
            "validation_threshold": threshold,
        }
    return result


def _operating_points_at_frozen_thresholds(
    false_alarm_score: np.ndarray,
    error_event_score: np.ndarray,
    duration_minutes: float,
    thresholds: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    """Evaluate preselected validation thresholds without reselection.

    The keys identify the target validation false-alarm budgets.  The achieved
    false-alarm rate on the query split is deliberately allowed to differ from
    that target; this is the deployment quantity that must be reported.
    """

    false_alarm_score = np.asarray(false_alarm_score, dtype=np.float64)
    error_event_score = np.asarray(error_event_score, dtype=np.float64)
    if (
        false_alarm_score.ndim != 1
        or error_event_score.ndim != 1
        or not len(error_event_score)
        or duration_minutes <= 0
        or not np.isfinite(false_alarm_score).all()
        or not np.isfinite(error_event_score).all()
        or not thresholds
    ):
        raise ValueError("frozen free-response inputs are malformed")
    result = {}
    for budget, raw_threshold in thresholds.items():
        threshold = float(raw_threshold)
        if np.isnan(threshold):
            raise ValueError("a frozen operating threshold cannot be NaN")
        result[str(budget)] = {
            "event_recall": float(np.mean(error_event_score >= threshold)),
            "false_alarms_per_minute": float(
                np.sum(false_alarm_score >= threshold) / duration_minutes
            ),
            "validation_threshold": threshold,
        }
    return result


def _free_response_false_alarm_scores(
    candidate_score: np.ndarray,
    assignment: np.ndarray,
    native_mode: np.ndarray,
) -> np.ndarray:
    """Keep at most one true warning for each native error event.

    Every insertion, every warning assigned to a non-error event, and every
    duplicate candidate around an error event is a false alarm.  Selecting the
    maximum-scoring candidate per native error is threshold invariant: whenever
    any candidate for that error crosses a threshold, the selected maximum also
    crosses it.
    """

    score = np.asarray(candidate_score, dtype=np.float64)
    assigned_native = np.asarray(assignment, dtype=np.int64)
    mode = np.asarray(native_mode, dtype=np.int64)
    if (
        score.ndim != 1
        or assigned_native.shape != score.shape
        or mode.ndim != 1
        or not np.isfinite(score).all()
        or np.any(assigned_native < -1)
        or np.any(assigned_native >= len(mode))
    ):
        raise ValueError("candidate assignment for free response is malformed")
    true_warning = np.zeros(len(score), dtype=bool)
    for native_index in np.flatnonzero(
        mode == CORRECTNESS_INDEX["error"]
    ):
        candidates = np.flatnonzero(assigned_native == native_index)
        if len(candidates):
            # np.argmax deterministically preserves the earliest candidate on
            # a score tie, but the false-alarm count is identical for all ties.
            best = candidates[int(np.argmax(score[candidates]))]
            true_warning[best] = True
    return score[~true_warning]


def _delay_summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "detections": int(len(values)),
        "mean_seconds": float(values.mean()) if len(values) else None,
        "median_seconds": float(np.median(values)) if len(values) else None,
        "p90_seconds": float(np.quantile(values, 0.9)) if len(values) else None,
    }


def _checkpoint_metrics(
    *,
    candidate_score: np.ndarray,
    candidate_target: np.ndarray,
    origin_score: np.ndarray,
    origin_target: np.ndarray,
    native_score: np.ndarray,
    native_mode: np.ndarray,
    covered: np.ndarray,
    delay: np.ndarray,
    false_alarm_score: np.ndarray,
    native_categories: Sequence[tuple[str, ...]],
    duration_minutes: float,
    detection_options: Sequence[np.ndarray] | None = None,
    delay_reference: str = "native_onset",
    frozen_operating_thresholds: Mapping[str, float] | None = None,
    include_origin_metrics: bool = True,
) -> dict[str, object]:
    """Compute one checkpoint's metrics under an identical event assignment."""

    error = native_mode == CORRECTNESS_INDEX["error"]
    correction = native_mode == CORRECTNESS_INDEX["correction"]
    if frozen_operating_thresholds is None:
        points = _threshold_points(
            false_alarm_score,
            native_score[error],
            duration_minutes,
        )
        operating = _operating_points(points)
        threshold_source = "selected on validation free-response curve"
    else:
        operating = _operating_points_at_frozen_thresholds(
            false_alarm_score,
            native_score[error],
            duration_minutes,
            frozen_operating_thresholds,
        )
        threshold_source = "frozen validation operating-point artifact"

    delay_result = {}
    recovery_result = {}
    category_result: dict[str, dict[str, float | int]] = {}
    for budget, values in operating.items():
        threshold = values["validation_threshold"]
        detected = error & covered & (native_score >= threshold)
        if detection_options is None:
            detected_delay = delay[detected]
        else:
            if len(detection_options) != len(native_score):
                raise ValueError("detection options must align with native events")
            values_at_threshold = []
            for index in np.flatnonzero(detected):
                options = np.asarray(
                    detection_options[int(index)], dtype=np.float64
                )
                if options.ndim != 2 or options.shape[1:] != (2,):
                    raise ValueError(
                        "each detection option must contain score and delay"
                    )
                eligible = options[options[:, 0] >= threshold, 1]
                if len(eligible):
                    values_at_threshold.append(float(eligible.min()))
            detected_delay = np.asarray(values_at_threshold, dtype=np.float64)
        delay_result[budget] = _delay_summary(detected_delay)
        covered_correction = correction & covered
        recovery_result[budget] = {
            "support": int(correction.sum()),
            "covered": int(covered_correction.sum()),
            "false_alarm_rate": float(
                np.mean(native_score[correction] >= threshold)
            )
            if correction.any()
            else float("nan"),
            "covered_only_false_alarm_rate": float(
                np.mean(native_score[covered_correction] >= threshold)
            )
            if covered_correction.any()
            else float("nan"),
        }
        for name in sorted({value for row in native_categories for value in row}):
            category = np.asarray(
                [name in row for row in native_categories], dtype=bool
            ) & error
            if not category.any():
                continue
            entry = category_result.setdefault(
                name, {"support": int(category.sum())}
            )
            entry[f"recall_at_{budget}_false_alarms_per_minute"] = float(
                np.mean(native_score[category] >= threshold)
            )

    if not delay_reference or not delay_reference.replace("_", "").isalnum():
        raise ValueError("delay reference must be a non-empty identifier")
    result = {
        "native_event": _binary_metrics(
            error.astype(np.int64), native_score, 0.5
        ),
        "decoded_candidate": _binary_metrics(
            candidate_target, candidate_score, 0.5
        ),
        "native_event_coverage": float(covered.mean()),
        "native_error_event_coverage": float(covered[error].mean()),
        "native_correction_event_coverage": float(covered[correction].mean())
        if correction.any()
        else float("nan"),
        "free_response": {
            "threshold_source": threshold_source,
            "operating_points": operating,
        },
        f"delay_relative_to_{delay_reference}": delay_result,
        "recovery_event_false_alarm_rate": recovery_result,
        "per_error_type": category_result,
    }
    if include_origin_metrics:
        result["origin"] = _binary_metrics(origin_target, origin_score, 0.5)
    return result
