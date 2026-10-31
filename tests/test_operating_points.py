import numpy as np

from provia.evaluation.operating_points import (
    _checkpoint_metrics,
    _delay_summary,
    _free_response_false_alarm_scores,
    _operating_points,
    _operating_points_at_frozen_thresholds,
    _threshold_points,
)
from provia.data.streaming_events import CORRECTNESS_INDEX


def test_role_free_response_uses_native_errors_once() -> None:
    points = _threshold_points(
        np.asarray([0.9, 0.2]),
        np.asarray([0.8, 0.7]),
        duration_minutes=10.0,
    )
    operating = _operating_points(points, budgets=(0.0, 0.1))
    assert operating["0.0"]["event_recall"] == 0.0
    assert operating["0.1"]["event_recall"] == 1.0
    assert operating["0.1"]["false_alarms_per_minute"] == 0.1


def test_role_delay_summary_preserves_early_detections() -> None:
    summary = _delay_summary(np.asarray([-0.5, 0.5, 1.5]))
    assert summary["detections"] == 3
    assert summary["median_seconds"] == 0.5
    assert summary["mean_seconds"] == 0.5


def test_duplicate_candidates_around_one_error_are_false_alarms() -> None:
    score = np.asarray([0.9, 0.8, 0.7, 0.6])
    assignment = np.asarray([0, 0, 1, -1])
    native_mode = np.asarray(
        [CORRECTNESS_INDEX["error"], CORRECTNESS_INDEX["correct"]]
    )
    false_alarm = _free_response_false_alarm_scores(
        score, assignment, native_mode
    )
    assert np.array_equal(false_alarm, np.asarray([0.8, 0.7, 0.6]))


def test_frozen_operating_threshold_is_not_reselected_on_query() -> None:
    result = _operating_points_at_frozen_thresholds(
        false_alarm_score=np.asarray([0.7, 0.6]),
        error_event_score=np.asarray([0.8, 0.5]),
        duration_minutes=2.0,
        thresholds={"0.1": 0.65},
    )
    assert result["0.1"]["validation_threshold"] == 0.65
    assert result["0.1"]["event_recall"] == 0.5
    # The validation target was 0.1 FA/min, but the achieved query rate is
    # measured rather than silently retuned to the target.
    assert result["0.1"]["false_alarms_per_minute"] == 0.5






def test_checkpoint_delay_is_the_first_threshold_crossing() -> None:
    result = _checkpoint_metrics(
        candidate_score=np.asarray([0.8, 0.75, 0.1]),
        candidate_target=np.asarray([1, 1, 0]),
        origin_score=np.asarray([0.9, 0.8, 0.2]),
        origin_target=np.asarray([1, 1, 0]),
        native_score=np.asarray([0.8, 0.75, 0.1]),
        native_mode=np.asarray(
            [
                CORRECTNESS_INDEX["error"],
                CORRECTNESS_INDEX["error"],
                CORRECTNESS_INDEX["correct"],
            ]
        ),
        covered=np.ones(3, dtype=bool),
        delay=np.asarray([2.0, 0.5, 0.0]),
        false_alarm_score=np.asarray([0.7]),
        native_categories=((), (), ()),
        duration_minutes=10.0,
        detection_options=(
            np.asarray(((0.8, 2.0), (0.75, 1.0))),
            np.asarray(((0.75, 0.5),)),
            np.asarray(((0.1, 0.0),)),
        ),
    )
    delay = result["delay_relative_to_native_onset"]["0.1"]
    assert delay["detections"] == 2
    assert delay["mean_seconds"] == 0.75
