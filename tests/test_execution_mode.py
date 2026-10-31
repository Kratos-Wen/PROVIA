from types import SimpleNamespace

from provia.data.streaming_events import CORRECTNESS_INDEX
from provia.training.role_transport_fit import (
    _fit_role_execution_mode_dynamics,
)


TRANSITION = (
    (0.8, 0.15, 0.05),
    (0.2, 0.3, 0.5),
    (0.8, 0.1, 0.1),
)












def test_role_dynamics_exclude_cross_role_transitions():
    interval = lambda role, onset, correctness: SimpleNamespace(
        role=role,
        onset=onset,
        start=onset,
        correctness=CORRECTNESS_INDEX[correctness],
    )
    stream = SimpleNamespace(
        intervals=(
            interval(0, 0, "correct"),
            interval(1, 1, "correct"),
            interval(0, 2, "error"),
            interval(1, 3, "correct"),
            interval(0, 4, "correction"),
        )
    )
    _, _, audit = _fit_role_execution_mode_dynamics((stream,))
    assert audit["initial_counts"] == [2, 0, 0]
    assert audit["transition_counts"] == [
        [1, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ]
