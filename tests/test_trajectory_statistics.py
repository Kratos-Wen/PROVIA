import math

import pytest
import torch

from provia.observer.role_statistics import (role_interval_trajectory_statistics)
from provia.observer.trajectory_statistics import (interval_trajectory_statistics, trajectory_statistics)


def test_trajectory_statistics_are_causal_and_capture_path_shape() -> None:
    features = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [2.0, 2.0]]
    )
    timestamps = torch.arange(5, dtype=torch.float32)
    indices = torch.tensor([2, 4])
    previous, statistic = trajectory_statistics(features, timestamps, indices)
    assert torch.equal(previous, features[[0, 2]])
    assert statistic.shape == (2, 7)
    assert torch.allclose(statistic[:, :2], torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
    assert torch.allclose(statistic[:, 2:4], torch.zeros(2, 2))
    assert torch.allclose(statistic[:, -1], torch.full((2,), math.log(2.0)))

    extended = torch.cat((features, torch.tensor([[100.0, -100.0]])), dim=0)
    extended_time = torch.arange(6, dtype=torch.float32)
    _, extended_statistic = trajectory_statistics(extended, extended_time, indices)
    assert torch.equal(statistic, extended_statistic)




def test_trajectory_statistics_reject_zero_length_path() -> None:
    with pytest.raises(ValueError, match="at least one transition"):
        trajectory_statistics(
            torch.zeros(2, 3), torch.tensor([0.0, 1.0]), torch.tensor([0])
        )




def test_current_event_interval_statistics_do_not_leak_previous_event_motion() -> None:
    features = torch.tensor(
        [
            [[0.0], [10.0]],
            [[4.0], [11.0]],
            [[5.0], [12.0]],
            [[6.0], [16.0]],
            [[7.0], [17.0]],
        ]
    )
    timestamp = torch.arange(5, dtype=torch.float32)
    previous, statistic, valid = role_interval_trajectory_statistics(
        features,
        timestamp,
        ("left", "right"),
        torch.tensor((1, 2)),
        torch.tensor((4, 4)),
    )
    assert valid.tolist() == [True, True]
    assert previous[:, 0].tolist() == [4.0, 12.0]
    assert statistic[:, 0].tolist() == [3.0, 5.0]
    direct = interval_trajectory_statistics(
        features[:, 0], timestamp, torch.tensor((1,)), torch.tensor((4,))
    )
    assert torch.equal(statistic[0], direct[0])
