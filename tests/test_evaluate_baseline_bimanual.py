from __future__ import annotations

import torch

from provia.baselines.run_bimanual import _role_window
from provia.evaluation.evaluate_baseline_bimanual import _prefix_windows


class _Stream:
    def __init__(self, global_feature: torch.Tensor, role_feature: torch.Tensor) -> None:
        self.global_features = global_feature
        self.role_features = role_feature


def test_vectorized_prefix_windows_equal_frozen_role_window_definition() -> None:
    generator = torch.Generator().manual_seed(7)
    global_feature = torch.randn(9, 5, generator=generator)
    role_feature = torch.randn(9, 2, 5, generator=generator)
    frame = torch.tensor((0, 2, 5, 8), dtype=torch.long)
    role = torch.tensor((0, 1, 0, 1), dtype=torch.long)
    actual = _prefix_windows(global_feature, role_feature, frame, role, 4)
    stream = _Stream(global_feature, role_feature)
    expected = torch.stack(
        [
            _role_window(stream, int(end), int(hand), 4)  # type: ignore[arg-type]
            for end, hand in zip(frame, role)
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
