"""Causal finite-variation paths for asynchronous role events."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from provia.observer.factors import EVENT_ROLES
from provia.observer.trajectory_statistics import interval_trajectory_statistics


def role_interval_trajectory_statistics(
    role_features: Tensor,
    timestamps_sec: Tensor,
    event_roles: Sequence[str],
    start_indices: Tensor,
    end_indices: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Summarize each role event over its own observed execution interval.

    Unlike :func:`role_trajectory_statistics`, this statistic is attached to
    the current event: it starts at that event's interaction onset and ends at
    its causally decoded completion.  Interleaved hands and overlapping
    left/right intervals remain independent.
    """

    if role_features.ndim != 3 or role_features.shape[1] != len(EVENT_ROLES):
        raise ValueError("role features must have shape [time,roles,channels]")
    if timestamps_sec.shape != (len(role_features),):
        raise ValueError("timestamps must align with role features")
    if (
        start_indices.shape != (len(event_roles),)
        or end_indices.shape != (len(event_roles),)
    ):
        raise ValueError("event roles and interval indices must align")
    unknown = set(event_roles) - set(EVENT_ROLES)
    if unknown:
        raise ValueError(f"unknown event roles: {sorted(unknown)}")
    dimension = int(role_features.shape[-1])
    previous = role_features.new_zeros((len(event_roles), dimension))
    statistic = role_features.new_zeros((len(event_roles), 3 * dimension + 1))
    start = start_indices.to(dtype=torch.long)
    end = end_indices.to(dtype=torch.long)
    valid = (start >= 0) & (end > start) & (end < len(role_features))
    for role_index, role_name in enumerate(EVENT_ROLES):
        positions = [
            index
            for index, value in enumerate(event_roles)
            if value == role_name and bool(valid[index])
        ]
        if not positions:
            continue
        selected = torch.tensor(positions, dtype=torch.long)
        role_start = start.index_select(0, selected)
        role_end = end.index_select(0, selected)
        previous[selected] = role_features[role_start, role_index]
        statistic[selected] = interval_trajectory_statistics(
            role_features[:, role_index],
            timestamps_sec,
            role_start,
            role_end,
        )
    return previous, statistic, valid
