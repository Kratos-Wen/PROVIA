"""Intent-conditioned density over causal visual execution paths."""

from __future__ import annotations


import torch
from torch import Tensor



def interval_trajectory_statistics(
    features: Tensor,
    timestamps_sec: Tensor,
    start_indices: Tensor,
    end_indices: Tensor,
) -> Tensor:
    """Return finite-variation statistics on arbitrary causal intervals.

    Every row summarizes ``(start, end]`` by net displacement, a
    time-oriented first moment, coordinatewise quadratic variation, and log
    duration.  Prefix sums keep overlapping intervals exact and avoid temporal
    resampling or phase bins.
    """
    if features.ndim != 2 or not len(features):
        raise ValueError("features must have shape [time,channels]")
    if timestamps_sec.shape != (len(features),):
        raise ValueError("timestamps must align with features")
    if (
        start_indices.ndim != 1
        or end_indices.ndim != 1
        or start_indices.shape != end_indices.shape
    ):
        raise ValueError("interval indices must be aligned vectors")
    if len(timestamps_sec) > 1 and torch.any(
        timestamps_sec[1:] <= timestamps_sec[:-1]
    ):
        raise ValueError("timestamps must be strictly increasing")
    start = start_indices.to(device=features.device, dtype=torch.long)
    end = end_indices.to(device=features.device, dtype=torch.long)
    if not len(start):
        return features.new_empty((0, 3 * features.shape[-1] + 1))
    if (
        torch.any(start < 0)
        or torch.any(end >= len(features))
        or torch.any(end <= start)
    ):
        raise ValueError(
            "each interval must be in range and contain at least one transition"
        )
    steps = end - start

    increments = features[1:] - features[:-1]
    delta_sec = (timestamps_sec[1:] - timestamps_sec[:-1]).to(features)
    velocity = increments / delta_sec[:, None]
    midpoint = 0.5 * (timestamps_sec[1:] + timestamps_sec[:-1]).to(features)

    zero = features.new_zeros((1, features.shape[-1]))
    displacement_prefix = torch.cat((zero, increments.cumsum(dim=0)), dim=0)
    time_moment_prefix = torch.cat(
        (zero, (midpoint[:, None] * increments).cumsum(dim=0)), dim=0
    )
    variation_prefix = torch.cat(
        (zero, velocity.square().cumsum(dim=0)), dim=0
    )

    displacement = displacement_prefix[end] - displacement_prefix[start]
    duration = (timestamps_sec[end] - timestamps_sec[start]).to(features)
    weighted = time_moment_prefix[end] - time_moment_prefix[start]
    oriented_moment = (
        2.0
        * (weighted - timestamps_sec[start].to(features)[:, None] * displacement)
        / duration[:, None]
        - displacement
    )
    quadratic_variation = (
        variation_prefix[end] - variation_prefix[start]
    ) / steps.to(features)[:, None]
    statistic = torch.cat(
        (
            displacement,
            oriented_moment,
            torch.log1p(quadratic_variation),
            duration.log()[:, None],
        ),
        dim=-1,
    )
    if not torch.isfinite(statistic).all():
        raise ValueError("trajectory statistics are not finite")
    return statistic


def trajectory_statistics(
    features: Tensor,
    timestamps_sec: Tensor,
    completion_indices: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return previous states and successive finite-variation statistics.

    Each completion is summarized using only observations since the preceding
    completion.  This is the successive-interval specialization of
    :func:`interval_trajectory_statistics`.
    """

    if completion_indices.ndim != 1:
        raise ValueError("completion indices must be one-dimensional")
    indices = completion_indices.to(device=features.device, dtype=torch.long)
    if not len(indices):
        if features.ndim != 2:
            raise ValueError("features must have shape [time,channels]")
        return (
            features.new_empty((0, features.shape[-1])),
            features.new_empty((0, 3 * features.shape[-1] + 1)),
        )
    if (
        torch.any(indices < 0)
        or torch.any(indices >= len(features))
        or (len(indices) > 1 and torch.any(indices[1:] <= indices[:-1]))
    ):
        raise ValueError("completion indices must be increasing and in range")
    previous = torch.cat((indices.new_zeros(1), indices[:-1]))
    statistic = interval_trajectory_statistics(
        features, timestamps_sec, previous, indices
    )
    return features[previous], statistic

