"""Conjugate role-wise observation channel for uncertain event streams.

The channel separates two observer errors from procedural deviations.  A
latent canonical event can be deleted before observation, while a decoded
event can be an insertion that leaves procedure state unchanged.  Both laws
are estimated from event-localization supervision only; anomaly labels and
validation metrics are outside the channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from provia.observer.factors import EVENT_FACTORS, EVENT_ROLES


ORIGINS = ("genuine", "insertion")


@dataclass(frozen=True)
class FactorizedRoleObservationChannel:
    """Posterior-predictive edit channel with factorized insertion marks."""

    factor_labels: Mapping[str, tuple[str, ...]]
    missed_event_probability: tuple[float, float]
    origin_probability: tuple[tuple[float, float], tuple[float, float]]
    origin_factor_probability: Mapping[
        str,
        tuple[
            tuple[tuple[float, ...], tuple[float, ...]],
            tuple[tuple[float, ...], tuple[float, ...]],
        ],
    ]

    def validate(self) -> None:
        if tuple(self.factor_labels) != EVENT_FACTORS:
            raise ValueError(f"factor labels must follow {EVENT_FACTORS}")
        if any(
            not values or len(values) != len(set(values))
            for values in self.factor_labels.values()
        ):
            raise ValueError("factor labels must be non-empty and unique")
        missed = np.asarray(self.missed_event_probability, dtype=np.float64)
        origin = np.asarray(self.origin_probability, dtype=np.float64)
        if (
            missed.shape != (len(EVENT_ROLES),)
            or not np.isfinite(missed).all()
            or np.any(missed < 0)
            or np.any(missed > 1)
        ):
            raise ValueError("missed-event probability must lie in [0,1]^2")
        if (
            origin.shape != (len(EVENT_ROLES), len(ORIGINS))
            or not np.isfinite(origin).all()
            or np.any(origin <= 0)
            or not np.allclose(origin.sum(axis=-1), 1.0)
        ):
            raise ValueError("origin probabilities must be positive distributions")
        if tuple(self.origin_factor_probability) != EVENT_FACTORS:
            raise ValueError(f"origin factors must follow {EVENT_FACTORS}")
        for name in EVENT_FACTORS:
            values = np.asarray(
                self.origin_factor_probability[name], dtype=np.float64
            )
            if (
                values.shape
                != (
                    len(EVENT_ROLES),
                    len(ORIGINS),
                    len(self.factor_labels[name]),
                )
                or not np.isfinite(values).all()
                or np.any(values <= 0)
                or not np.allclose(values.sum(axis=-1), 1.0)
            ):
                raise ValueError(f"invalid {name} origin likelihood")
