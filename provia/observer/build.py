"""One streaming step segmentation model for every actor stream.

The single-actor completion process (``completion_process``) is applied to
each hand-object stream separately and exposed through the marked-renewal
interface consumed by the two-hand filter.  Each hand always carries an open
step (interaction phase from the first frame): the open step closes at the
decoded completion and the next one opens at once (``RENEW_ONSET``), so the
candidates tile each hand stream exactly as the single-actor partition does.
The completion intensity is emitted one frame late so that the completion frame
belongs to the step it completes; the mark of a closed step is read at its end
frame (``end-frame`` statistic).  No parameter of the renewal observer is used.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from .marked_renewal import MarkedRenewalRoleObserver

MARKED_RENEWAL_ARCHITECTURE = "marked_renewal_observer"
ROLE_EVENT_OBSERVER_ARCHITECTURES = frozenset(
    {MARKED_RENEWAL_ARCHITECTURE}
)
def build_role_event_observer(
    metadata: Mapping[str, object],
    input_dim: int,
    factor_labels: Mapping[str, Sequence[str]],
) -> MarkedRenewalRoleObserver:
    """Instantiate the observer class certified by checkpoint metadata."""

    architecture = metadata.get("observer_architecture")
    if architecture == MARKED_RENEWAL_ARCHITECTURE:
        return MarkedRenewalRoleObserver(
            input_dim,
            factor_labels,
            torch.tensor(metadata["base_rates_per_second"]),  # type: ignore[arg-type]
            torch.tensor(metadata["initial_phase_probability"]),  # type: ignore[arg-type]
        )
    raise ValueError(f"unknown role event observer architecture: {architecture}")
