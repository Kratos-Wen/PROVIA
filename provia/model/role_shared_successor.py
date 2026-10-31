"""Role-preserving successor co-reference with atomic completion effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from torch import Tensor


from .successor_posterior import SuccessorCoreferencePosterior




@dataclass(frozen=True)
class RoleCoreferencedSuccessorObservation:
    """Observed semantics and two sources of successor evidence."""

    factor_probability: Mapping[str, Tensor]
    learned_node_log_bayes_factor: Tensor
    endpoint_effect_log_bayes_factor: Tensor | None


@dataclass(frozen=True)
class RoleCoreferencedSuccessorLaw:
    """Exact branch law conditioned on one role's completed observation."""

    observation: RoleCoreferencedSuccessorObservation
    posterior: SuccessorCoreferencePosterior
    shared_mixed: Tensor
    shared_reference: Tensor
    independent_mixed: Tensor
    independent_reference: Tensor
    shared_node_log_bayes_factor: Tensor
    independent_node_log_bayes_factor: Tensor | None


__all__ = [
    "RoleCoreferencedSuccessorLaw",
    "RoleCoreferencedSuccessorObservation",
]
