"""Exact model averaging over successor co-reference structure.

The learned relation to a legal successor can either identify the same latent
successor that transports accepted procedure state or provide an independent
intent-set likelihood.  Neither structural assumption is selected from a
validation set.  This module assigns the two hypotheses equal prior mass,
marginalizes them in the normal observation law, and transports each
branch-conditioned normal mass with its own exact kernel.

This is Bayesian model averaging inside one causal state-space model.  It is
not score fusion: insertion, error, correction, and the two normal hypotheses
are normalized together before any accepted-state update.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


from .successor_posterior import SuccessorCoreferencePosterior




@dataclass(frozen=True)
class CoreferencedSuccessorLaw:
    """Normal likelihood and conditional kernels for every causal prefix."""

    posterior: SuccessorCoreferencePosterior
    shared_mixed: Tensor
    shared_reference: Tensor
    independent_mixed: Tensor
    independent_reference: Tensor
    shared_endpoint_node_log_bayes_factor: Tensor
    independent_endpoint_node_log_bayes_factor: Tensor | None


__all__ = [
    "CoreferencedSuccessorLaw",
]
