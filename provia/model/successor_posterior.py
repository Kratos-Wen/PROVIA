"""Exact Bayesian co-reference between emission and transport successors."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class SuccessorCoreferencePosterior:
    """Marginal likelihood and branch posterior for one normal mark."""

    log_marginal_likelihood: Tensor
    shared_probability: Tensor
    independent_probability: Tensor


__all__ = [
    "SuccessorCoreferencePosterior",
]
