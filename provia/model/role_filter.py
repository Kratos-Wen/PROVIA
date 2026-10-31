"""Exact bimanual filtering of step candidates.

An event proposal starts in one unresolved lifecycle state.  At every causal
prefix a shared scalar clock either keeps it unresolved or makes its
explanation identifiable.  A decision branches into insertion, normal, error,
or correction under one normalized state transition.  Only the normal branch
applies the current soft procedure operator; all decided branches continue to
receive later visual evidence and can therefore be reweighted.

For two active hands, the two same-frame operators are averaged in both
orders.  This is exact marginalization of an unobserved within-frame order,
not a fixed left-first convention.  The implementation never uses an endpoint
semantic mark at an earlier prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn


from .mode_filter import (
    ERROR,
    EXPLANATION_NAMES,
    MODE_NAMES,
    ReliabilityAwareModeFilter,
)


INACTIVE = 0
UNRESOLVED = 1
INSERTION = 2
NORMAL_LIFECYCLE = 3
ERROR_LIFECYCLE = 4
CORRECTION_LIFECYCLE = 5
ROLE_LIFECYCLE_NAMES = (
    "inactive",
    "unresolved",
    "insertion",
    "normal",
    "error",
    "correction",
)


def explanation_lifecycle(explanation: int) -> int:
    if not 0 <= int(explanation) < len(EXPLANATION_NAMES):
        raise ValueError("unknown explanation")
    return INSERTION + int(explanation)


@dataclass(frozen=True)
class RoleFilterSummary:
    """Posterior summary of one hand at one causal prefix."""

    explanation_probability: Tensor
    anomaly_probability: Tensor
    commitment_probability: Tensor
    unresolved_probability: Tensor


class RoleFilter(nn.Module):
    """Exact joint frontier/mode/lifecycle recursion for two hands."""

    def __init__(
        self,
        transition_probability: Tensor | Sequence[Sequence[float]],
        initial_probability: Tensor | Sequence[float],
        genuine_probability: Tensor | float,
    ) -> None:
        super().__init__()
        template = ReliabilityAwareModeFilter(
            transition_probability, initial_probability, genuine_probability
        )
        self.register_buffer(
            "transition_probability",
            template.transition_probability.detach().clone(),
        )
        self.register_buffer(
            "initial_probability", template.initial_probability.detach().clone()
        )
        self.register_buffer(
            "prior_genuine_probability",
            template.prior_genuine_probability.detach().clone(),
        )

    def initial_belief(self, procedure_probability: Tensor) -> Tensor:
        if (
            procedure_probability.ndim != 1
            or not torch.isfinite(procedure_probability).all()
            or torch.any(procedure_probability < 0)
            or procedure_probability.sum() <= 0
        ):
            raise ValueError("initial procedure probability is malformed")
        procedure = procedure_probability / procedure_probability.sum()
        modes = self.initial_probability.to(procedure)
        belief = procedure.new_zeros(
            len(procedure),
            len(MODE_NAMES),
            len(MODE_NAMES),
            len(ROLE_LIFECYCLE_NAMES),
            len(ROLE_LIFECYCLE_NAMES),
        )
        belief[:, :, :, INACTIVE, INACTIVE] = (
            procedure[:, None, None] * modes[None, :, None] * modes[None, None, :]
        )
        return belief

    @staticmethod
    def _validate(probability: Tensor, *, normalize: bool = True) -> Tensor:
        expected = (
            len(MODE_NAMES),
            len(MODE_NAMES),
            len(ROLE_LIFECYCLE_NAMES),
            len(ROLE_LIFECYCLE_NAMES),
        )
        if probability.ndim != 5 or probability.shape[1:] != expected:
            raise ValueError(
                "bimanual belief must have shape [states,3,3,6,6]"
            )
        if (
            not torch.isfinite(probability).all()
            or torch.any(probability < 0)
            or probability.sum() <= 0
        ):
            raise ValueError("bimanual belief is malformed")
        return probability / probability.sum() if normalize else probability

    @staticmethod
    def _canonical(probability: Tensor, role: int) -> Tensor:
        if int(role) == 0:
            return probability
        if int(role) == 1:
            return probability.permute(0, 2, 1, 4, 3)
        raise ValueError("role must select left or right")

    @staticmethod
    def _restore(probability: Tensor, role: int) -> Tensor:
        return RoleFilter._canonical(probability, role)

    def summarize(self, probability: Tensor, role: int) -> RoleFilterSummary:
        canonical = self._canonical(self._validate(probability), role)
        lifecycle = canonical.sum(dim=(0, 1, 2, 4))
        explanation = torch.stack(
            tuple(
                lifecycle[explanation_lifecycle(index)]
                for index in range(len(EXPLANATION_NAMES))
            )
        )
        return RoleFilterSummary(
            explanation_probability=explanation,
            anomaly_probability=explanation[1 + ERROR],
            commitment_probability=explanation.sum(),
            unresolved_probability=lifecycle[UNRESOLVED],
        )
