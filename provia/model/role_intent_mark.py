"""Role-preserving intent-conditioned marks with atomic effect transport.

Candidate occurrence and candidate meaning are separate random variables.
For every active hand, a shared observation--successor likelihood predicts the
candidate mark conditional on each viable procedure frontier.  Concurrent
left/right marks are marginalized without an imposed order.  A completed
normal mark transports the shared procedure state exactly once; an unfinished
mark can condition the current posterior but cannot commit progress.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor, nn


from .mode_filter import EXPLANATION_NAMES, MODE_NAMES
from .bimanual_filter import EffectCommittedRoleFilter
from .step_effects import ExpertEffectPosterior
from .interval_transport import ContinuationTransport
from .role_filter import (
    INACTIVE,
    INSERTION,
    RoleFilterSummary,
    explanation_lifecycle,
)
from .role_energy import RolePrefixEnergy




class RoleIntentConditionedMarkEnergy(RolePrefixEnergy):
    """One shared observation--expert-intent likelihood for both roles."""

    def __init__(
        self,
        input_dim: int,
        factor_labels: Mapping[str, Sequence[str]],
        *,
        model_dim: int = 192,
    ) -> None:
        super().__init__(input_dim, factor_labels, model_dim=model_dim)
        del self.visual_likelihood
        self.frontier_structural_map = nn.Sequential(
            nn.LayerNorm(3), nn.Linear(3, self.model_dim)
        )
        self.frontier_norm = nn.LayerNorm(self.model_dim)
        self.frontier_key = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.mark_query = nn.Linear(
            self.model_dim,
            len(EXPLANATION_NAMES) * self.model_dim,
            bias=False,
        )
        self.mark_base = nn.Linear(self.model_dim, len(EXPLANATION_NAMES))
        nn.init.zeros_(self.mark_query.weight)
        nn.init.zeros_(self.mark_base.weight)
        nn.init.zeros_(self.mark_base.bias)

class RoleIntentConditionedMarkFilter(EffectCommittedRoleFilter):
    """Exact joint mark posterior and completion-only procedure effects."""

    def _mark_role_unnormalized(
        self,
        previous_probability: Tensor,
        role: int,
        log_likelihood: Tensor,
    ) -> Tensor:
        previous = self._validate(previous_probability, normalize=False)
        canonical = self._canonical(previous, role)
        if canonical[:, :, :, 1:, :].sum() > 2e-6:
            raise ValueError("the selected role already has a candidate mark")
        likelihood = self._validate_mark(log_likelihood, len(canonical)).to(
            canonical
        )
        genuine = self.prior_genuine_probability.to(canonical).reshape(())
        insertion_log = torch.log1p(-genuine) + likelihood[:, 0]
        mode_log = torch.log(genuine) + likelihood[:, 1:]
        scale = torch.max(torch.cat((insertion_log, mode_log.flatten())))
        source = canonical[:, :, :, INACTIVE, :]
        following = torch.zeros_like(canonical)
        following[:, :, :, INSERTION, :] = source * torch.exp(
            insertion_log[:, None, None, None] - scale
        )
        transition = self.transition_probability.to(canonical)
        for destination_mode in range(len(MODE_NAMES)):
            destination = (
                source
                * transition[:, destination_mode][None, :, None, None]
            ).sum(dim=1)
            destination = destination * torch.exp(
                mode_log[:, destination_mode, None, None] - scale
            )
            following[
                :, destination_mode, :, explanation_lifecycle(1 + destination_mode), :
            ] = destination
        restored = self._restore(following, role)
        if restored.sum() <= 0 or not torch.isfinite(restored).all():
            raise ValueError("role mark update has no finite support")
        return restored

    def mark_joint(
        self,
        previous_probability: Tensor,
        log_likelihood: Mapping[int, Tensor],
    ) -> Tensor:
        previous = self._validate(previous_probability)
        if not log_likelihood or any(role not in {0, 1} for role in log_likelihood):
            raise ValueError("joint marks must contain one or both roles")

        def apply(source: Tensor, role: int) -> Tensor:
            return self._mark_role_unnormalized(
                source, role, log_likelihood[role]
            )

        roles = sorted(log_likelihood)
        if len(roles) == 1:
            following = apply(previous, roles[0])
        elif len(roles) == 2:
            following = 0.5 * (
                apply(apply(previous, roles[0]), roles[1])
                + apply(apply(previous, roles[1]), roles[0])
            )
        else:
            raise ValueError("only two roles are defined")
        return self._validate(following)

class RoleIntentMarkModel(nn.Module):
    """Shared bimanual marked-renewal model with atomic state updates."""

    def __init__(
        self,
        input_dim: int,
        factor_labels: Mapping[str, Sequence[str]],
        transition_probability: Tensor | Sequence[Sequence[float]],
        initial_probability: Tensor | Sequence[float],
        genuine_probability: Tensor | float,
        *,
        model_dim: int = 192,
    ) -> None:
        super().__init__()
        self.energy = RoleIntentConditionedMarkEnergy(
            input_dim, factor_labels, model_dim=model_dim
        )
        self.filter = RoleIntentConditionedMarkFilter(
            transition_probability, initial_probability, genuine_probability
        )

    @staticmethod
    def node_effect_log_bayes_factor(
        effect_statistic: Tensor,
        expert_effect: ExpertEffectPosterior,
        transport: ContinuationTransport,
    ) -> Tensor:
        if expert_effect.node_ids != tuple(transport.language.node_ids):
            raise ValueError("expert effects and continuation nodes disagree")
        if effect_statistic.ndim != 1:
            raise ValueError("effect statistic must be one endpoint vector")
        return expert_effect.node_log_bayes_factor(effect_statistic)[0]

    def condition_for_readout(
        self,
        previous_probability: Tensor,
        observations: Mapping[int, tuple[Mapping[str, Tensor], Tensor | None]],
        transport: ContinuationTransport,
    ) -> Tensor:
        belief = previous_probability
        for role in sorted(observations):
            factors, effect = observations[role]
            mixed, _, _, _ = transport.predictive_reference_statistics(
                factors,
                role,
                node_effect_log_bayes_factor=effect,
            )
            belief = self.filter.condition_normal(belief, role, mixed)
        return belief

__all__ = [
    "RoleIntentConditionedMarkEnergy",
    "RoleIntentConditionedMarkFilter",
    "RoleIntentMarkModel",
    "RoleFilterSummary",
]
