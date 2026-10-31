"""Role-preserving reduction of candidate-scoped predictive-state marks."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor, nn


from .state_mark import CausalFrameEncoder
from .role_intent_mark import (
    RoleIntentConditionedMarkEnergy,
    RoleIntentMarkModel,
)




class RolePredictiveStateMarkEnergy(RoleIntentConditionedMarkEnergy):
    """One shared candidate recurrence with fixed left/right history slots."""

    def __init__(
        self,
        input_dim: int,
        factor_labels: Mapping[str, Sequence[str]],
        *,
        model_dim: int = 192,
    ) -> None:
        super().__init__(input_dim, factor_labels, model_dim=model_dim)
        self.window_encoder = CausalFrameEncoder(
            input_dim, model_dim=self.model_dim
        )
        self.candidate_state = nn.GRU(
            self.model_dim, self.model_dim, batch_first=True
        )
        self.state_norm = nn.LayerNorm(self.model_dim)

    def forward_state_path(
        self,
        encoded_hidden: Tensor,
        factor_probability: Mapping[str, Tensor],
        event_probability: Tensor,
    ) -> Tensor:
        if (
            encoded_hidden.ndim != 2
            or encoded_hidden.shape[-1] != self.model_dim
            or not len(encoded_hidden)
            or not torch.isfinite(encoded_hidden).all()
        ):
            raise ValueError("encoded role candidate is malformed")
        if (
            event_probability.shape != (len(encoded_hidden),)
            or not torch.isfinite(event_probability).all()
            or torch.any((event_probability < 0) | (event_probability > 1))
        ):
            raise ValueError("role event probability must lie in [0,1]")
        semantic = self.semantic_expectation(factor_probability).to(
            encoded_hidden
        )
        event = self.event_probability_map(
            event_probability.to(encoded_hidden).unsqueeze(-1)
        )
        recurrent_input = self.query_norm(
            encoded_hidden + semantic + event
        )
        state, _ = self.candidate_state(recurrent_input.unsqueeze(0))
        return state.squeeze(0)

class RoleStateMarkModel(
    RoleIntentMarkModel
):
    """Asynchronous bimanual predictive states with atomic shared effects."""

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
        super().__init__(
            input_dim,
            factor_labels,
            transition_probability,
            initial_probability,
            genuine_probability,
            model_dim=model_dim,
        )
        self.energy = RolePredictiveStateMarkEnergy(
            input_dim, factor_labels, model_dim=model_dim
        )


__all__ = [
    "RolePredictiveStateMarkEnergy",
    "RoleStateMarkModel",
]
