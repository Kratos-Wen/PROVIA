"""Role-preserving intent-set marginal predictive states.

Both hands share one candidate-state update and one successor relation.  Each
active hand retains its own causal state and fixed factual-history slot, while
the latent legal successor is marginalized with the role-specific expert law.
Concurrent completions remain order-marginalized by the parent filter.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from provia.observer.factors import EVENT_FACTORS, EVENT_ROLES

from .interval_transport import ContinuationTransport
from .role_state_mark import (
    RolePredictiveStateMarkEnergy,
    RoleStateMarkModel,
)




class RoleIntentSetPredictiveStateMarkEnergy(
    RolePredictiveStateMarkEnergy
):
    """Shared bimanual candidate state with exact successor-set readout."""

    def __init__(
        self,
        input_dim: int,
        factor_labels: Mapping[str, Sequence[str]],
        *,
        model_dim: int = 192,
    ) -> None:
        super().__init__(input_dim, factor_labels, model_dim=model_dim)
        self.node_norm = nn.LayerNorm(self.model_dim)
        self.node_key = nn.Linear(
            self.model_dim, self.model_dim, bias=False
        )

    def _node_semantic_key(
        self, transport: ContinuationTransport
    ) -> Tensor:
        reference = self.node_key.weight
        node = reference.new_zeros(
            len(transport.language.node_ids), self.model_dim
        )
        for name in EVENT_FACTORS:
            probability = getattr(
                transport, f"predictive_node_{name}_probability"
            ).to(reference)
            node = node + probability @ self.factor_embeddings[name].weight
        return self.node_key(self.node_norm(node))

    def build_frontier_key(
        self,
        transport: ContinuationTransport,
        role: int,
    ) -> Tensor:
        if not 0 <= int(role) < len(EVENT_ROLES):
            raise ValueError("role must select a fixed left/right slot")
        reference = self.frontier_key.weight
        structural = torch.stack(
            (
                getattr(
                    transport,
                    f"frontier_successor_support_{int(role)}",
                ),
                transport.frontier_terminal_probability,
                transport.frontier_closure_entropy,
            ),
            dim=-1,
        ).to(reference)
        frontier = self.frontier_key(
            self.frontier_norm(self.frontier_structural_map(structural))
        )
        return torch.cat((frontier, self._node_semantic_key(transport)), dim=0)

class RoleIntentSetModel(
    RoleStateMarkModel
):
    """Bimanual exact association/intent marginal predictive state."""

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
        self.energy = RoleIntentSetPredictiveStateMarkEnergy(
            input_dim, factor_labels, model_dim=model_dim
        )


__all__ = [
    "RoleIntentSetPredictiveStateMarkEnergy",
    "RoleIntentSetModel",
]
