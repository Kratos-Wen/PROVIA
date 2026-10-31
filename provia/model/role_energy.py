"""Causal evidence with joint bimanual procedure transport.

Each hand keeps a fixed factual-history slot.  Before interaction onset, a
candidate is read against the current procedure belief without changing it.
At onset, all insertion/normal/error/correction hypotheses are retained, but
only the normal branch is transported by the raw sub-stochastic event
operator.  This is a probabilistic reservation: later visual-effect evidence
can reweight the retained branches, so an early normal hypothesis is not an
irreversible decision.  Closing the event releases its lifecycle without a
second procedure transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from provia.observer.factors import EVENT_FACTORS, EVENT_ROLES

from .mode_filter import EXPLANATION_NAMES
from .observer_state import CausalWindowEncoder


@dataclass(frozen=True)
class RoleStep:
    """One active role event and its immutable onset step."""

    role: int
    legal_probability: Tensor
    neutral_probability: Tensor
    node_match_probability: Tensor
    effect_log_bayes_factor: Tensor | None


# Lifecycle states are sufficient statistics for the current event
# explanation.  Keeping the inactive category explicit permits the same
# reduction to expose the non-active hand's execution-mode posterior.
class RolePrefixEnergy(nn.Module):
    """Shared causal cumulative likelihood with fixed left/right history slots."""

    def __init__(
        self,
        input_dim: int,
        factor_labels: Mapping[str, Sequence[str]],
        *,
        model_dim: int = 192,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or model_dim <= 0:
            raise ValueError("role evidence dimensions must be positive")
        if tuple(factor_labels) != EVENT_FACTORS:
            raise ValueError(f"factor labels must follow {EVENT_FACTORS}")
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.factor_labels = {
            name: tuple(str(value) for value in factor_labels[name])
            for name in EVENT_FACTORS
        }
        self.window_encoder = CausalWindowEncoder(
            self.input_dim, model_dim=self.model_dim
        )
        self.factor_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(
                    len(self.factor_labels[name]), self.model_dim
                )
                for name in EVENT_FACTORS
            }
        )
        self.event_probability_map = nn.Sequential(
            nn.Linear(1, self.model_dim), nn.GELU()
        )
        self.role_embedding = nn.Embedding(len(EVENT_ROLES), self.model_dim)
        self.other_role_map = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.query_norm = nn.LayerNorm(self.model_dim)
        self.visual_likelihood = nn.Linear(
            self.model_dim, len(EXPLANATION_NAMES)
        )
        self.history_input = nn.LayerNorm(self.model_dim)
        self.history_cell = nn.GRUCell(self.model_dim, self.model_dim)
        nn.init.zeros_(self.visual_likelihood.weight)
        nn.init.zeros_(self.visual_likelihood.bias)

    def initial_history(self, reference: Tensor) -> Tensor:
        return reference.new_zeros(len(EVENT_ROLES), self.model_dim)

    def semantic_expectation(
        self, factor_probability: Mapping[str, Tensor]
    ) -> Tensor:
        if tuple(factor_probability) != EVENT_FACTORS:
            raise ValueError(f"factor probabilities must follow {EVENT_FACTORS}")
        first = factor_probability[EVENT_FACTORS[0]]
        if first.ndim != 2 or not len(first):
            raise ValueError("role prefix semantics must have a non-empty time axis")
        result = None
        for name in EVENT_FACTORS:
            probability = factor_probability[name]
            expected = (len(first), len(self.factor_labels[name]))
            if probability.shape != expected:
                raise ValueError(f"{name} probability has the wrong shape")
            if (
                not torch.isfinite(probability).all()
                or torch.any(probability < 0)
                or torch.any(probability.sum(dim=-1) <= 0)
            ):
                raise ValueError(f"{name} probability is malformed")
            normalized = probability / probability.sum(dim=-1, keepdim=True)
            value = normalized @ self.factor_embeddings[name].weight
            result = value if result is None else result + value
        if result is None:
            raise RuntimeError("event factor ontology is empty")
        return result

    def update_history(
        self,
        factual_history: Tensor,
        endpoint_hidden: Tensor,
        endpoint_factor_probability: Mapping[str, Tensor],
        role: int,
    ) -> Tensor:
        if factual_history.shape != (len(EVENT_ROLES), self.model_dim):
            raise ValueError("role factual history has the wrong shape")
        if endpoint_hidden.shape != (self.model_dim,):
            raise ValueError("endpoint hidden state has the wrong shape")
        semantic = self.semantic_expectation(
            {
                name: probability.unsqueeze(0)
                for name, probability in endpoint_factor_probability.items()
            }
        )[0]
        value = self.history_input(endpoint_hidden + semantic.to(endpoint_hidden))
        following = factual_history.clone()
        following[int(role)] = self.history_cell(
            value.unsqueeze(0), factual_history[int(role)].unsqueeze(0)
        )[0]
        return following

