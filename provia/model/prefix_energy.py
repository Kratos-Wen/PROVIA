"""Cumulative causal evidence with a separate execution-effect clock.

At every prefix the network predicts one normalized cumulative likelihood over
insertion, normal, error, and correction.  Moving from prefix ``t-1`` to ``t``
uses only the likelihood ratio ``L_t / L_{t-1}``; therefore the complete update
sums to ``L_t`` and correlated past frames are never counted repeatedly.
Unlike a survival/hazard construction, a neutral observation reproduces the
same prior at every candidate length and introduces no decision-time prior.

The accepted procedure frontier is read while forming the likelihood but is
transported only at the fully predicted effect completion.  Endpoint semantics
and expert consequence evidence are unavailable to every earlier prefix.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from provia.observer.factors import EVENT_FACTORS

from .mode_filter import EXPLANATION_NAMES
from .observer_state import CausalWindowEncoder


class PrefixEnergy(nn.Module):
    """Cumulative visual likelihood with no learned normative compatibility.

    The event observer, its soft semantics, and the complete factual history
    form one causal observation likelihood.  Procedure feasibility is supplied
    separately by the exact continuation operator, so this module has no SOP
    key, applicability gate, dataset embedding, or post-hoc fusion weight.
    """

    def __init__(
        self,
        input_dim: int,
        factor_labels: Mapping[str, Sequence[str]],
        *,
        model_dim: int = 192,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or model_dim <= 0:
            raise ValueError("prefix energy dimensions must be positive")
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
        self.history_input = nn.LayerNorm(self.model_dim)
        self.history_cell = nn.GRUCell(self.model_dim, self.model_dim)
        self.query_norm = nn.LayerNorm(self.model_dim)
        self.visual_likelihood = nn.Linear(
            self.model_dim, len(EXPLANATION_NAMES)
        )
        # A zero potential is a uniform likelihood and therefore exactly
        # reproduces the state-space prior before learning.
        nn.init.zeros_(self.visual_likelihood.weight)
        nn.init.zeros_(self.visual_likelihood.bias)

    def initial_history(self, reference: Tensor) -> Tensor:
        attention = getattr(self, "attention_history", None)
        if attention is not None:
            attention.reset()
        return reference.new_zeros(self.model_dim)

    def semantic_expectation(
        self, factor_probability: Mapping[str, Tensor]
    ) -> Tensor:
        if tuple(factor_probability) != EVENT_FACTORS:
            raise ValueError(f"factor probabilities must follow {EVENT_FACTORS}")
        first = factor_probability[EVENT_FACTORS[0]]
        if first.ndim not in (1, 2):
            raise ValueError(
                "factor probabilities must describe one or many prefixes"
            )
        batch = first.ndim == 2
        result = None
        for name in EVENT_FACTORS:
            probability = factor_probability[name]
            expected = (
                (len(first), len(self.factor_labels[name]))
                if batch
                else (len(self.factor_labels[name]),)
            )
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
        previous_history: Tensor,
        endpoint_hidden: Tensor,
        endpoint_factor_probability: Mapping[str, Tensor],
    ) -> Tensor:
        if previous_history.shape != (self.model_dim,) or endpoint_hidden.shape != (
            self.model_dim,
        ):
            raise ValueError("factual history update has incompatible dimensions")
        semantic = self.semantic_expectation(endpoint_factor_probability)
        value = self.history_input(endpoint_hidden + semantic)
        attention = getattr(self, "attention_history", None)
        if attention is not None:
            identity = endpoint_factor_probability[EVENT_FACTORS[0]] if getattr(attention, "identity_keyed", False) else None
            return attention.readout(value, identity)
        return self.history_cell(
            value.unsqueeze(0), previous_history.unsqueeze(0)
        )[0]

