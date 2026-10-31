"""Intent-conditioned marks for causal procedural event streams.

Event occurrence and event meaning are different random variables.  The
renewal observer determines when a candidate is available; conditional on
that candidate, this module infers whether its mark is an insertion, normal
progress, an execution error, or a correction.  Consequently idle time cannot
accumulate anomaly probability merely because no event has completed.

The mark likelihood is conditioned jointly on the observed causal prefix and
every still-viable expert successor.  It is one state-conditional likelihood,
not a visual score fused with a procedure score.  The exact expert predictive
law remains an analytic normal-likelihood offset, while a shared bilinear
residual learns only observation--intent relations not explained by that law.
All parameters are shared across procedures; task identity enters solely
through the automatically compiled expert continuation language.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from provia.observer.factors import EVENT_FACTORS

from .mode_filter import EXPLANATION_NAMES
from .prefix_energy import PrefixEnergy




class IntentConditionedMarkEnergy(PrefixEnergy):
    """Shared observation--successor likelihood over latent frontiers."""

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
        # The exact zero-residual model is the analytic marked-renewal
        # posterior. Training starts from that auditable reduction.
        nn.init.zeros_(self.mark_query.weight)
        nn.init.zeros_(self.mark_base.weight)
        nn.init.zeros_(self.mark_base.bias)

class IntentConditionedMarkedRenewalCompletionModel(nn.Module):
    """Causal candidate-mark posterior with completion-time state effects."""

    @staticmethod
    def _endpoint_causal_factors(
        prefix_factor_probability: Mapping[str, Tensor],
        endpoint_factor_probability: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        if tuple(prefix_factor_probability) != EVENT_FACTORS or tuple(
            endpoint_factor_probability
        ) != EVENT_FACTORS:
            raise ValueError(f"factor probabilities must follow {EVENT_FACTORS}")
        first = prefix_factor_probability[EVENT_FACTORS[0]]
        if first.ndim != 2 or not len(first):
            raise ValueError("one candidate needs non-empty causal semantics")
        result = {}
        for name in EVENT_FACTORS:
            prefix = prefix_factor_probability[name]
            endpoint = endpoint_factor_probability[name]
            if prefix.ndim != 2 or len(prefix) != len(first):
                raise ValueError("prefix semantic probabilities must align")
            if endpoint.shape != prefix.shape[1:]:
                raise ValueError("endpoint semantic probability has wrong shape")
            result[name] = torch.cat(
                (prefix[:-1], endpoint.to(prefix).unsqueeze(0)), dim=0
            )
        return result
