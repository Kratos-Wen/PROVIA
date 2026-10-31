"""Exact intent-set marginalization for causal candidate predictive states.

A procedural frontier generally admits a *set* of next events.  Replacing
that set by its mean embedding is not a probability-preserving operation: the
mean of two distinct legal intentions may resemble neither.  This module
keeps successor identity latent and marginalizes the learned observation
likelihood with the expert-induced successor probabilities.

For a candidate prefix state ``h_t``, frontier ``s``, explanation ``k``, and
legal successor ``z``, one shared relation produces ``ell(h_t,z,k)``.  The
frontier likelihood is

    log sum_z p(z | s) exp ell(h_t,z,k).

No path is selected and no score weight is introduced.  Candidate-to-event
association is marginalized by the training objective; successor intention
is marginalized here; and only a completed normal mark transports procedure
state.  These are nested latent variables of one conditional likelihood.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from provia.observer.factors import EVENT_FACTORS

from .completion_transport import FactorizedCompletionTransport
from .state_mark import PredictiveStateMarkEnergy




def exact_intent_set_log_mixture(
    node_log_likelihood: Tensor,
    frontier_node_probability: Tensor,
) -> Tensor:
    """Marginalize successor identity without selecting or mean-pooling it.

    ``node_log_likelihood`` has shape ``[time,node,explanation]`` and
    ``frontier_node_probability`` has shape ``[frontier,node]``.  Rows with
    no represented successor return a neutral log factor; their terminal/open
    status is represented separately by the frontier structural key and the
    analytic open-world law.
    """

    if node_log_likelihood.ndim != 3 or not torch.isfinite(
        node_log_likelihood
    ).all():
        raise ValueError(
            "node log likelihood must have shape [time,node,explanation]"
        )
    if (
        frontier_node_probability.ndim != 2
        or frontier_node_probability.shape[1]
        != node_log_likelihood.shape[1]
        or not torch.isfinite(frontier_node_probability).all()
        or torch.any(frontier_node_probability < 0)
    ):
        raise ValueError(
            "frontier successor probabilities must have shape [frontier,node]"
        )
    support = frontier_node_probability.sum(dim=-1, keepdim=True)
    if torch.any(support > 1.0 + 4e-5):
        raise ValueError("frontier successor probability exceeds one")
    probability = frontier_node_probability / support.clamp_min(
        torch.finfo(frontier_node_probability.dtype).tiny
    )
    offset = node_log_likelihood.amax(dim=1)
    stabilized = torch.exp(node_log_likelihood - offset.unsqueeze(1))
    mixture = torch.einsum(
        "sn,tnk->tsk", probability.to(stabilized), stabilized
    )
    result = mixture.clamp_min(torch.finfo(mixture.dtype).tiny).log()
    result = result + offset.unsqueeze(1)
    return torch.where(
        support.to(dtype=torch.bool).unsqueeze(0),
        result,
        torch.zeros_like(result),
    )


class IntentSetPredictiveStateMarkEnergy(PredictiveStateMarkEnergy):
    """Candidate state read against every legal successor before marginalizing."""

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
        self, transport: FactorizedCompletionTransport
    ) -> Tensor:
        reference = self.node_key.weight
        node = reference.new_zeros(
            len(transport.language.node_ids), self.model_dim
        )
        for name in EVENT_FACTORS:
            possibility = getattr(
                transport, f"node_{name}_possibility"
            ).to(reference)
            possibility = possibility / possibility.sum(
                dim=-1, keepdim=True
            ).clamp_min(torch.finfo(possibility.dtype).tiny)
            node = node + possibility @ self.factor_embeddings[name].weight
        return self.node_key(self.node_norm(node))

    def build_frontier_key(
        self, transport: FactorizedCompletionTransport
    ) -> Tensor:
        """Cache structural frontier keys followed by successor-token keys."""

        reference = self.frontier_key.weight
        structural = torch.stack(
            (
                transport.frontier_successor_support,
                transport.frontier_terminal_probability,
                transport.frontier_closure_entropy,
            ),
            dim=-1,
        ).to(reference)
        frontier = self.frontier_key(
            self.frontier_norm(self.frontier_structural_map(structural))
        )
        return torch.cat((frontier, self._node_semantic_key(transport)), dim=0)

__all__ = [
    "IntentSetPredictiveStateMarkEnergy",
    "exact_intent_set_log_mixture",
]
