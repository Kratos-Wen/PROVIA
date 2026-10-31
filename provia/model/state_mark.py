"""Candidate-scoped predictive states for causal procedural event marks.

Finite causal windows discard early evidence in long events, while summing
unconstrained frame scores makes the posterior scale with event duration.
Here a candidate owns one bounded recurrent sufficient state.  The state is
reset when the candidate begins, updated once per observed frame, and read as
one conditional mark likelihood at every prefix.  Consequently later evidence
can revise an earlier interpretation without repeatedly counting old frames.

The learned state summarizes observations only.  The expert successor law is
kept as an explicit probabilistic factor and is applied once when the current
mark is read.  Procedure progress is transported only at predicted completion.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor, nn


from .intent_mark import IntentConditionedMarkEnergy


RECURRENT_CANDIDATE_ENCODER = "gated_recurrent_unit"


def validate_candidate_stream_state(
    encoder: nn.Module,
    state: Tensor | None,
    *,
    model_dim: int,
    batch_size: int = 1,
) -> None:
    """Validate either registered constant-memory candidate state."""

    if state is None:
        return
    if not isinstance(state, Tensor) or state.shape != (1, batch_size, model_dim):
        raise ValueError("candidate recurrent state has the wrong shape")


class CausalFrameEncoder(nn.Module):
    """Constant-memory frame map used by the candidate predictive state."""

    def __init__(self, input_dim: int, *, model_dim: int = 192) -> None:
        super().__init__()
        if input_dim <= 0 or model_dim <= 0:
            raise ValueError("frame encoder dimensions must be positive")
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.input = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.model_dim),
            nn.GELU(),
        )

    def forward_stream(self, causal_stream: Tensor) -> Tensor:
        if causal_stream.ndim != 2 or causal_stream.shape[-1] != self.input_dim:
            raise ValueError("causal stream must have shape [time,input_dim]")
        if not torch.isfinite(causal_stream).all():
            raise ValueError("causal stream must be finite")
        return self.input(causal_stream)

class PredictiveStateMarkEnergy(IntentConditionedMarkEnergy):
    """A bounded candidate state with a frontier-conditional mark readout."""

    endpoint_requires_complete_prefix = True

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

__all__ = [
    "CausalFrameEncoder",
    "PredictiveStateMarkEnergy",
]
