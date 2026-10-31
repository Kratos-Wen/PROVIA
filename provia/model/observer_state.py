"""Causal observer-state evidence with exact procedure-state transport.

The visual observer is frozen before anomaly learning.  At each causally
available candidate checkpoint it supplies one hidden state and soft event
semantics.  An autonomous event recurrence summarizes only this observed
query prefix.  The expert-derived procedure frontier conditions the current
likelihood but can never be written into the factual recurrence.

The resulting likelihood updates one normalized posterior over candidate
origin, execution mode, event commitment, and procedure frontier.  A normal
genuine event advances the frontier exactly once at commitment; insertions,
errors, and corrections preserve it.  Completion-only datasets are the exact
unit-commitment reduction of the same model.
"""

from __future__ import annotations


from torch import nn

from provia.observer.temporal import CausalTemporalEncoder



class CausalWindowEncoder(nn.Module):
    """Trainable frame-to-event map with a fixed 127-frame past support."""

    DILATIONS = (1, 2, 4, 8, 16, 32)
    RECEPTIVE_FIELD = 1 + 2 * sum(DILATIONS)

    def __init__(
        self, input_dim: int, *, model_dim: int = 192, dropout: float = 0.1
    ) -> None:
        super().__init__()
        if input_dim <= 0 or model_dim <= 0:
            raise ValueError("window encoder dimensions must be positive")
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.input = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.model_dim),
            nn.GELU(),
        )
        self.temporal = CausalTemporalEncoder(
            self.model_dim,
            dilations=self.DILATIONS,
            kernel_size=3,
            expansion_factor=4,
            dropout=dropout,
        )
