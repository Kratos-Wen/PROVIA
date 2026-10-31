"""Causal temporal blocks with an exactly equivalent streaming path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TemporalStreamState:
    """Cached normalized inputs for every dilated temporal block."""

    histories: tuple[Tensor, ...]

class CausalDepthwiseBlock(nn.Module):
    """Pre-norm depthwise temporal convolution with left context only."""

    def __init__(
        self,
        dimension: int,
        *,
        kernel_size: int,
        dilation: int,
        expansion: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if kernel_size < 2:
            raise ValueError("kernel_size must be at least two")
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        if expansion < dimension:
            raise ValueError("expansion must be at least dimension")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        self.dimension = int(dimension)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.context = self.dilation * (self.kernel_size - 1)
        self.norm = nn.LayerNorm(self.dimension)
        self.depthwise = nn.Conv1d(
            self.dimension,
            self.dimension,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            groups=self.dimension,
        )
        self.channel_mixer = nn.Sequential(
            nn.Linear(self.dimension, 2 * expansion),
            nn.GLU(dim=-1),
            nn.Linear(expansion, self.dimension),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: Tensor) -> Tensor:
        if sequence.ndim != 3 or sequence.shape[-1] != self.dimension:
            raise ValueError(
                f"expected [batch,time,{self.dimension}], got {tuple(sequence.shape)}"
            )
        normalized = self.norm(sequence).transpose(1, 2)
        convolved = self.depthwise(F.pad(normalized, (self.context, 0)))
        update = self.channel_mixer(F.silu(convolved.transpose(1, 2)))
        return sequence + self.dropout(update)

    def initial_state(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        return torch.zeros(
            batch_size,
            self.dimension,
            self.context,
            device=device,
            dtype=dtype,
        )

    def step(self, value: Tensor, history: Optional[Tensor]) -> tuple[Tensor, Tensor]:
        """Consume one token and return the output and updated left context."""

        if value.ndim != 2 or value.shape[-1] != self.dimension:
            raise ValueError(
                f"expected [batch,{self.dimension}], got {tuple(value.shape)}"
            )
        normalized = self.norm(value).unsqueeze(-1)
        if history is None:
            history = self.initial_state(
                len(value), device=value.device, dtype=value.dtype
            )
        expected = (len(value), self.dimension, self.context)
        if tuple(history.shape) != expected:
            raise ValueError(
                f"stream history has shape {tuple(history.shape)}, expected {expected}"
            )
        window = torch.cat((history, normalized), dim=-1)
        convolved = self.depthwise(window).squeeze(-1)
        update = self.channel_mixer(F.silu(convolved))
        output = value + self.dropout(update)
        next_history = window[:, :, -self.context :]
        return output, next_history


class CausalTemporalEncoder(nn.Module):
    """A bounded-memory TCN whose batch and frame-wise paths share weights."""

    def __init__(
        self,
        dimension: int,
        *,
        dilations: Sequence[int] = (1, 2, 4, 8),
        kernel_size: int = 3,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not dilations:
            raise ValueError("at least one dilation is required")
        if any(int(value) <= 0 for value in dilations):
            raise ValueError("dilations must be positive")
        self.dimension = int(dimension)
        expansion = self.dimension * int(expansion_factor)
        self.blocks = nn.ModuleList(
            CausalDepthwiseBlock(
                self.dimension,
                kernel_size=kernel_size,
                dilation=int(dilation),
                expansion=expansion,
                dropout=dropout,
            )
            for dilation in dilations
        )
        self.output_norm = nn.LayerNorm(self.dimension)

    def forward(self, sequence: Tensor) -> Tensor:
        for block in self.blocks:
            sequence = block(sequence)
        return self.output_norm(sequence)

    def step(
        self, value: Tensor, state: Optional[TemporalStreamState] = None
    ) -> tuple[Tensor, TemporalStreamState]:
        if state is None:
            histories: tuple[Optional[Tensor], ...] = (None,) * len(self.blocks)
        else:
            if len(state.histories) != len(self.blocks):
                raise ValueError("stream state does not match encoder depth")
            histories = state.histories
        next_histories: list[Tensor] = []
        for block, history in zip(self.blocks, histories):
            value, next_history = block.step(value, history)
            next_histories.append(next_history)
        return self.output_norm(value), TemporalStreamState(tuple(next_histories))
