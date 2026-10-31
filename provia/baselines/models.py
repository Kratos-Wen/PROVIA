"""Past-only neural models of the baseline comparison.

The MiniROAD block follows the official PREGO release.  MistSenseRGB retains the RGB
Video-Q-Former classification path but consumes the shared frozen visual
features and omits the explanation-only LLM.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CausalConvBlock(nn.Module):
    """Residual temporal block with explicit left-only padding."""

    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = 2 * dilation
        self.conv = nn.Conv1d(
            width, width, kernel_size=3, dilation=dilation, padding=0
        )
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: Tensor) -> Tensor:
        residual = value
        value = F.pad(value, (self.left_padding, 0))
        value = self.conv(value).transpose(1, 2)
        value = self.dropout(F.gelu(self.norm(value))).transpose(1, 2)
        return residual + value


class CausalTCN(nn.Module):
    """Strong supervised visual baseline with no procedure representation."""

    def __init__(
        self,
        input_dim: int,
        width: int = 256,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU()
        )
        self.blocks = nn.ModuleList(
            CausalConvBlock(width, int(dilation), dropout)
            for dilation in dilations
        )
        self.head = nn.Linear(width, 1)

    def forward(self, value: Tensor) -> Tensor:
        value = self.input(value).transpose(1, 2)
        for block in self.blocks:
            value = block(value)
        return self.head(value[:, :, -1]).squeeze(-1)


class MiniROAD(nn.Module):
    """Official PREGO recognition architecture with a configurable input size."""

    def __init__(
        self,
        input_dim: int,
        classes: int,
        embedding_dim: int = 2048,
        hidden_dim: int = 1024,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, classes)

    def forward(self, value: Tensor) -> Tensor:
        hidden, _ = self.gru(self.embedding(value))
        return self.head(F.relu(hidden))


class MistSenseRGB(nn.Module):
    """RGB-only causal Video-Q-Former classification path."""

    def __init__(
        self,
        input_dim: int,
        width: int = 384,
        queries: int = 32,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.memory_projection = nn.Sequential(
            nn.Linear(input_dim, width), nn.LayerNorm(width)
        )
        self.queries = nn.Parameter(torch.empty(queries, width))
        nn.init.normal_(self.queries, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=width,
            nhead=8,
            dim_feedforward=4 * width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.qformer = nn.TransformerDecoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, value: Tensor) -> Tensor:
        memory = self.memory_projection(value)
        query = self.queries.unsqueeze(0).expand(len(value), -1, -1)
        representation = self.qformer(query, memory).mean(dim=1)
        return self.head(representation).squeeze(-1)
