"""Causal marked point process for datasets with completion-only events.

Some procedural benchmarks release overlapping action intervals without actor
identity.  Treating those intervals as left/right hand tracks would invent
supervision.  Their observable causal event is instead the completion of one
or more actions.  This module is the single-state reduction of the marked
renewal observer: one positive event intensity and one conditional semantic
mark distribution are inferred from past-and-present visual evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from provia.automaton.decoded_events import DecodedEvent
from provia.observer.temporal import CausalTemporalEncoder, TemporalStreamState
from provia.observer.factors import EVENT_FACTORS


@dataclass(frozen=True)
class CompletionProcessEvidence:
    """Per-frame event intensity and semantic mark evidence."""

    intensity_per_second: Tensor
    factor_logits: Mapping[str, Tensor]


@dataclass(frozen=True)
class CompletionProcessStreamState:
    """Bounded causal state for one frame-at-a-time inference."""

    previous_global: Tensor
    previous_timestamp_sec: Tensor
    event_temporal: TemporalStreamState | None
    semantic_temporal: TemporalStreamState | None

class MarkedCompletionProcess(nn.Module):
    """A causal visual point process over set-valued action completions."""

    def __init__(
        self,
        input_dim: int,
        factor_labels: Mapping[str, Sequence[str]],
        base_rate_per_second: Tensor | float,
        *,
        model_dim: int | None = None,
        dilations: Sequence[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if tuple(factor_labels) != EVENT_FACTORS:
            raise ValueError(f"factor labels must follow {EVENT_FACTORS}")
        base_rate = torch.as_tensor(base_rate_per_second, dtype=torch.float32)
        if base_rate.numel() != 1 or not torch.isfinite(base_rate).all() or base_rate.item() <= 0:
            raise ValueError("base completion rate must be finite and positive")
        self.input_dim = int(input_dim)
        self.factor_labels = {
            name: tuple(str(value) for value in factor_labels[name])
            for name in EVENT_FACTORS
        }
        if (model_dim is None) != (dilations is None):
            raise ValueError("model_dim and dilations must be supplied together")
        if model_dim is not None and model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if dilations is not None and (
            not dilations or any(int(value) <= 0 for value in dilations)
        ):
            raise ValueError("dilations must be non-empty and positive")
        self.model_dim = self.input_dim if model_dim is None else int(model_dim)
        self.dilations = (
            None if dilations is None else tuple(int(value) for value in dilations)
        )
        feature_dim = 3 * self.input_dim
        self.event_map = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, self.model_dim),
            nn.GELU(),
            nn.LayerNorm(self.model_dim),
        )
        self.semantic_map = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, self.model_dim),
            nn.GELU(),
            nn.LayerNorm(self.model_dim),
        )
        if self.dilations is None:
            self.event_temporal: nn.Module = nn.Identity()
            self.semantic_temporal: nn.Module = nn.Identity()
        else:
            self.event_temporal = CausalTemporalEncoder(
                self.model_dim,
                dilations=self.dilations,
                kernel_size=3,
                expansion_factor=4,
                dropout=dropout,
            )
            self.semantic_temporal = CausalTemporalEncoder(
                self.model_dim,
                dilations=self.dilations,
                kernel_size=3,
                expansion_factor=4,
                dropout=dropout,
            )
        self.intensity_head = nn.Linear(self.model_dim, 1)
        self.factor_heads = nn.ModuleDict(
            {
                name: nn.Linear(self.model_dim, len(self.factor_labels[name]))
                for name in EVENT_FACTORS
            }
        )
        nn.init.zeros_(self.intensity_head.weight)
        nn.init.zeros_(self.intensity_head.bias)
        self.register_buffer("base_rate_per_second", base_rate.reshape(()))

    def _causal_features(
        self,
        global_features: Tensor,
        role_change_features: Tensor,
        timestamps_sec: Tensor,
    ) -> Tensor:
        if global_features.ndim != 2 or global_features.shape[-1] != self.input_dim:
            raise ValueError("global_features must have shape [time,input_dim]")
        if role_change_features.shape != (len(global_features), 2, self.input_dim):
            raise ValueError("role_change_features must have shape [time,2,input_dim]")
        if timestamps_sec.shape != (len(global_features),):
            raise ValueError("timestamps must align with visual features")
        if len(timestamps_sec) > 1 and torch.any(timestamps_sec[1:] <= timestamps_sec[:-1]):
            raise ValueError("timestamps must be strictly increasing")
        previous = torch.cat((global_features[:1], global_features[:-1]), dim=0)
        delta = torch.cat(
            (
                torch.ones_like(timestamps_sec[:1]),
                timestamps_sec[1:] - timestamps_sec[:-1],
            )
        ).to(dtype=global_features.dtype)
        velocity = role_change_features[:, 0].to(global_features) / delta[:, None]
        return torch.cat((global_features, previous, velocity), dim=-1)

    def forward(
        self,
        global_features: Tensor,
        role_change_features: Tensor,
        timestamps_sec: Tensor,
    ) -> CompletionProcessEvidence:
        features = self._causal_features(
            global_features, role_change_features, timestamps_sec
        )
        event_hidden = self.event_map(features)
        semantic_hidden = self.semantic_map(features)
        if self.dilations is not None:
            event_hidden = self.event_temporal(event_hidden.unsqueeze(0)).squeeze(0)
            semantic_hidden = self.semantic_temporal(
                semantic_hidden.unsqueeze(0)
            ).squeeze(0)
        # softplus gives a strictly positive intensity.  The fixed offset makes
        # the zero-initialized neural correction reproduce the fold-local base
        # rate exactly; there is no clipping or learned score threshold.
        offset = torch.log(torch.expm1(self.base_rate_per_second.to(event_hidden)))
        intensity = F.softplus(self.intensity_head(event_hidden).squeeze(-1) + offset)
        return CompletionProcessEvidence(
            intensity_per_second=intensity,
            factor_logits={
                name: head(semantic_hidden) for name, head in self.factor_heads.items()
            },
        )

    def step(
        self,
        global_feature: Tensor,
        role_change_feature: Tensor,
        timestamp_sec: Tensor,
        state: CompletionProcessStreamState | None = None,
    ) -> tuple[CompletionProcessEvidence, CompletionProcessStreamState]:
        """Consume one causal frame with exactly the same computation as ``forward``."""

        if global_feature.ndim != 2 or global_feature.shape[-1] != self.input_dim:
            raise ValueError("global_feature must have shape [batch,input_dim]")
        expected_change = (len(global_feature), 2, self.input_dim)
        if tuple(role_change_feature.shape) != expected_change:
            raise ValueError("role_change_feature must have shape [batch,2,input_dim]")
        if timestamp_sec.shape != (len(global_feature),):
            raise ValueError("timestamp_sec must have shape [batch]")
        if state is None:
            previous = global_feature
            delta = torch.ones_like(timestamp_sec)
            event_state = None
            semantic_state = None
        else:
            if state.previous_global.shape != global_feature.shape:
                raise ValueError("stream state batch or feature dimension changed")
            if state.previous_timestamp_sec.shape != timestamp_sec.shape:
                raise ValueError("stream state timestamp shape changed")
            delta = timestamp_sec - state.previous_timestamp_sec.to(timestamp_sec)
            if torch.any(delta <= 0):
                raise ValueError("stream timestamps must be strictly increasing")
            previous = state.previous_global.to(global_feature)
            event_state = state.event_temporal
            semantic_state = state.semantic_temporal
        velocity = role_change_feature[:, 0].to(global_feature) / delta.to(
            global_feature
        )[:, None]
        features = torch.cat((global_feature, previous, velocity), dim=-1)
        event_hidden = self.event_map(features)
        semantic_hidden = self.semantic_map(features)
        if self.dilations is None:
            next_event_state = None
            next_semantic_state = None
        else:
            if not isinstance(self.event_temporal, CausalTemporalEncoder) or not isinstance(
                self.semantic_temporal, CausalTemporalEncoder
            ):
                raise TypeError("long-context temporal encoders are malformed")
            event_hidden, next_event_state = self.event_temporal.step(
                event_hidden, event_state
            )
            semantic_hidden, next_semantic_state = self.semantic_temporal.step(
                semantic_hidden, semantic_state
            )
        offset = torch.log(torch.expm1(self.base_rate_per_second.to(event_hidden)))
        evidence = CompletionProcessEvidence(
            intensity_per_second=F.softplus(
                self.intensity_head(event_hidden).squeeze(-1) + offset
            ),
            factor_logits={
                name: head(semantic_hidden) for name, head in self.factor_heads.items()
            },
        )
        next_state = CompletionProcessStreamState(
            previous_global=global_feature,
            previous_timestamp_sec=timestamp_sec,
            event_temporal=next_event_state,
            semantic_temporal=next_semantic_state,
        )
        return evidence, next_state

    @staticmethod
    def event_probability(intensity_per_second: Tensor, timestamps_sec: Tensor) -> Tensor:
        if intensity_per_second.shape != timestamps_sec.shape:
            raise ValueError("intensity and timestamps must have the same shape")
        if len(timestamps_sec) < 2:
            return torch.zeros_like(intensity_per_second)
        delta = torch.cat(
            (
                torch.zeros_like(timestamps_sec[:1]),
                timestamps_sec[1:] - timestamps_sec[:-1],
            )
        ).to(intensity_per_second)
        integrated = intensity_per_second * delta
        return -torch.expm1(-integrated)


def build_marked_completion_process(
    metadata: Mapping[str, object],
    input_dim: int,
    factor_labels: Mapping[str, Sequence[str]],
) -> MarkedCompletionProcess:
    """Construct the version certified by checkpoint metadata."""

    architecture = metadata.get("observer_architecture")
    base_rate = float(metadata["base_rate_per_second"])
    if architecture == "marked_completion_process":
        return MarkedCompletionProcess(input_dim, factor_labels, base_rate)
    if architecture == "completion_process_observer":
        raw_dilations = metadata.get("dilations")
        if not isinstance(raw_dilations, Sequence) or isinstance(
            raw_dilations, (str, bytes)
        ):
            raise ValueError("checkpoint lacks causal dilations")
        return MarkedCompletionProcess(
            input_dim,
            factor_labels,
            base_rate,
            model_dim=int(metadata["model_dim"]),
            dilations=tuple(int(value) for value in raw_dilations),
            dropout=float(metadata.get("dropout", 0.1)),
        )
    raise ValueError(f"unknown completion observer architecture: {architecture}")


def decode_completion_posterior_medians(
    event_probability: Tensor,
    factor_logits: Mapping[str, Tensor],
    factor_priors: Mapping[str, Tensor],
) -> tuple[DecodedEvent, ...]:
    """Emit a completion when its accumulated causal posterior reaches 1/2."""

    if event_probability.ndim != 1:
        raise ValueError("event_probability must be a vector")
    if any(
        name not in factor_logits
        or factor_logits[name].shape[0] != len(event_probability)
        or factor_logits[name].ndim != 2
        for name in EVENT_FACTORS
    ):
        raise ValueError("factor logits must align with completion probabilities")
    survival = 1.0
    result: list[DecodedEvent] = []
    for index in range(1, len(event_probability)):
        probability = float(event_probability[index].detach().cpu())
        if not 0.0 <= probability <= 1.0:
            raise ValueError("event probabilities must lie in [0,1]")
        survival *= 1.0 - probability
        if survival > 0.5:
            continue
        distributions = {}
        for name in EVENT_FACTORS:
            prior = factor_priors[name].to(factor_logits[name])
            if prior.shape != factor_logits[name][index].shape:
                raise ValueError(f"{name} prior has the wrong shape")
            distributions[name] = tuple(
                float(value)
                for value in (
                    factor_logits[name][index] + prior.log()
                ).log_softmax(dim=-1).detach().cpu()
            )
        result.append(
            DecodedEvent(
                role=0,
                start_index=index,
                onset_index=index,
                end_index=index,
                factor_log_probability=distributions,
            )
        )
        survival = 1.0
    return tuple(result)
