"""Causal marked-renewal observation of role-specific procedural events.

The process is defined over event boundaries rather than frame occupations.
Each hand carries one of three event phases and a marked competing-risk process
decides when that phase is renewed.  Compound renewal marks preserve adjacent
events even when no sampled idle frame separates them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .factors import EVENT_FACTORS, EVENT_PHASES, EVENT_ROLES

if TYPE_CHECKING:
    from provia.data.role_stream import CompetitiveRoleStream
    from provia.data.streaming_events import StreamingEventInterval


RENEWAL_MARKS = (
    "none",
    "start",
    "start_onset",
    "onset",
    "end",
    "renew_start",
    "renew_onset",
)

NONE, START, START_ONSET, ONSET, END, RENEW_START, RENEW_ONSET = range(
    len(RENEWAL_MARKS)
)

# A negative destination denotes an impossible mark from that source phase.
# Rows follow idle, approach, interaction; columns follow RENEWAL_MARKS.
MARK_DESTINATION = (
    (0, 1, 2, -1, -1, -1, -1),
    (1, -1, -1, 2, -1, -1, -1),
    (2, -1, -1, -1, 0, 1, 2),
)
MARK_SOURCE = (-1, 0, 0, 1, 2, 2, 2)


@dataclass(frozen=True)
class MarkedRenewalEvidence:
    """Observation-conditioned boundary intensities and semantic evidence."""

    intensity_per_second: Tensor
    factor_logits: Mapping[str, Tensor]


@dataclass(frozen=True)
class MarkedRenewalPosterior:
    """Causal phase and boundary-mark posteriors for one video stream."""

    state_probability: Tensor
    mark_probability: Tensor
    conditional_mark_probability: Tensor


def renewal_mark_targets(
    intervals: Sequence["StreamingEventInterval"],
    length: int,
) -> Tensor:
    """Rasterize exact boundary marks without inventing idle samples.

    A boundary at frame ``t`` describes the update from frame ``t-1`` to
    frame ``t``.  Boundaries before the first observed frame are left
    censored and are represented by the initial phase distribution instead.
    """

    if length <= 0:
        raise ValueError("a renewal stream must contain at least one frame")
    target = torch.full(
        (length, len(EVENT_ROLES)), NONE, dtype=torch.long
    )
    for role in range(len(EVENT_ROLES)):
        ordered = sorted(
            (interval for interval in intervals if interval.role == role),
            key=lambda value: (value.start, value.onset, value.end),
        )
        for interval in ordered:
            if not 0 <= interval.start <= interval.onset <= interval.end < length:
                raise ValueError("an event interval falls outside the stream")
        ending = {interval.end + 1: interval for interval in ordered if interval.end + 1 < length}
        starting = {interval.start: interval for interval in ordered if interval.start > 0}
        onsetting = {
            interval.onset: interval
            for interval in ordered
            if interval.onset > interval.start
        }
        if len(ending) != sum(interval.end + 1 < length for interval in ordered):
            raise ValueError("multiple same-role events end at one sampled frame")
        if len(starting) != sum(interval.start > 0 for interval in ordered):
            raise ValueError("multiple same-role events start at one sampled frame")
        if len(onsetting) != sum(interval.onset > interval.start for interval in ordered):
            raise ValueError("multiple same-role events reach onset at one sampled frame")
        for index in range(1, length):
            previous_ended = ending.get(index)
            current_started = starting.get(index)
            current_onset = onsetting.get(index)
            if previous_ended is not None and current_started is not None:
                if current_onset is not None:
                    raise ValueError("a renewal onset must belong to its new event")
                target[index, role] = (
                    RENEW_ONSET
                    if current_started.onset == index
                    else RENEW_START
                )
            elif previous_ended is not None:
                if current_onset is not None:
                    raise ValueError("an onset cannot coincide with an unrelated end")
                target[index, role] = END
            elif current_started is not None:
                if current_onset is not None:
                    raise ValueError("a start-onset event was counted twice")
                target[index, role] = (
                    START_ONSET if current_started.onset == index else START
                )
            elif current_onset is not None:
                target[index, role] = ONSET
    return target


def estimate_renewal_statistics(
    streams: Sequence["CompetitiveRoleStream"],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Estimate fold-local base hazards and the initial phase law.

    Jeffreys half-counts keep every legal competing risk learnable without a
    manually chosen smoothing constant.  Exposure and rates are in seconds.
    """

    if not streams:
        raise ValueError("renewal statistics require at least one stream")
    exposure = torch.zeros(len(EVENT_PHASES), dtype=torch.float64)
    mark_counts = torch.zeros(len(RENEWAL_MARKS), dtype=torch.float64)
    initial_counts = torch.zeros(len(EVENT_PHASES), dtype=torch.float64)
    for stream in streams:
        target = renewal_mark_targets(stream.intervals, len(stream.frame_ids))
        phases = stream.phase_targets.to(torch.long)
        if phases.shape != (len(stream.frame_ids), len(EVENT_ROLES)):
            raise ValueError("phase targets do not align with the role stream")
        initial_counts += torch.bincount(
            phases[0], minlength=len(EVENT_PHASES)
        ).to(torch.float64)
        if len(stream.frame_ids) == 1:
            continue
        delta = stream.timestamps_sec[1:].to(torch.float64) - stream.timestamps_sec[:-1].to(torch.float64)
        if torch.any(delta <= 0):
            raise ValueError("timestamps must be strictly increasing")
        source = phases[:-1]
        exposure.scatter_add_(
            0,
            source.reshape(-1),
            delta[:, None].expand_as(source).reshape(-1),
        )
        mark_counts += torch.bincount(
            target[1:].reshape(-1), minlength=len(RENEWAL_MARKS)
        ).to(torch.float64)
        destination = phases[1:]
        table = torch.tensor(MARK_DESTINATION, dtype=torch.long)
        expected = table[source, target[1:]]
        if torch.any(expected < 0) or not torch.equal(expected, destination):
            mismatch = int((expected != destination).sum())
            raise ValueError(
                f"event boundaries and phase targets disagree at {mismatch} updates"
            )
    if torch.any(exposure <= 0):
        raise ValueError("every renewal phase needs positive training exposure")
    base_rates = torch.empty(len(RENEWAL_MARKS) - 1, dtype=torch.float64)
    for mark in range(1, len(RENEWAL_MARKS)):
        source = MARK_SOURCE[mark]
        base_rates[mark - 1] = (mark_counts[mark] + 0.5) / exposure[source]
    initial = initial_counts + 0.5
    initial = initial / initial.sum()
    return (
        base_rates.to(torch.float32),
        initial.to(torch.float32),
        mark_counts,
        exposure,
    )


class MarkedRenewalRoleObserver(nn.Module):
    """Role-symmetric marked point process over procedural event renewals."""

    def __init__(
        self,
        input_dim: int,
        factor_labels: Mapping[str, Sequence[str]],
        base_rates_per_second: Tensor,
        initial_phase_probability: Tensor,
        *,
        normalize_change_by_time: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input dimension must be positive")
        if tuple(factor_labels) != EVENT_FACTORS:
            raise ValueError(f"factor labels must follow {EVENT_FACTORS}")
        if base_rates_per_second.shape != (len(RENEWAL_MARKS) - 1,):
            raise ValueError("one positive base rate is required per non-null mark")
        if torch.any(base_rates_per_second <= 0) or not torch.isfinite(
            base_rates_per_second
        ).all():
            raise ValueError("base renewal rates must be finite and positive")
        if initial_phase_probability.shape != (len(EVENT_PHASES),):
            raise ValueError("initial phase law has an incompatible shape")
        if torch.any(initial_phase_probability <= 0) or not torch.isclose(
            initial_phase_probability.sum(),
            torch.tensor(1.0, dtype=initial_phase_probability.dtype),
            atol=1e-5,
        ):
            raise ValueError("initial phase law must be positive and normalized")
        self.input_dim = int(input_dim)
        self.factor_labels = {
            name: tuple(str(value) for value in factor_labels[name])
            for name in EVENT_FACTORS
        }
        self.normalize_change_by_time = bool(normalize_change_by_time)
        boundary_dim = 5 * self.input_dim
        self.boundary_map = nn.Sequential(
            nn.LayerNorm(boundary_dim),
            nn.Linear(boundary_dim, self.input_dim),
            nn.GELU(),
            nn.LayerNorm(self.input_dim),
        )
        self.semantic_map = nn.Sequential(
            nn.LayerNorm(boundary_dim),
            nn.Linear(boundary_dim, self.input_dim),
            nn.GELU(),
            nn.LayerNorm(self.input_dim),
        )
        self.intensity_head = nn.Linear(
            self.input_dim, len(RENEWAL_MARKS) - 1
        )
        self.factor_heads = nn.ModuleDict(
            {
                name: nn.Linear(self.input_dim, len(self.factor_labels[name]))
                for name in EVENT_FACTORS
            }
        )
        nn.init.zeros_(self.intensity_head.weight)
        nn.init.zeros_(self.intensity_head.bias)
        self.register_buffer(
            "base_rates_per_second",
            base_rates_per_second.detach().clone().to(torch.float32),
        )
        self.register_buffer(
            "initial_phase_probability",
            initial_phase_probability.detach().clone().to(torch.float32),
        )
        destination = torch.tensor(MARK_DESTINATION, dtype=torch.long)
        destination_one_hot = torch.zeros(
            len(EVENT_PHASES), len(RENEWAL_MARKS), len(EVENT_PHASES)
        )
        for source in range(len(EVENT_PHASES)):
            for mark in range(len(RENEWAL_MARKS)):
                following = int(destination[source, mark])
                if following >= 0:
                    destination_one_hot[source, mark, following] = 1.0
        self.register_buffer("mark_destination", destination)
        self.register_buffer("destination_one_hot", destination_one_hot)

    def markov_boundary(
        self,
        global_features: Tensor,
        role_features: Tensor,
        previous_global: Tensor,
        previous_roles: Tensor,
        delta_sec: Tensor,
        role_change_features: Optional[Tensor] = None,
    ) -> Tensor:
        if global_features.ndim != 2 or global_features.shape[-1] != self.input_dim:
            raise ValueError("global features have an incompatible shape")
        if role_features.shape != (
            len(global_features), len(EVENT_ROLES), self.input_dim
        ):
            raise ValueError("role features have an incompatible shape")
        if previous_global.shape != global_features.shape or previous_roles.shape != role_features.shape:
            raise ValueError("previous features have an incompatible shape")
        if delta_sec.shape != (len(global_features),) or torch.any(delta_sec < 0):
            raise ValueError("frame deltas must be a non-negative vector")
        if role_change_features is not None and role_change_features.shape != role_features.shape:
            raise ValueError("role-change features have an incompatible shape")
        shared = global_features[:, None].expand_as(role_features)
        concurrent = role_features - role_features.flip(dims=(1,))
        shared_change = (global_features - previous_global)[:, None].expand_as(
            role_features
        )
        role_change = role_features - previous_roles
        local_change = shared_change if role_change_features is None else role_change_features
        if self.normalize_change_by_time:
            inverse_delta = torch.where(
                delta_sec > 0,
                delta_sec.reciprocal(),
                torch.zeros_like(delta_sec),
            ).view(-1, 1, 1)
            shared_change = shared_change * inverse_delta
            role_change = role_change * inverse_delta
            local_change = local_change * inverse_delta
        return torch.cat(
            (role_features, shared, concurrent, role_change, local_change),
            dim=-1,
        )

    def emit(
        self,
        global_features: Tensor,
        role_features: Tensor,
        previous_global: Tensor,
        previous_roles: Tensor,
        delta_sec: Tensor,
        role_change_features: Optional[Tensor] = None,
    ) -> MarkedRenewalEvidence:
        boundary = self.markov_boundary(
            global_features,
            role_features,
            previous_global,
            previous_roles,
            delta_sec,
            role_change_features,
        )
        boundary_hidden = self.boundary_map(boundary)
        semantic_hidden = self.semantic_map(boundary)
        residual = self.intensity_head(boundary_hidden)
        base = self.base_rates_per_second.to(residual)
        base_logit = torch.log(torch.expm1(base)).view(1, 1, -1)
        intensity = F.softplus(base_logit + residual)
        return MarkedRenewalEvidence(
            intensity_per_second=intensity,
            factor_logits={
                name: head(semantic_hidden)
                for name, head in self.factor_heads.items()
            },
        )

    def emit_stream(
        self,
        global_features: Tensor,
        role_features: Tensor,
        timestamps_sec: Tensor,
        role_change_features: Optional[Tensor] = None,
    ) -> MarkedRenewalEvidence:
        if global_features.ndim != 2 or role_features.ndim != 3:
            raise ValueError("a renewal stream must be unbatched [time,...]")
        if timestamps_sec.shape != (len(global_features),):
            raise ValueError("timestamps do not align with the feature stream")
        if torch.any(timestamps_sec[1:] <= timestamps_sec[:-1]):
            raise ValueError("timestamps must be strictly increasing")
        previous_global = torch.cat(
            (global_features[:1], global_features[:-1]), dim=0
        )
        previous_roles = torch.cat((role_features[:1], role_features[:-1]), dim=0)
        delta = torch.cat(
            (
                torch.zeros_like(timestamps_sec[:1]),
                timestamps_sec[1:] - timestamps_sec[:-1],
            )
        ).to(global_features)
        return self.emit(
            global_features,
            role_features,
            previous_global,
            previous_roles,
            delta,
            role_change_features,
        )

    def conditional_mark_probabilities(
        self, intensity_per_second: Tensor, delta_sec: Tensor
    ) -> Tensor:
        """Convert intensities to exact interval-censored competing risks."""

        if intensity_per_second.ndim != 3 or intensity_per_second.shape[1:] != (
            len(EVENT_ROLES), len(RENEWAL_MARKS) - 1
        ):
            raise ValueError("renewal intensities must have shape [time,2,6]")
        if delta_sec.shape != (len(intensity_per_second),) or torch.any(delta_sec < 0):
            raise ValueError("frame deltas do not align with renewal intensities")
        delta = delta_sec.to(intensity_per_second).view(-1, 1, 1)
        rows = []
        for source in range(len(EVENT_PHASES)):
            allowed = [
                mark
                for mark in range(1, len(RENEWAL_MARKS))
                if MARK_DESTINATION[source][mark] >= 0
            ]
            rates = intensity_per_second[..., [mark - 1 for mark in allowed]]
            total = rates.sum(dim=-1, keepdim=True)
            integrated = total * delta
            no_event = torch.exp(-integrated)
            event_mass = -torch.expm1(-integrated)
            cause = event_mass * rates / total.clamp_min(
                torch.finfo(rates.dtype).tiny
            )
            row = torch.zeros(
                *intensity_per_second.shape[:-1],
                len(RENEWAL_MARKS),
                dtype=intensity_per_second.dtype,
                device=intensity_per_second.device,
            )
            row[..., NONE] = no_event.squeeze(-1)
            row[..., allowed] = cause
            rows.append(row)
        probability = torch.stack(rows, dim=-2)
        return probability / probability.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(probability.dtype).tiny
        )

    def filter(
        self,
        conditional_mark_probability: Tensor,
    ) -> MarkedRenewalPosterior:
        """Marginalize the unobserved renewal marks in strict temporal order."""

        expected = (
            conditional_mark_probability.ndim == 4
            and conditional_mark_probability.shape[1:] == (
                len(EVENT_ROLES), len(EVENT_PHASES), len(RENEWAL_MARKS)
            )
        )
        if not expected:
            raise ValueError("conditional marks must have shape [time,2,3,7]")
        length = len(conditional_mark_probability)
        state = self.initial_phase_probability.to(
            conditional_mark_probability
        ).view(1, -1).expand(len(EVENT_ROLES), -1)
        states = [state]
        initial_mark = torch.zeros(
            len(EVENT_ROLES), len(RENEWAL_MARKS),
            dtype=conditional_mark_probability.dtype,
            device=conditional_mark_probability.device,
        )
        initial_mark[:, NONE] = 1.0
        marks = [initial_mark]
        destination = self.destination_one_hot.to(conditional_mark_probability)
        for index in range(1, length):
            conditional = conditional_mark_probability[index]
            mark = torch.einsum("rs,rsm->rm", state, conditional)
            following = torch.einsum(
                "rs,rsm,smd->rd", state, conditional, destination
            )
            following = following / following.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(following.dtype).tiny
            )
            marks.append(mark)
            states.append(following)
            state = following
        return MarkedRenewalPosterior(
            state_probability=torch.stack(states),
            mark_probability=torch.stack(marks),
            conditional_mark_probability=conditional_mark_probability,
        )

    def forward(
        self,
        global_features: Tensor,
        role_features: Tensor,
        timestamps_sec: Tensor,
        role_change_features: Optional[Tensor] = None,
    ) -> tuple[MarkedRenewalEvidence, MarkedRenewalPosterior]:
        evidence = self.emit_stream(
            global_features,
            role_features,
            timestamps_sec,
            role_change_features,
        )
        delta = torch.cat(
            (
                torch.zeros_like(timestamps_sec[:1]),
                timestamps_sec[1:] - timestamps_sec[:-1],
            )
        )
        conditional = self.conditional_mark_probabilities(
            evidence.intensity_per_second, delta
        )
        return evidence, self.filter(conditional)
