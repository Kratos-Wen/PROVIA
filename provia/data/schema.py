"""Versioned, lossless contracts for partially annotated procedural video.

Procedural-video datasets expose different forms of supervision.
The schema therefore represents absence explicitly.  An unavailable field is
``None`` and its capability is marked ``unavailable``; it is never silently
converted into a normal/negative label.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Mapping, Optional, Sequence


SCHEMA_VERSION = "1.0.0"
AVAILABILITY_STATES = frozenset({"available", "partial", "unavailable"})
CAPABILITIES = (
    "temporal_segments",
    "action_semantics",
    "error_binary",
    "error_type",
    "correction",
    "hand_identity",
    "hand_tracks",
    "object_tracks",
    "component_state",
    "procedure_graph",
    "human_intervention",
    "prospective_pnr",
)
VALID_HANDS = frozenset({"left", "right", "bimanual", "unknown"})
VALID_CORRECTNESS = frozenset({"correct", "error", "correction", "unknown"})


class SchemaError(ValueError):
    """Raised when an episode violates the canonical data contract."""


def _finite_nonnegative(value: Optional[float], name: str) -> list[str]:
    if value is None:
        return []
    if not isfinite(float(value)) or float(value) < 0:
        return [f"{name} must be finite and non-negative, got {value!r}"]
    return []


def _clean_optional(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).lstrip("\ufeff").strip()
    return text or None


@dataclass(frozen=True)
class Segment:
    """One observed or explicitly omitted procedural action."""

    segment_id: str
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    onset_sec: Optional[float] = None
    observed: bool = True
    action_id: Optional[str] = None
    action_text: Optional[str] = None
    verb: Optional[str] = None
    noun: Optional[str] = None
    tool: Optional[str] = None
    hand: Optional[str] = None
    correctness: Optional[str] = None
    error_types: tuple[str, ...] = ()
    native_action_type: Optional[str] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        issues: list[str] = []
        if not str(self.segment_id).strip():
            issues.append("segment_id is empty")
        issues.extend(_finite_nonnegative(self.start_sec, "start_sec"))
        issues.extend(_finite_nonnegative(self.end_sec, "end_sec"))
        issues.extend(_finite_nonnegative(self.onset_sec, "onset_sec"))
        if (self.start_sec is None) != (self.end_sec is None):
            issues.append("start_sec and end_sec must be both present or both absent")
        if self.start_sec is not None and self.end_sec is not None:
            if self.end_sec < self.start_sec:
                issues.append("end_sec precedes start_sec")
            if self.onset_sec is not None and not (
                self.start_sec <= self.onset_sec <= self.end_sec
            ):
                issues.append("onset_sec is outside [start_sec, end_sec]")
        elif self.onset_sec is not None:
            issues.append("onset_sec cannot exist without a temporal interval")
        if not self.observed and (self.start_sec is not None or self.end_sec is not None):
            issues.append("an unobserved/omitted segment cannot have an invented interval")
        if self.hand is not None and self.hand not in VALID_HANDS:
            issues.append(f"invalid hand {self.hand!r}")
        if self.correctness is not None and self.correctness not in VALID_CORRECTNESS:
            issues.append(f"invalid correctness {self.correctness!r}")
        if self.correctness == "correct" and self.error_types:
            issues.append("a correct segment cannot carry error_types")
        return issues

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Segment":
        payload = dict(value)
        payload["error_types"] = tuple(payload.get("error_types", ()))
        return cls(**payload)


@dataclass(frozen=True)
class StateObservation:
    """A timestamped component-state vector or a single state change."""

    observation_id: str
    time_sec: float
    frame_index: Optional[int]
    values: Mapping[str, Any]
    source: str

    def validate(self) -> list[str]:
        issues = _finite_nonnegative(self.time_sec, "time_sec")
        if self.frame_index is not None and self.frame_index < 0:
            issues.append("frame_index must be non-negative")
        if not self.values:
            issues.append("state observation has no component values")
        return issues


@dataclass(frozen=True)
class Intervention:
    """An observed human/model intervention, not automatically an optimal PNR."""

    intervention_id: str
    start_sec: float
    end_sec: Optional[float]
    intervention_type: str
    text: Optional[str] = None
    linked_segment_id: Optional[str] = None
    source: str = "annotation"

    def validate(self) -> list[str]:
        issues = _finite_nonnegative(self.start_sec, "start_sec")
        issues.extend(_finite_nonnegative(self.end_sec, "end_sec"))
        if self.end_sec is not None and self.end_sec < self.start_sec:
            issues.append("intervention end precedes start")
        if not self.intervention_type.strip():
            issues.append("intervention_type is empty")
        return issues


@dataclass(frozen=True)
class DecisionWindow:
    """Prospective warning reference available only when explicitly annotated."""

    window_id: str
    segment_id: str
    anomaly_onset_sec: float
    commitment_sec: float
    recovery_start_sec: Optional[float] = None
    recovery_end_sec: Optional[float] = None
    actionable: Optional[bool] = None
    source: str = "manual_annotation"

@dataclass(frozen=True)
class ProcedureNode:
    node_id: str
    text: str
    action_id: Optional[str] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProcedureEdge:
    source: str
    target: str
    relation: str = "precedes"
    confidence: Optional[float] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        issues: list[str] = []
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            issues.append("procedure-edge confidence must lie in [0, 1]")
        if not self.source or not self.target:
            issues.append("procedure edge has an empty endpoint")
        return issues


@dataclass(frozen=True)
class ProcedureStructure:
    structure_id: str
    nodes: tuple[ProcedureNode, ...]
    edges: tuple[ProcedureEdge, ...]
    source: str
    verified: Optional[bool] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        issues: list[str] = []
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            issues.append("procedure node IDs are not unique")
        known = set(node_ids)
        for index, edge in enumerate(self.edges):
            issues.extend(f"edge[{index}]: {item}" for item in edge.validate())
            if edge.source not in known or edge.target not in known:
                issues.append(
                    f"edge[{index}] references unknown node {edge.source!r}->{edge.target!r}"
                )
        return issues

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProcedureStructure":
        payload = dict(value)
        payload["nodes"] = tuple(ProcedureNode(**item) for item in payload["nodes"])
        payload["edges"] = tuple(ProcedureEdge(**item) for item in payload["edges"])
        return cls(**payload)


@dataclass(frozen=True)
class LabelAvailability:
    """Availability status and a human-readable reason per capability."""

    status: Mapping[str, str]
    notes: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> list[str]:
        issues: list[str] = []
        unknown_keys = set(self.status) - set(CAPABILITIES)
        if unknown_keys:
            issues.append(f"unknown capabilities: {sorted(unknown_keys)}")
        missing_keys = set(CAPABILITIES) - set(self.status)
        if missing_keys:
            issues.append(f"missing capability declarations: {sorted(missing_keys)}")
        for key, value in self.status.items():
            if value not in AVAILABILITY_STATES:
                issues.append(f"invalid availability {key}={value!r}")
        return issues

    def is_available(self, capability: str) -> bool:
        return self.status.get(capability) in {"available", "partial"}

    @classmethod
    def from_partial(
        cls,
        available: Sequence[str] = (),
        partial: Sequence[str] = (),
        notes: Optional[Mapping[str, str]] = None,
    ) -> "LabelAvailability":
        status = {key: "unavailable" for key in CAPABILITIES}
        for key in available:
            if key not in status:
                raise KeyError(f"Unknown capability {key!r}")
            status[key] = "available"
        for key in partial:
            if key not in status:
                raise KeyError(f"Unknown capability {key!r}")
            status[key] = "partial"
        return cls(status=status, notes=dict(notes or {}))


@dataclass(frozen=True)
class CanonicalEpisode:
    """A single query/reference recording in the unified PROVIA corpus."""

    dataset: str
    video_id: str
    task_id: str
    split: str
    video_path: Optional[str]
    fps: Optional[float]
    frame_count: Optional[int]
    duration_sec: Optional[float]
    subject_id: Optional[str]
    environment_id: Optional[str]
    segments: tuple[Segment, ...]
    state_observations: tuple[StateObservation, ...]
    procedure: Optional[ProcedureStructure]
    interventions: tuple[Intervention, ...]
    decision_windows: tuple[DecisionWindow, ...]
    hand_tracks_path: Optional[str]
    object_tracks_path: Optional[str]
    availability: LabelAvailability
    source_files: tuple[str, ...]
    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @property
    def episode_id(self) -> str:
        return f"{self.dataset}:{self.video_id}"

    def validate(self, raise_on_error: bool = False) -> list[str]:
        issues: list[str] = []
        if self.schema_version != SCHEMA_VERSION:
            issues.append(
                f"schema_version {self.schema_version!r} != supported {SCHEMA_VERSION!r}"
            )
        for name, value in (
            ("dataset", self.dataset),
            ("video_id", self.video_id),
            ("task_id", self.task_id),
            ("split", self.split),
        ):
            if not str(value).strip():
                issues.append(f"{name} is empty")
        if self.fps is not None and (not isfinite(self.fps) or self.fps <= 0):
            issues.append(f"fps must be finite and positive, got {self.fps!r}")
        if self.frame_count is not None and self.frame_count < 0:
            issues.append("frame_count must be non-negative")
        issues.extend(_finite_nonnegative(self.duration_sec, "duration_sec"))
        segment_ids = [item.segment_id for item in self.segments]
        if len(segment_ids) != len(set(segment_ids)):
            issues.append("segment IDs are not unique")
        for index, item in enumerate(self.segments):
            issues.extend(f"segment[{index}]: {message}" for message in item.validate())
        for index, item in enumerate(self.state_observations):
            issues.extend(f"state[{index}]: {message}" for message in item.validate())
        if self.procedure is not None:
            issues.extend(f"procedure: {message}" for message in self.procedure.validate())
        for index, item in enumerate(self.interventions):
            issues.extend(
                f"intervention[{index}]: {message}" for message in item.validate()
            )
        known_segments = set(segment_ids)
        for index, item in enumerate(self.decision_windows):
            issues.extend(f"decision[{index}]: {message}" for message in item.validate())
            if item.segment_id not in known_segments:
                issues.append(
                    f"decision[{index}] references unknown segment {item.segment_id!r}"
                )
        issues.extend(f"availability: {message}" for message in self.availability.validate())
        consistency_checks = (
            (bool(self.procedure), "procedure_graph"),
            (bool(self.state_observations), "component_state"),
            (bool(self.interventions), "human_intervention"),
            (bool(self.decision_windows), "prospective_pnr"),
            (bool(self.hand_tracks_path), "hand_tracks"),
            (bool(self.object_tracks_path), "object_tracks"),
        )
        for present, capability in consistency_checks:
            if present and not self.availability.is_available(capability):
                issues.append(
                    f"{capability} data are present but availability marks them unavailable"
                )
        if raise_on_error and issues:
            raise SchemaError(f"{self.episode_id}: " + "; ".join(issues))
        return issues

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalEpisode":
        payload = dict(value)
        payload.setdefault("object_tracks_path", None)
        payload["segments"] = tuple(
            Segment.from_dict(item) for item in payload.get("segments", ())
        )
        payload["state_observations"] = tuple(
            StateObservation(**item)
            for item in payload.get("state_observations", ())
        )
        procedure = payload.get("procedure")
        payload["procedure"] = (
            ProcedureStructure.from_dict(procedure) if procedure is not None else None
        )
        payload["interventions"] = tuple(
            Intervention(**item) for item in payload.get("interventions", ())
        )
        payload["decision_windows"] = tuple(
            DecisionWindow(**item) for item in payload.get("decision_windows", ())
        )
        payload["availability"] = LabelAvailability(**payload["availability"])
        payload["source_files"] = tuple(payload.get("source_files", ()))
        return cls(**payload)


def clean_text(value: object) -> Optional[str]:
    """Public helper shared by adapters for BOM/empty-string normalization."""

    return _clean_optional(value)
