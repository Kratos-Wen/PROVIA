"""Bimanual example streams and prefix evidence.

The model emits one insertion/normal/error/correction posterior at every
causal prefix.  Approaches are read-only.  Interaction onset creates a
probabilistic step whose normal branch alone traverses the soft
procedure operator; later visual and action-effect evidence revises all
retained explanations, and event close never transports the frontier again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from provia.data.role_stream import _validate_cache
from provia.observer.factors import EVENT_FACTORS, EVENT_ROLES
from provia.training.common import _sha256
from provia.model.role_energy import RoleStep


@dataclass
class _Candidate:
    index: int
    start: int
    onset: int
    end: int
    role: int
    factor_probability: Mapping[str, Tensor]
    effect_statistic: Tensor
    cumulative_log_likelihood: Tensor | None = None
    endpoint_hidden: Tensor | None = None
    step: RoleStep | None = None


def _examples_binding_hashes(examples):
    """sha256 of the examples file and, for gap-candidate examples, of the base examples whose registries they inherit."""
    hashes = {_sha256(examples)}
    try:
        meta = torch.load(examples, map_location="cpu", weights_only=False).get("metadata", {})
        base = meta.get("gap_base_examples_sha256") if isinstance(meta, dict) else None
        if base: hashes.add(str(base))
    except Exception:
        pass
    return hashes


def _load_prefix_registry(root: Path, examples: Path) -> Mapping[str, object]:
    path = root / "registry.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    registry = json.loads(path.read_text())
    if (
        not isinstance(registry, Mapping)
        or registry.get("schema")
        != "role_prefix_observer_evidence"
        or registry.get("examples_sha256") not in _examples_binding_hashes(examples)
        or registry.get("causal") is not True
        or registry.get("parameters_fitted") is not False
        or registry.get("annotations_used_to_compute_evidence") is not False
        or registry.get("test_split_accessed") is not False
        or not isinstance(registry.get("videos"), Mapping)
    ):
        raise ValueError("role prefix-evidence registry is incompatible")
    for video_id, record in registry["videos"].items():  # type: ignore[union-attr]
        if not isinstance(record, Mapping):
            raise ValueError(f"{video_id}: malformed prefix-evidence record")
        file = root / str(record.get("file"))
        if not file.is_file() or _sha256(file) != record.get("sha256"):
            raise ValueError(f"{video_id}: prefix evidence changed")
    return registry


def _role_stream(
    feature_root: Path, video_id: str
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    frame, timestamp, global_feature, role_feature, _ = _validate_cache(
        feature_root / f"{video_id}_stream_features.npz"
    )
    return (
        torch.from_numpy(frame.copy()),
        torch.from_numpy(timestamp.copy()),
        torch.from_numpy(global_feature.copy()),
        torch.from_numpy(role_feature.copy()),
    )


def _prefix_evidence(
    root: Path,
    registry: Mapping[str, object],
    video_id: str,
    frame: Tensor,
    timestamp: Tensor,
    labels: Mapping[str, Sequence[str]],
) -> tuple[Tensor, dict[str, Tensor]]:
    records = registry["videos"]
    if not isinstance(records, Mapping) or video_id not in records:
        raise ValueError(f"{video_id}: no role prefix evidence")
    record = records[video_id]
    if not isinstance(record, Mapping):
        raise ValueError(f"{video_id}: malformed role prefix evidence")
    with np.load(root / str(record["file"]), allow_pickle=False) as payload:
        evidence_frame = np.asarray(payload["frame_ids"], dtype=np.int64).copy()
        evidence_time = np.asarray(
            payload["timestamps_sec"], dtype=np.float64
        ).copy()
        event = np.asarray(payload["event_probability"], dtype=np.float32).copy()
        factor = {
            name: np.asarray(
                payload[f"factor_{name}_probability"], dtype=np.float32
            ).copy()
            for name in EVENT_FACTORS
        }
    frames = len(frame)
    if (
        not np.array_equal(evidence_frame, frame.numpy())
        or not np.array_equal(evidence_time, timestamp.numpy())
        or event.shape != (frames, len(EVENT_ROLES))
        or not np.isfinite(event).all()
        or np.any((event < 0) | (event > 1))
    ):
        raise ValueError(f"{video_id}: role evidence and feature stream disagree")
    for name in EVENT_FACTORS:
        expected = (frames, len(EVENT_ROLES), len(labels[name]))
        value = factor[name]
        if (
            value.shape != expected
            or not np.isfinite(value).all()
            or np.any(value < 0)
            or np.any(value.sum(axis=-1) <= 0)
        ):
            raise ValueError(f"{video_id}: malformed {name} prefix evidence")
    return torch.from_numpy(event), {
        name: torch.from_numpy(value) for name, value in factor.items()
    }


def _validate_sequence(
    sequence: Mapping[str, object], frames: int, labels: Mapping[str, Sequence[str]]
) -> list[_Candidate]:
    required = (
        "start_index",
        "onset_index",
        "end_index",
        "role_index",
        "statistic",
    )
    if any(not isinstance(sequence.get(name), Tensor) for name in required):
        raise ValueError("role candidate sequence lacks tensor fields")
    start = sequence["start_index"]  # type: ignore[assignment]
    onset = sequence["onset_index"]  # type: ignore[assignment]
    end = sequence["end_index"]  # type: ignore[assignment]
    role = sequence["role_index"]  # type: ignore[assignment]
    count = len(start)
    if (
        not count
        or start.shape != onset.shape
        or start.shape != end.shape
        or start.shape != role.shape
        or start.dtype != torch.long
        or onset.dtype != torch.long
        or end.dtype != torch.long
        or role.dtype != torch.long
        or torch.any(start < 0)
        or torch.any(start > onset)
        or torch.any(onset > end)
        or torch.any(end >= frames)
        or torch.any((role < 0) | (role >= len(EVENT_ROLES)))
    ):
        raise ValueError("role candidate boundaries are malformed")
    for selected_role in range(len(EVENT_ROLES)):
        indices = sorted(
            (index for index in range(count) if int(role[index]) == selected_role),
            key=lambda index: (int(start[index]), int(onset[index]), int(end[index])),
        )
        if any(int(start[right]) <= int(end[left]) for left, right in zip(indices, indices[1:])):
            raise ValueError("decoded candidates overlap within one role")
    candidates = []
    for index in range(count):
        factors = {}
        for name in EVENT_FACTORS:
            value = sequence[f"factor_{name}"]
            if not isinstance(value, Tensor) or value.shape != (
                count,
                len(labels[name]),
            ):
                raise ValueError(f"candidate {name} probability is malformed")
            factors[name] = value[index]
        candidates.append(
            _Candidate(
                index=index,
                start=int(start[index]),
                onset=int(onset[index]),
                end=int(end[index]),
                role=int(role[index]),
                factor_probability=factors,
                effect_statistic=sequence["statistic"][index],  # type: ignore[index]
            )
        )
    return candidates
