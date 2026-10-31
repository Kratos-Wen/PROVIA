"""Frame-aligned targets for competitive bimanual feature streams."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from provia.observer.factors import EVENT_FACTORS, EVENT_PHASES, EVENT_ROLES
from provia.model.intents import CompositionalIntentSpace, segment_factors

from .schema import CanonicalEpisode, Segment
from .streaming_events import CORRECTNESS_INDEX, IGNORE_INDEX, StreamingEventInterval


@dataclass(frozen=True)
class CompetitiveRoleStream:
    """One causal stream with a shared scene and two competitive role slots."""

    episode_id: str
    video_id: str
    global_features: Tensor
    role_features: Tensor
    role_change_features: Tensor | None
    frame_ids: Tensor
    timestamps_sec: Tensor
    phase_targets: Tensor
    factor_targets: Mapping[str, Tensor]
    factor_weights: Tensor
    anomaly_target: Tensor
    recovery_target: Tensor
    intervals: tuple[StreamingEventInterval, ...]


def _nearest(values: np.ndarray, target: float) -> int:
    position = int(np.searchsorted(values, target, side="left"))
    if position <= 0:
        return 0
    if position >= len(values):
        return len(values) - 1
    before = position - 1
    return (
        before
        if target - values[before] <= values[position] - target
        else position
    )


def _factor_indices(
    segment: Segment, intent_space: CompositionalIntentSpace
) -> tuple[int, int, int]:
    values = segment_factors(segment)
    result: list[int] = []
    for name, value in zip(("hand", *EVENT_FACTORS), values):
        if name == "hand":
            continue
        label = value if value in intent_space.labels[name] else "<unk>"
        result.append(intent_space.labels[name].index(label))
    return tuple(result)  # type: ignore[return-value]


def _validate_cache(
    path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
]:
    with np.load(path, allow_pickle=False) as payload:
        base = {
            "frame_ids",
            "timestamps_sec",
            "global_features",
            "role_features",
        }
        enhanced = base | {"role_change_features"}
        if set(payload.files) not in (base, enhanced):
            raise ValueError(f"{path}: expected {sorted(base)} or {sorted(enhanced)}")
        frame_ids = np.asarray(payload["frame_ids"], dtype=np.int64)
        timestamps = np.asarray(payload["timestamps_sec"], dtype=np.float64)
        global_features = np.asarray(payload["global_features"], dtype=np.float32)
        role_features = np.asarray(payload["role_features"], dtype=np.float32)
        role_change_features = (
            np.asarray(payload["role_change_features"], dtype=np.float32)
            if "role_change_features" in payload.files
            else None
        )
    length = len(frame_ids)
    if (
        frame_ids.ndim != 1
        or timestamps.ndim != 1
        or global_features.ndim != 2
        or role_features.ndim != 3
        or role_features.shape[1] != len(EVENT_ROLES)
        or global_features.shape[-1] != role_features.shape[-1]
        or not length
        or any(len(value) != length for value in (timestamps, global_features, role_features))
        or np.any(np.diff(frame_ids) <= 0)
        or np.any(np.diff(timestamps) <= 0)
        or not np.isfinite(global_features).all()
        or not np.isfinite(role_features).all()
        or (
            role_change_features is not None
            and (
                role_change_features.shape != role_features.shape
                or not np.isfinite(role_change_features).all()
            )
        )
    ):
        raise ValueError(f"{path}: malformed competitive-role stream")
    return (
        frame_ids,
        timestamps,
        global_features,
        role_features,
        role_change_features,
    )


def load_competitive_role_stream(
    episode: CanonicalEpisode,
    feature_root: Path,
    intent_space: CompositionalIntentSpace,
    *,
    file_pattern: str = "{video_id}_stream_features.npz",
) -> CompetitiveRoleStream:
    """Load one stream without consulting labels outside its episode record."""

    path = feature_root / file_pattern.format(video_id=episode.video_id)
    if not path.is_file():
        raise FileNotFoundError(path)
    (
        frame_ids,
        timestamps,
        global_features,
        role_features,
        role_change_features,
    ) = _validate_cache(path)
    length = len(frame_ids)
    phase = np.full((length, len(EVENT_ROLES)), EVENT_PHASES.index("idle"), np.int64)
    factor_targets = {
        name: np.full((length, len(EVENT_ROLES)), IGNORE_INDEX, np.int64)
        for name in EVENT_FACTORS
    }
    factor_weights = np.zeros((length, len(EVENT_ROLES)), np.float32)
    anomaly = np.zeros(length, np.bool_)
    recovery = np.zeros(length, np.bool_)
    intervals: list[StreamingEventInterval] = []
    role_index = {name: index for index, name in enumerate(EVENT_ROLES)}

    ordered = sorted(
        (
            segment
            for segment in episode.segments
            if segment.observed
            and segment.start_sec is not None
            and segment.onset_sec is not None
            and segment.end_sec is not None
            and segment.correctness in CORRECTNESS_INDEX
        ),
        key=lambda value: (
            float(value.start_sec),
            float(value.onset_sec),
            value.segment_id,
        ),
    )
    for segment in ordered:
        if segment.hand not in role_index:
            raise ValueError(
                f"{episode.video_id}/{segment.segment_id}: expected a left/right role, "
                f"got {segment.hand!r}"
            )
        role = role_index[str(segment.hand)]
        indices = np.flatnonzero(
            (timestamps >= float(segment.start_sec))
            & (timestamps <= float(segment.end_sec))
        )
        if not len(indices):
            midpoint = 0.5 * (
                float(segment.start_sec) + float(segment.end_sec)
            )
            indices = np.asarray([_nearest(timestamps, midpoint)], dtype=np.int64)
        start = int(indices[0])
        end = int(indices[-1])
        # Keep the onset inside the rasterized event.  This handles a contact
        # annotated at the final native frame without inventing an extra frame.
        onset = int(
            indices[
                np.abs(timestamps[indices] - float(segment.onset_sec)).argmin()
            ]
        )
        if np.any(phase[start : end + 1, role] != EVENT_PHASES.index("idle")):
            raise ValueError(
                f"{episode.video_id}/{segment.segment_id}: overlapping events for "
                f"the same role"
            )
        phase[indices[indices < onset], role] = EVENT_PHASES.index("approach")
        phase[indices[indices >= onset], role] = EVENT_PHASES.index("interaction")
        duration = len(indices)
        for name, index in zip(EVENT_FACTORS, _factor_indices(segment, intent_space)):
            factor_targets[name][indices, role] = index
        factor_weights[indices, role] = 1.0 / duration
        correctness = CORRECTNESS_INDEX[str(segment.correctness)]
        if correctness == CORRECTNESS_INDEX["error"]:
            anomaly[indices] = True
        elif correctness == CORRECTNESS_INDEX["correction"]:
            recovery[indices] = True
        intervals.append(
            StreamingEventInterval(
                segment_id=segment.segment_id,
                role=role,
                start=start,
                onset=onset,
                end=end,
                correctness=correctness,
                error_types=tuple(segment.error_types),
            )
        )

    return CompetitiveRoleStream(
        episode_id=f"{episode.dataset}:{episode.video_id}",
        video_id=episode.video_id,
        global_features=torch.from_numpy(global_features.copy()),
        role_features=torch.from_numpy(role_features.copy()),
        role_change_features=(
            None
            if role_change_features is None
            else torch.from_numpy(role_change_features.copy())
        ),
        frame_ids=torch.from_numpy(frame_ids.copy()),
        timestamps_sec=torch.from_numpy(timestamps.copy()),
        phase_targets=torch.from_numpy(phase),
        factor_targets={
            name: torch.from_numpy(value) for name, value in factor_targets.items()
        },
        factor_weights=torch.from_numpy(factor_weights),
        anomaly_target=torch.from_numpy(anomaly),
        recovery_target=torch.from_numpy(recovery),
        intervals=tuple(intervals),
    )


def load_competitive_role_streams(
    episodes: Sequence[CanonicalEpisode],
    feature_root: Path,
    intent_space: CompositionalIntentSpace,
    *,
    file_pattern: str = "{video_id}_stream_features.npz",
) -> tuple[CompetitiveRoleStream, ...]:
    streams = tuple(
        load_competitive_role_stream(
            episode,
            feature_root,
            intent_space,
            file_pattern=file_pattern,
        )
        for episode in episodes
    )
    if not streams:
        raise ValueError("no competitive-role stream is available")
    dimensions = {stream.global_features.shape[-1] for stream in streams}
    if len(dimensions) != 1:
        raise ValueError("feature dimensions differ across streams")
    return streams
