"""Lightweight streams for causal single-actor completion modeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from provia.data.schema import CanonicalEpisode
from provia.data.streaming_events import CORRECTNESS_INDEX
from provia.observer.factors import EVENT_FACTORS
from provia.model.intents import CompositionalIntentSpace, segment_factors


@dataclass(frozen=True)
class CompletionStream:
    episode_id: str
    video_id: str
    task_id: str
    global_features: Tensor
    change_features: Tensor
    frame_ids: Tensor
    timestamps_sec: Tensor
    event_indices: Tensor
    factor_targets: Mapping[str, Tensor]
    correctness: Tensor
    error_types: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class CompletionInferenceStream:
    """Annotation-free view admitted to query-time visual inference."""

    episode_id: str
    video_id: str
    task_id: str
    global_features: Tensor
    change_features: Tensor
    frame_ids: Tensor
    timestamps_sec: Tensor


def completion_inference_stream(
    stream: CompletionStream,
) -> CompletionInferenceStream:
    """Remove every native event, semantic, and correctness annotation."""

    return CompletionInferenceStream(
        episode_id=stream.episode_id,
        video_id=stream.video_id,
        task_id=stream.task_id,
        global_features=stream.global_features,
        change_features=stream.change_features,
        frame_ids=stream.frame_ids,
        timestamps_sec=stream.timestamps_sec,
    )


def _nearest(values: np.ndarray, target: float) -> int:
    position = int(np.searchsorted(values, target, side="left"))
    if position <= 0:
        return 0
    if position >= len(values):
        return len(values) - 1
    before = position - 1
    return before if target - values[before] <= values[position] - target else position


def load_completion_stream(
    episode: CanonicalEpisode,
    feature_root: Path,
    intent_space: CompositionalIntentSpace,
    *,
    file_pattern: str = "{video_id}_stream_features.npz",
) -> CompletionStream:
    """Load only arrays required by the completion-process reduction."""

    path = feature_root / file_pattern.format(video_id=episode.video_id)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "frame_ids",
            "timestamps_sec",
            "global_features",
            "role_features",
            "role_change_features",
        }
        if set(payload.files) != required:
            raise ValueError(f"{path}: incompatible single-actor cache")
        frame_ids = np.asarray(payload["frame_ids"], dtype=np.int64)
        timestamps = np.asarray(payload["timestamps_sec"], dtype=np.float64)
        global_features = np.asarray(payload["global_features"], dtype=np.float32)
        change_features = np.asarray(
            payload["role_change_features"][:, 0], dtype=np.float32
        )
    length = len(frame_ids)
    if (
        frame_ids.ndim != 1
        or timestamps.shape != (length,)
        or global_features.ndim != 2
        or change_features.shape != global_features.shape
        or not length
        or np.any(np.diff(frame_ids) <= 0)
        or np.any(np.diff(timestamps) <= 0)
        or not np.isfinite(global_features).all()
        or not np.isfinite(change_features).all()
    ):
        raise ValueError(f"{path}: malformed completion stream")
    segments = tuple(
        sorted(
            (
                segment
                for segment in episode.segments
                if segment.observed
                and segment.end_sec is not None
                and segment.correctness in CORRECTNESS_INDEX
            ),
            key=lambda item: (float(item.end_sec), item.segment_id),
        )
    )
    if not segments or any(segment.hand != "left" for segment in segments):
        raise ValueError(
            f"{episode.episode_id}: completion projection needs an active left actor"
        )
    event_indices = np.asarray(
        [_nearest(timestamps, float(segment.end_sec)) for segment in segments],
        dtype=np.int64,
    )
    if len(set(event_indices.tolist())) != len(event_indices):
        raise ValueError(
            f"{episode.episode_id}: simultaneous completions were not compounded"
        )
    factor_values = {name: [] for name in EVENT_FACTORS}
    for segment in segments:
        for name, value in zip(EVENT_FACTORS, segment_factors(segment)[1:]):
            label = value if value in intent_space.labels[name] else "<unk>"
            factor_values[name].append(intent_space.labels[name].index(label))
    return CompletionStream(
        episode_id=f"{episode.dataset}:{episode.video_id}",
        video_id=episode.video_id,
        task_id=episode.task_id,
        global_features=torch.from_numpy(global_features.copy()),
        change_features=torch.from_numpy(change_features.copy()),
        frame_ids=torch.from_numpy(frame_ids.copy()),
        timestamps_sec=torch.from_numpy(timestamps.copy()),
        event_indices=torch.from_numpy(event_indices),
        factor_targets={
            name: torch.tensor(values, dtype=torch.long)
            for name, values in factor_values.items()
        },
        correctness=torch.tensor(
            [CORRECTNESS_INDEX[str(segment.correctness)] for segment in segments],
            dtype=torch.long,
        ),
        error_types=tuple(tuple(segment.error_types) for segment in segments),
    )


def load_completion_streams(
    episodes: Sequence[CanonicalEpisode],
    feature_root: Path,
    intent_space: CompositionalIntentSpace,
) -> tuple[CompletionStream, ...]:
    return tuple(
        load_completion_stream(episode, feature_root, intent_space)
        for episode in episodes
    )
