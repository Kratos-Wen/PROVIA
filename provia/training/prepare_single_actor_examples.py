"""Freeze sequence-level examples for reliability-aware causal training.

Only fold-training and validation annotations are read.  Within each native
completion Voronoi cell, the chronologically decoded candidate nearest to the
native completion is the unique genuine match; all other decoded candidates
are observer insertions.  This deterministic one-to-one reduction fits the
edit channel without a timing tolerance or anomaly-dependent matching rule.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from provia.data.completion_stream import (
    CompletionStream,
    completion_inference_stream,
    load_completion_streams,
)
from provia.data.io import read_jsonl
from provia.observer.completion_process import build_marked_completion_process
from provia.observer.factors import EVENT_FACTORS
from provia.observer.trajectory_statistics import trajectory_statistics
from provia.training.completion_metrics import (
    _cell_assignment,
    _decode,
    _fit_deletion_probability,
    _fit_mode_dynamics,
    _priors,
    _read_language,
)
from provia.training.common import (
    _cache_provenance,
    _partitions,
    _sha256,
)
from provia.model.intents import CompositionalIntentSpace


def _matched_candidates(
    native_indices: np.ndarray,
    decoded_indices: np.ndarray,
    timestamps_sec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return native-cell assignments and one deterministic match per cell."""

    native = np.asarray(native_indices, dtype=np.int64)
    decoded = np.asarray(decoded_indices, dtype=np.int64)
    timestamps = np.asarray(timestamps_sec, dtype=np.float64)
    if native.ndim != 1 or decoded.ndim != 1 or timestamps.ndim != 1:
        raise ValueError("completion matching inputs must be one-dimensional")
    if not len(native):
        raise ValueError("completion matching requires native events")
    if np.any(np.diff(native) <= 0) or np.any(np.diff(decoded) <= 0):
        raise ValueError("native and decoded completion indices must increase")
    if np.any(native < 0) or np.any(native >= len(timestamps)):
        raise ValueError("native completion index is outside the stream")
    if np.any(decoded < 0) or np.any(decoded >= len(timestamps)):
        raise ValueError("decoded completion index is outside the stream")
    assignment = _cell_assignment(native, decoded, timestamps)
    matched = np.zeros(len(decoded), dtype=np.bool_)
    for native_index in range(len(native)):
        candidates = np.flatnonzero(assignment == native_index)
        if not len(candidates):
            continue
        distance = np.abs(
            timestamps[decoded[candidates]] - timestamps[native[native_index]]
        )
        # np.lexsort uses the final key first: distance, then decoded time,
        # then candidate index provide a stable chronology-preserving tie rule.
        order = np.lexsort((candidates, decoded[candidates], distance))
        matched[candidates[order[0]]] = True
    return assignment, matched


def _sequence(
    observer: torch.nn.Module,
    stream: CompletionStream,
    priors: Mapping[str, Tensor],
    task_index: int,
    device: torch.device,
) -> dict[str, object] | None:
    decoded_indices, raw_marks = _decode(
        observer, completion_inference_stream(stream), priors, device
    )
    marks = list(raw_marks)
    if len(decoded_indices) and decoded_indices[0] == 0:
        decoded_indices = decoded_indices[1:]
        marks = marks[1:]
    if not len(decoded_indices):
        return None
    native_indices = stream.event_indices.numpy()
    assignment, genuine = _matched_candidates(
        native_indices, decoded_indices, stream.timestamps_sec.numpy()
    )
    mode = np.full(len(decoded_indices), -1, dtype=np.int64)
    native_mode = stream.correctness.numpy()
    mode[genuine] = native_mode[assignment[genuine]]
    previous, statistic = trajectory_statistics(
        stream.global_features,
        stream.timestamps_sec,
        torch.from_numpy(decoded_indices),
    )
    return {
        "episode_id": stream.episode_id,
        "video_id": stream.video_id,
        "task_id": stream.task_id,
        "task_index": int(task_index),
        "decoded_index": torch.from_numpy(decoded_indices.copy()),
        "decoded_time_sec": stream.timestamps_sec[
            torch.from_numpy(decoded_indices)
        ].float(),
        "native_assignment": torch.from_numpy(assignment.copy()),
        "native_event_index": stream.event_indices.clone(),
        "native_time_sec": stream.timestamps_sec.index_select(
            0, stream.event_indices
        ).float(),
        "native_mode": stream.correctness.clone(),
        "native_error_types": stream.error_types,
        "stream_start_sec": float(stream.timestamps_sec[0]),
        "stream_end_sec": float(stream.timestamps_sec[-1]),
        "previous": previous.float(),
        "statistic": statistic.float(),
        **{
            f"factor_{name}": torch.tensor(
                [mark[name] for mark in marks], dtype=torch.float32
            )
            for name in EVENT_FACTORS
        },
        "origin_target": torch.from_numpy(genuine.copy()),
        "mode_target": torch.from_numpy(mode),
    }


def _sequences(
    observer: torch.nn.Module,
    streams: Sequence[CompletionStream],
    priors: Mapping[str, Tensor],
    task_lookup: Mapping[str, int],
    device: torch.device,
) -> list[dict[str, object]]:
    result = []
    for stream in streams:
        value = _sequence(
            observer, stream, priors, task_lookup[stream.task_id], device
        )
        if value is not None:
            result.append(value)
    if not result:
        raise ValueError("observer decoded no causal event candidates")
    return result


def _normalization(
    sequences: Sequence[Mapping[str, object]],
) -> tuple[Tensor, Tensor]:
    statistics = torch.cat(
        [value["statistic"] for value in sequences]  # type: ignore[list-item]
    ).float()
    mean = statistics.mean(dim=0)
    variance = (statistics - mean).square().mean(dim=0)
    positive = variance[variance > 0]
    reference = positive.mean() if len(positive) else variance.new_tensor(1.0)
    floor = torch.finfo(variance.dtype).eps * reference
    return mean, variance.clamp_min(floor).sqrt()


def _origin_audit(
    sequences: Sequence[Mapping[str, object]],
) -> tuple[float, dict[str, object]]:
    values = torch.cat(
        [value["origin_target"] for value in sequences]  # type: ignore[list-item]
    ).bool()
    genuine = int(values.sum())
    total = int(len(values))
    probability = (genuine + 0.5) / (total + 1.0)
    return probability, {
        "decoded_candidates": total,
        "genuine_matches": genuine,
        "insertions": total - genuine,
        "genuine_probability": probability,
        "timing_tolerance": None,
        "anomaly_labels_used_for_origin": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--observer-checkpoint", type=Path, required=True)
    parser.add_argument("--language-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    episodes = tuple(read_jsonl(args.manifest))
    train_native, validation_native = _partitions(episodes, args.fold)
    intent_space = CompositionalIntentSpace.from_segments(
        segment for episode in train_native for segment in episode.segments
    )
    labels = {name: intent_space.labels[name] for name in EVENT_FACTORS}
    payload = torch.load(
        args.observer_checkpoint, map_location="cpu", weights_only=False
    )
    metadata = payload.get("metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("test_split_accessed") is not False
        or int(metadata.get("fold", -1)) != args.fold
    ):
        raise ValueError("observer checkpoint does not belong to this fold")
    cache = metadata.get("cache")
    input_dim = int(cache["global_dim"]) if isinstance(cache, Mapping) else 384
    observer = build_marked_completion_process(metadata, input_dim, labels)
    observer.load_state_dict(payload["model"], strict=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    observer.to(device).eval()
    priors = _priors(metadata)
    train_streams = load_completion_streams(
        train_native, args.feature_root, intent_space
    )
    validation_streams = load_completion_streams(
        validation_native, args.feature_root, intent_space
    )
    task_ids = tuple(sorted({stream.task_id for stream in train_streams}))
    task_lookup = {task: index for index, task in enumerate(task_ids)}
    unseen = {stream.task_id for stream in validation_streams} - set(task_ids)
    if unseen:
        raise ValueError("validation contains tasks unseen in training")

    with open(args.language_registry, encoding="utf-8") as handle:
        registry = json.load(handle)
    if (
        registry.get("test_videos_used") is not False
        or registry.get("checkpoint_sha256") != _sha256(args.observer_checkpoint)
    ):
        raise ValueError("language registry does not belong to this observer")
    registry_root = args.language_registry.parent
    language_records = {
        str(task): {
            "path": str((registry_root / record["language"]).resolve()),
            "sha256": _sha256(registry_root / record["language"]),
        }
        for task, record in registry["tasks"].items()
    }
    if set(task_ids) - set(language_records):
        raise ValueError("one or more training tasks lack a continuation language")
    # Validate every selected language now; training never recompiles context.
    for record in language_records.values():
        _read_language(Path(record["path"]))

    deletion, deletion_audit = _fit_deletion_probability(
        observer, train_streams, priors, device
    )
    transition, initial, mode_audit = _fit_mode_dynamics(train_streams)
    train = _sequences(observer, train_streams, priors, task_lookup, device)
    validation = _sequences(
        observer, validation_streams, priors, task_lookup, device
    )
    mean, scale = _normalization(train)
    genuine_probability, origin_audit = _origin_audit(train)
    train_events = sum(len(value["origin_target"]) for value in train)
    validation_events = sum(len(value["origin_target"]) for value in validation)
    artifact_metadata = {
        "schema": "completion_examples",
        "fold": args.fold,
        "manifest": str(args.manifest.resolve()),
        "feature_root": str(args.feature_root.resolve()),
        "cache": _cache_provenance(args.feature_root, args.fold),
        "observer_checkpoint": str(args.observer_checkpoint.resolve()),
        "observer_checkpoint_sha256": _sha256(args.observer_checkpoint),
        "language_registry": str(args.language_registry.resolve()),
        "language_registry_sha256": _sha256(args.language_registry),
        "languages": language_records,
        "task_ids": list(task_ids),
        "input_dim": input_dim,
        "statistic_dim": int(mean.numel()),
        "factor_labels": {name: list(labels[name]) for name in EVENT_FACTORS},
        "trajectory_normalization": {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "source": "all fold-training decoded candidates",
        },
        "origin_channel": origin_audit,
        "deletion_channel": deletion_audit,
        "missed_event_probability": deletion,
        "mode_transition_probability": transition.tolist(),
        "mode_initial_probability": initial.tolist(),
        "mode_dynamics": mode_audit,
        "train_sequences": len(train),
        "validation_sequences": len(validation),
        "train_decoded_candidates": train_events,
        "validation_decoded_candidates": validation_events,
        "test_split_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    torch.save(
        {"metadata": artifact_metadata, "train": train, "validation": validation},
        temporary,
    )
    os.replace(temporary, args.output)
    print(
        {
            "output": str(args.output),
            "train_sequences": len(train),
            "validation_sequences": len(validation),
            "train_candidates": train_events,
            "validation_candidates": validation_events,
            "genuine_probability": genuine_probability,
            "missed_event_probability": deletion,
        }
    )


if __name__ == "__main__":
    main()
