"""Induce task-specific continuation languages from fold-training videos.

Complete successful recordings are preferred.  If a training task has no
complete success recording, a deterministic sequence medoid is selected from
all of that task's decoded training recordings.  The fallback is label-free at
the event level and avoids dropping an otherwise evaluable task or admitting
every low-frequency execution path into the normative language.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import torch

from provia.data.completion_stream import CompletionStream, load_completion_streams
from provia.data.folds import materialize_cross_validation_fold
from provia.data.io import read_jsonl
from provia.observer.completion_process import (
    MarkedCompletionProcess,
    build_marked_completion_process,
    decode_completion_posterior_medians,
)
from provia.observer.factors import EVENT_FACTORS
from provia.training.train_single_actor_observer import _emit
from provia.training.common import _jeffreys_prior, _sha256
from provia.model.intents import CompositionalIntentSpace
from provia.model.interval_transport import (
    CommittedReferenceAction,
    compile_interval_continuations,
)
from provia.model.reference_event import UncertainReferenceEvent


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _priors(metadata: Mapping[str, object]) -> dict[str, torch.Tensor]:
    raw = metadata.get("event_balanced_factor_counts")
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint lacks factor counts")
    return {
        name: _jeffreys_prior(
            torch.tensor(raw[name], dtype=torch.float64),  # type: ignore[arg-type]
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        for name in EVENT_FACTORS
    }


def _trace_signature(
    trace: Sequence[CommittedReferenceAction],
) -> tuple[tuple[str, int, int, int], ...]:
    """Return the role and factor modes used only for reference selection."""

    return tuple(
        (
            action.role,
            int(torch.tensor(action.event.verb).argmax()),
            int(torch.tensor(action.event.noun).argmax()),
            int(torch.tensor(action.event.tool).argmax()),
        )
        for action in trace
    )


def _normalized_sequence_distance(
    first: Sequence[object], second: Sequence[object]
) -> float:
    """Levenshtein distance normalized by the longer decoded sequence."""

    if not first and not second:
        return 0.0
    previous = list(range(len(second) + 1))
    for row, left in enumerate(first, start=1):
        current = [row]
        for column, right in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + int(left != right),
                )
            )
        previous = current
    return float(previous[-1]) / float(max(len(first), len(second)))


def _medoid_index(
    video_ids: Sequence[str], signatures: Sequence[Sequence[object]]
) -> tuple[int, tuple[float, ...]]:
    """Select the minimum-total-distance observed trace without a threshold."""

    if not video_ids or len(video_ids) != len(signatures):
        raise ValueError("medoid selection requires aligned non-empty candidates")
    total_distance = tuple(
        sum(
            _normalized_sequence_distance(signature, other)
            for other_index, other in enumerate(signatures)
            if other_index != index
        )
        for index, signature in enumerate(signatures)
    )
    selected = min(
        range(len(video_ids)),
        key=lambda index: (total_distance[index], str(video_ids[index])),
    )
    return selected, total_distance


@torch.no_grad()
def _trace(
    model: MarkedCompletionProcess,
    stream: CompletionStream,
    priors: Mapping[str, torch.Tensor],
    device: torch.device,
) -> tuple[CommittedReferenceAction, ...]:
    evidence = _emit(model, stream, device)
    probability = model.event_probability(
        evidence.intensity_per_second,
        stream.timestamps_sec.to(device=device, dtype=torch.float32),
    )
    events = decode_completion_posterior_medians(
        probability.cpu(),
        {name: values.cpu() for name, values in evidence.factor_logits.items()},
        priors,
    )
    return tuple(
        CommittedReferenceAction(
            role="left",
            start=float(event.start_index),
            commitment=float(event.onset_index),
            end=float(event.end_index),
            event=UncertainReferenceEvent(
                verb=tuple(
                    math.exp(value)
                    for value in event.factor_log_probability["verb"]
                ),
                noun=tuple(
                    math.exp(value)
                    for value in event.factor_log_probability["noun"]
                ),
                tool=tuple(
                    math.exp(value)
                    for value in event.factor_log_probability["tool"]
                ),
            ),
        )
        for event in events
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("observer_architecture")
        not in {
            "marked_completion_process",
            "completion_process_observer",
            "token_completion_process",
            "out_of_fold_completion_process",
        }
        or metadata.get("test_split_accessed") is not False
        or int(metadata.get("fold", -1)) != args.fold
    ):
        raise ValueError("checkpoint is not a completion observer")
    episodes = materialize_cross_validation_fold(
        tuple(read_jsonl(args.manifest)), args.fold
    )
    train_native = tuple(episode for episode in episodes if episode.split == "train")
    if not train_native:
        raise ValueError("training partition is empty")
    intent_space = CompositionalIntentSpace.from_segments(
        segment for episode in train_native for segment in episode.segments
    )
    labels = {name: intent_space.labels[name] for name in EVENT_FACTORS}
    cache = metadata.get("cache")
    input_dim = int(cache["global_dim"]) if isinstance(cache, Mapping) else 384
    model = build_marked_completion_process(metadata, input_dim, labels)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model.to(device).eval()
    priors = _priors(metadata)

    successful_native = tuple(
        episode
        for episode in train_native
        if episode.segments
        and all(segment.correctness == "correct" for segment in episode.segments)
    )
    train_tasks = {episode.task_id for episode in train_native}
    successful_tasks = {episode.task_id for episode in successful_native}
    fallback_tasks = train_tasks - successful_tasks
    reference_candidates = successful_native + tuple(
        episode for episode in train_native if episode.task_id in fallback_tasks
    )
    reference_streams = load_completion_streams(
        reference_candidates, args.feature_root, intent_space
    )
    candidates_by_task: dict[
        str, list[tuple[str, tuple[CommittedReferenceAction, ...]]]
    ] = defaultdict(list)
    for stream in reference_streams:
        trace = _trace(model, stream, priors, device)
        if not trace:
            raise RuntimeError(
                f"reference candidate {stream.video_id} decoded no completion"
            )
        candidates_by_task[stream.task_id].append((stream.video_id, trace))

    by_task: dict[str, list[tuple[str, tuple[CommittedReferenceAction, ...]]]] = {}
    selection_audit: dict[str, dict[str, object]] = {}
    for task in sorted(train_tasks):
        candidates = sorted(candidates_by_task[task], key=lambda item: item[0])
        if not candidates:
            raise RuntimeError(f"task {task} has no decoded training reference")
        if task in successful_tasks:
            by_task[task] = candidates
            selection_audit[task] = {
                "mode": "all_complete_successes",
                "candidate_videos": [video_id for video_id, _ in candidates],
                "selected_videos": [video_id for video_id, _ in candidates],
            }
        else:
            selected, distances = _medoid_index(
                [video_id for video_id, _ in candidates],
                [_trace_signature(trace) for _, trace in candidates],
            )
            by_task[task] = [candidates[selected]]
            selection_audit[task] = {
                "mode": "decoded_training_sequence_medoid",
                "candidate_videos": [video_id for video_id, _ in candidates],
                "selected_videos": [candidates[selected][0]],
                "total_normalized_edit_distance": {
                    video_id: distance
                    for (video_id, _), distance in zip(candidates, distances)
                },
                "event_correctness_used_for_medoid": False,
            }

    registry = {}
    for task, references in sorted(by_task.items()):
        language = compile_interval_continuations(
            [trace for _, trace in references],
            factor_labels=labels,
        )
        filename = f"task_{task}.language.json"
        _atomic_json(args.output_dir / filename, language.to_dict())
        registry[task] = {
            "language": filename,
            "expert_videos": [video_id for video_id, _ in references],
            "expert_video_count": len(references),
            "decoded_event_counts": [len(trace) for _, trace in references],
            "states": language.state_count,
            "transitions": len(language.transitions),
            "event_types": len(language.node_ids),
            "source_traces": language.source_trace_count,
            
        }
    audit = {
        "schema": "automatic_completion_language_registry",
        "method": "PROVIA",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "manifest": str(args.manifest.resolve()),
        "feature_root": str(args.feature_root.resolve()),
        "fold": args.fold,
        "expert_event_annotations_used": False,
        "expert_video_success_status_used": True,
        "fallback_tasks": sorted(fallback_tasks),
        "fallback_event_correctness_used": False,
        "fallback_hyperparameters": 0,
        "validation_videos_used": False,
        "test_videos_used": False,
        "tasks": registry,
    }
    _atomic_json(args.output_dir / "registry.json", audit)
    _atomic_json(
        args.output_dir / "complete.json",
        {"status": "complete", "test_split_accessed": False},
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
