"""Induce Bayesian predictive-state procedures from successful videos only.

The source expert videos and their frozen causal event decodes are fixed by an
existing fold-training registry.  No anomaly label, validation video, or test
video is inspected.  Exact demonstration paths are replaced by the
Jeffreys-evidence predictive quotient, which shares histories only when one
next-event law has greater integrated evidence than separate laws.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor

from provia.automaton.compile_languages import _atomic_json
from provia.observer.factors import EVENT_FACTORS
from provia.training.common import _sha256
from provia.model.predictive_state_induction import (
    JEFFREYS_CONCENTRATION,
    quotient_interval_language,
)
from provia.model.interval_transport import IntervalContinuationLanguage


def _read_mapping(path: Path) -> Mapping[str, object]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain one mapping")
    return value


def _source_language(
    source: Mapping[str, object], task: str
) -> tuple[Path, IntervalContinuationLanguage]:
    records = source.get("languages")
    if not isinstance(records, Mapping):
        raise ValueError("examples lack source languages")
    record = records.get(task)
    if not isinstance(record, Mapping):
        raise ValueError(f"task {task} lacks a source language")
    path = Path(str(record["path"]))
    if not path.is_file() or _sha256(path) != record.get("sha256"):
        raise ValueError(f"task {task} source language is missing or changed")
    return path, IntervalContinuationLanguage.from_dict(_read_mapping(path))


def _trace_nodes(
    sequence: Mapping[str, object],
    language: IntervalContinuationLanguage,
    labels: Mapping[str, Sequence[str]],
) -> tuple[int, ...]:
    factors = {}
    for name in EVENT_FACTORS:
        value = sequence.get(f"factor_{name}")
        if not isinstance(value, Tensor) or value.ndim != 2:
            raise ValueError("expert sequence lacks frozen factor probabilities")
        factors[name] = value
    events = len(factors[EVENT_FACTORS[0]])
    if not events or any(len(value) != events for value in factors.values()):
        raise ValueError("expert factor probabilities are empty or misaligned")
    node_index = {node_id: index for index, node_id in enumerate(language.node_ids)}
    result = []
    for event in range(events):
        verb = str(labels["verb"][int(factors["verb"][event].argmax())])
        noun = str(labels["noun"][int(factors["noun"][event].argmax())])
        identity = f"left|{verb}|{noun}"
        node = node_index.get(identity)
        if node is None:
            raise ValueError(
                f"decoded expert event {identity} is absent from its source language"
            )
        result.append(node)
    return tuple(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    artifact = torch.load(args.examples, map_location="cpu", weights_only=False)
    source = artifact.get("metadata") if isinstance(artifact, Mapping) else None
    train = artifact.get("train") if isinstance(artifact, Mapping) else None
    if (
        not isinstance(source, Mapping)
        or source.get("schema") != "completion_examples"
        or source.get("test_split_accessed") is not False
        or not isinstance(train, Sequence)
        or not train
    ):
        raise ValueError("examples are malformed")
    registry = _read_mapping(args.source_registry)
    if (
        registry.get("schema") != "automatic_completion_language_registry"
        or registry.get("validation_videos_used") is not False
        or registry.get("test_videos_used") is not False
        or registry.get("expert_event_annotations_used") is not False
    ):
        raise ValueError("source expert registry is malformed or leaks evaluation data")
    task_records = registry.get("tasks")
    labels_raw = source.get("factor_labels")
    if not isinstance(task_records, Mapping) or not isinstance(labels_raw, Mapping):
        raise ValueError("source registry lacks tasks or event ontology")
    labels = {
        name: tuple(str(value) for value in labels_raw[name])  # type: ignore[index]
        for name in EVENT_FACTORS
    }
    by_video = {}
    for sequence in train:
        if not isinstance(sequence, Mapping):
            raise ValueError("training sequence must be a mapping")
        video_id = str(sequence.get("video_id"))
        if video_id in by_video:
            raise ValueError(f"duplicate training video {video_id}")
        by_video[video_id] = sequence

    tasks = {}
    for task in (str(value) for value in source["task_ids"]):  # type: ignore[index]
        record = task_records.get(task)
        if not isinstance(record, Mapping):
            raise ValueError(f"task {task} lacks expert-video provenance")
        expert_videos = tuple(str(value) for value in record["expert_videos"])  # type: ignore[index]
        expected_counts = tuple(int(value) for value in record["decoded_event_counts"])  # type: ignore[index]
        if not expert_videos or len(expert_videos) != len(expected_counts):
            raise ValueError(f"task {task} has malformed expert-video provenance")
        source_path, source_language = _source_language(source, task)
        traces = []
        for video_id, expected in zip(expert_videos, expected_counts):
            sequence = by_video.get(video_id)
            if sequence is None:
                raise ValueError(f"expert video {video_id} is absent from training")
            if str(sequence.get("task_id")) != task:
                raise ValueError(f"expert video {video_id} has the wrong task")
            trace = _trace_nodes(sequence, source_language, labels)
            if len(trace) != expected:
                raise ValueError(
                    f"expert video {video_id} decode changed: {len(trace)} != {expected}"
                )
            traces.append(trace)
        language, quotient = quotient_interval_language(source_language, traces)
        filename = f"task_{task}.language.json"
        path = args.output_dir / filename
        _atomic_json(path, language.to_dict())
        tasks[task] = {
            "language": filename,
            "sha256": _sha256(path),
            "source_language": str(source_path.resolve()),
            "source_language_sha256": _sha256(source_path),
            "expert_videos": list(expert_videos),
            "source_traces": len(traces),
            "source_exact_states": source_language.state_count,
            "source_process_states": quotient.source_state_count,
            "predictive_states": quotient.state_count,
            "merged_source_states": quotient.merged_state_count,
            "log_bayes_evidence_gain": quotient.log_bayes_evidence_gain,
            "latent_same_event_destinations": sum(
                1
                for source_state, node in {
                    (edge[0], edge[1]) for edge in quotient.transitions
                }
                if sum(
                    edge[0] == source_state and edge[1] == node
                    for edge in quotient.transitions
                )
                > 1
            ),
        }

    audit = {
        "schema": "bayesian_predictive_language_registry",
        "method": "factorized Jeffreys-evidence predictive-state quotient",
        "source_examples": str(args.examples.resolve()),
        "source_examples_sha256": _sha256(args.examples),
        "source_registry": str(args.source_registry.resolve()),
        "source_registry_sha256": _sha256(args.source_registry),
        "fold": int(source["fold"]),
        "unvisited_valid_source_states_preserved": True,
        "repetition_or_revisit_cycles_preserved": True,
        "model_prior_odds": 1.0,
        "categorical_prior": {
            "name": "Jeffreys",
            "concentration_per_outcome": JEFFREYS_CONCENTRATION,
        },
        "similarity_threshold": False,
        "selected_state_count": False,
        "anomaly_labels_used": False,
        "expert_event_annotations_used": False,
        "validation_videos_used": False,
        "test_videos_used": False,
        "tasks": tasks,
    }
    _atomic_json(args.output_dir / "registry.json", audit)
    _atomic_json(
        args.output_dir / "complete.json",
        {"status": "complete", "test_split_accessed": False},
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
