"""Decode a frozen observer on a query split without fitting any parameter.

Candidate decoding receives an annotation-free visual stream.  Native events
are read only after decoding to create metric assignments and targets; they do
not affect boundaries, marks, parameters, checkpoint selection, or thresholds.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import torch

from provia.data.completion_stream import load_completion_streams
from provia.data.folds import materialize_cross_validation_fold
from provia.data.io import read_jsonl
from provia.evaluation.single_actor_query import COMPLETION_QUERY_SCHEMA
from provia.observer.completion_process import build_marked_completion_process
from provia.observer.factors import EVENT_FACTORS
from provia.training.completion_metrics import _priors
from provia.training.prepare_single_actor_examples import _sequences
from provia.training.common import _partitions, _sha256
from provia.model.intents import CompositionalIntentSpace


def _query_cache(root: Path, fold: int, split: str) -> Mapping[str, object]:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != "role_stream_features"
        or int(value.get("fold", -1)) != fold
        or set(value.get("partitions", ())) != {split}
        or value.get("contains_annotations") is not False
        or value.get("test_split_accessed") is not (split == "test")
    ):
        raise ValueError("query feature cache is malformed or split-incompatible")
    return {"path": str(path.resolve()), "sha256": _sha256(path), **value}


def _labels(source: Mapping[str, object]) -> dict[str, tuple[str, ...]]:
    raw = source.get("factor_labels")
    if not isinstance(raw, Mapping):
        raise ValueError("source examples do not define the intent ontology")
    return {
        name: tuple(str(value) for value in raw[name])  # type: ignore[index]
        for name in EVENT_FACTORS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-examples", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    development = torch.load(
        args.source_examples, map_location="cpu", weights_only=False
    )
    source = development.get("metadata") if isinstance(development, Mapping) else None
    if (
        not isinstance(source, Mapping)
        or source.get("schema") != "completion_examples"
        or source.get("test_split_accessed") is not False
        or not isinstance(development.get("train"), Sequence)
        or not isinstance(development.get("validation"), Sequence)
    ):
        raise ValueError("source development examples are malformed")
    fold = int(source["fold"])
    if args.manifest.resolve() != Path(str(source["manifest"])).resolve():
        raise ValueError("query manifest differs from the frozen development manifest")

    episodes = tuple(read_jsonl(args.manifest))
    train_native, _ = _partitions(episodes, fold)
    materialized = materialize_cross_validation_fold(episodes, fold)
    query_native = tuple(item for item in materialized if item.split == args.split)
    if not query_native:
        raise ValueError(f"manifest has no {args.split} partition")
    intent = CompositionalIntentSpace.from_segments(
        segment for episode in train_native for segment in episode.segments
    )
    expected_labels = _labels(source)
    actual_labels = {name: tuple(intent.labels[name]) for name in EVENT_FACTORS}
    if actual_labels != expected_labels:
        raise ValueError("training ontology changed after development examples were frozen")

    cache = _query_cache(args.feature_root, fold, args.split)
    source_cache = source.get("cache")
    if (
        not isinstance(source_cache, Mapping)
        or int(cache.get("global_dim", -1)) != int(source_cache.get("global_dim", -2))
    ):
        raise ValueError("query and development feature dimensions disagree")
    observer_path = Path(str(source["observer_checkpoint"]))
    if _sha256(observer_path) != source.get("observer_checkpoint_sha256"):
        raise ValueError("frozen observer checkpoint changed")
    payload = torch.load(observer_path, map_location="cpu", weights_only=False)
    observer_metadata = payload.get("metadata")
    if (
        not isinstance(observer_metadata, Mapping)
        or observer_metadata.get("test_split_accessed") is not False
        or int(observer_metadata.get("fold", -1)) != fold
    ):
        raise ValueError("observer checkpoint does not belong to this fold")
    observer = build_marked_completion_process(
        observer_metadata, int(source_cache["global_dim"]), expected_labels
    )
    observer.load_state_dict(payload["model"], strict=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    observer.to(device).eval()
    streams = load_completion_streams(query_native, args.feature_root, intent)
    task_ids = tuple(str(value) for value in source["task_ids"])  # type: ignore[index]
    task_lookup = {task: index for index, task in enumerate(task_ids)}
    unseen = {stream.task_id for stream in streams} - set(task_ids)
    if unseen:
        raise ValueError(f"query contains unseen tasks: {sorted(unseen)}")
    with torch.no_grad():
        sequences = _sequences(
            observer, streams, _priors(observer_metadata), task_lookup, device
        )

    metadata = {
        "schema": COMPLETION_QUERY_SCHEMA,
        "source_examples": str(args.source_examples.resolve()),
        "source_examples_sha256": _sha256(args.source_examples),
        "source_examples_schema": source["schema"],
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "feature_root": str(args.feature_root.resolve()),
        "cache": cache,
        "fold": fold,
        "split": args.split,
        "task_ids": list(task_ids),
        "observer_checkpoint": str(observer_path.resolve()),
        "observer_checkpoint_sha256": _sha256(observer_path),
        "parameters_fitted": False,
        "query_annotations_used_by_candidate_decoder": False,
        "query_annotations_used_for_evaluation_only": True,
        "query_sequences": len(sequences),
        "query_decoded_candidates": sum(
            len(value["origin_target"]) for value in sequences
        ),
        "query_native_events": sum(len(value["native_mode"]) for value in sequences),
        "test_split_accessed": args.split == "test",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    torch.save({"metadata": metadata, args.split: sequences}, temporary)
    os.replace(temporary, args.output)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
