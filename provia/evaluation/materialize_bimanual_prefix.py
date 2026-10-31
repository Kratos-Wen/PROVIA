"""Compile causal bimanual observer evidence for a frozen query split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch

from provia.automaton.prefix_evidence_bimanual import _compile_stream
from provia.data.folds import materialize_cross_validation_fold
from provia.data.io import read_jsonl
from provia.data.role_stream import load_competitive_role_streams
from provia.evaluation.bimanual_query import (
    ROLE_QUERY_PREFIX_SCHEMA,
    load_role_query_artifact,
)
from provia.observer.factors import EVENT_FACTORS
from provia.observer.build import ROLE_EVENT_OBSERVER_ARCHITECTURES, build_role_event_observer
from provia.training.common import _atomic_json, _sha256
from provia.training.role_transport_fit import _priors
from provia.model.intents import CompositionalIntentSpace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-examples", type=Path, required=True)
    parser.add_argument("--query-examples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    query, sequences = load_role_query_artifact(
        args.query_examples,
        source_examples=args.source_examples,
        split=args.split,
    )
    development = torch.load(
        args.source_examples, map_location="cpu", weights_only=False
    )
    source = development.get("metadata") if isinstance(development, Mapping) else None
    if (
        not isinstance(source, Mapping)
        or source.get("schema") != "role_windowed_examples"
        or source.get("test_split_accessed") is not False
    ):
        raise ValueError("development role examples are malformed")

    episodes = materialize_cross_validation_fold(
        tuple(read_jsonl(Path(str(source["manifest"])))), int(source["fold"])
    )
    train = tuple(value for value in episodes if value.split == "train")
    selected = tuple(value for value in episodes if value.split == args.split)
    intent = CompositionalIntentSpace.from_segments(
        segment for episode in train for segment in episode.segments
    )
    labels = {name: tuple(intent.labels[name]) for name in EVENT_FACTORS}
    if any(
        tuple(source["factor_labels"][name]) != labels[name]  # type: ignore[index]
        for name in EVENT_FACTORS
    ):
        raise ValueError("query intent vocabulary differs from development")
    feature_root = Path(str(query["feature_root"]))
    streams = load_competitive_role_streams(selected, feature_root, intent)
    by_video = {str(stream.video_id): stream for stream in streams}
    required = {str(sequence["video_id"]) for sequence in sequences}
    if set(by_video) != required:
        raise ValueError("query streams and frozen decoded candidates disagree")

    checkpoint_path = Path(str(source["observer_checkpoint"]))
    if (
        _sha256(checkpoint_path) != source.get("observer_checkpoint_sha256")
        or query.get("observer_checkpoint_sha256") != _sha256(checkpoint_path)
    ):
        raise ValueError("query and development observer provenance disagree")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    metadata = checkpoint.get("metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("observer_architecture")
        not in ROLE_EVENT_OBSERVER_ARCHITECTURES
        or metadata.get("test_split_accessed") is not False
        or int(metadata.get("fold", -1)) != int(source["fold"])
    ):
        raise ValueError("frozen role observer is incompatible")
    observer = build_role_event_observer(metadata, int(source["input_dim"]), labels)
    observer.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    observer.to(device).eval()
    priors = _priors(metadata)

    records = {}
    for index, video_id in enumerate(sorted(required), start=1):
        output = args.output_dir / f"video_{index:04d}.npz"
        records[video_id] = _compile_stream(
            observer,
            metadata,
            priors,
            by_video[video_id],
            output,
            device,
        )
        print(
            json.dumps(
                {"compiled": index, "total": len(required), "video_id": video_id}
            ),
            flush=True,
        )
    registry = {
        "schema": ROLE_QUERY_PREFIX_SCHEMA,
        "query_examples": str(args.query_examples.resolve()),
        "query_examples_sha256": _sha256(args.query_examples),
        "source_examples": str(args.source_examples.resolve()),
        "source_examples_sha256": _sha256(args.source_examples),
        "observer_checkpoint": str(checkpoint_path.resolve()),
        "observer_checkpoint_sha256": _sha256(checkpoint_path),
        "fold": int(source["fold"]),
        "split": args.split,
        "causal": True,
        "parameters_fitted": False,
        "annotations_used_to_compute_evidence": False,
        "test_split_accessed": args.split == "test",
        "videos": records,
    }
    _atomic_json(args.output_dir / "registry.json", registry)
    print(json.dumps({"output": str(args.output_dir), "videos": len(records)}))


if __name__ == "__main__":
    main()
