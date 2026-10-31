"""Compile causal, annotation-free observer evidence for a frozen query split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch

from provia.automaton.prefix_evidence_single_actor import _compile_video
from provia.evaluation.single_actor_query import (
    COMPLETION_QUERY_PREFIX_SCHEMA,
    load_completion_query_artifact,
)
from provia.observer.completion_process import build_marked_completion_process
from provia.observer.factors import EVENT_FACTORS
from provia.training.completion_metrics import _priors
from provia.training.common import _atomic_json, _sha256


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

    query, sequences = load_completion_query_artifact(
        args.query_examples,
        source_examples=args.source_examples,
        split=args.split,
    )
    development = torch.load(
        args.source_examples, map_location="cpu", weights_only=False
    )
    source = development.get("metadata") if isinstance(development, Mapping) else None
    if not isinstance(source, Mapping) or source.get("test_split_accessed") is not False:
        raise ValueError("development examples are malformed")
    labels = {
        name: tuple(str(value) for value in source["factor_labels"][name])  # type: ignore[index]
        for name in EVENT_FACTORS
    }
    observer_path = Path(str(source["observer_checkpoint"]))
    if (
        _sha256(observer_path) != source.get("observer_checkpoint_sha256")
        or query.get("observer_checkpoint_sha256") != _sha256(observer_path)
    ):
        raise ValueError("query and development observer provenance disagree")
    payload = torch.load(observer_path, map_location="cpu", weights_only=False)
    observer_metadata = payload.get("metadata")
    cache = source.get("cache")
    if (
        not isinstance(observer_metadata, Mapping)
        or observer_metadata.get("test_split_accessed") is not False
        or not isinstance(cache, Mapping)
    ):
        raise ValueError("frozen observer metadata are malformed")
    observer = build_marked_completion_process(
        observer_metadata, int(cache["global_dim"]), labels
    )
    observer.load_state_dict(payload["model"], strict=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    observer.to(device).eval()
    priors = _priors(observer_metadata)
    feature_root = Path(str(query["feature_root"]))
    video_ids = tuple(sorted({str(sequence["video_id"]) for sequence in sequences}))
    records = {}
    for index, video_id in enumerate(video_ids, start=1):
        feature_path = feature_root / f"{video_id}_stream_features.npz"
        output_path = args.output_dir / f"{video_id}_prefix_evidence.npz"
        records[video_id] = _compile_video(
            observer, priors, feature_path, output_path, device
        )
        print(
            json.dumps({"compiled": index, "total": len(video_ids), "video_id": video_id}),
            flush=True,
        )
    registry = {
        "schema": COMPLETION_QUERY_PREFIX_SCHEMA,
        "query_examples": str(args.query_examples.resolve()),
        "query_examples_sha256": _sha256(args.query_examples),
        "source_examples": str(args.source_examples.resolve()),
        "source_examples_sha256": _sha256(args.source_examples),
        "observer_checkpoint": str(observer_path.resolve()),
        "observer_checkpoint_sha256": _sha256(observer_path),
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
