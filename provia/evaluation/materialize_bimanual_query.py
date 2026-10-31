"""Decode a bimanual query split with the development observer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping

import torch

from provia.data.folds import materialize_cross_validation_fold
from provia.data.io import read_jsonl
from provia.data.role_stream import load_competitive_role_streams
from provia.evaluation.bimanual_query import ROLE_QUERY_SCHEMA
from provia.observer.factors import EVENT_FACTORS
from provia.observer.build import ROLE_EVENT_OBSERVER_ARCHITECTURES, build_role_event_observer
from provia.training.prepare_bimanual_examples import (
    _causal_onset_events,
    _with_gap_candidates,
    _sequence,
)
from provia.training.common import _sha256
from provia.training.role_transport_fit import _priors
from provia.model.intents import CompositionalIntentSpace
from provia.model.interval_transport import IntervalContinuationLanguage


def _feature_provenance(path: Path) -> Mapping[str, object]:
    manifest = path / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    value = json.loads(manifest.read_text())
    if not isinstance(value, Mapping):
        raise ValueError("query feature manifest is malformed")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-examples", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    source_artifact = torch.load(
        args.source_examples, map_location="cpu", weights_only=False
    )
    source = (
        source_artifact.get("metadata")
        if isinstance(source_artifact, Mapping)
        else None
    )
    if (
        not isinstance(source, Mapping)
        or source.get("schema") != "role_windowed_examples"
        or source.get("test_split_accessed") is not False
    ):
        raise ValueError("source role examples are malformed")

    manifest = Path(str(source["manifest"]))
    fold = int(source["fold"])
    episodes = materialize_cross_validation_fold(
        tuple(read_jsonl(manifest)), fold
    )
    train = tuple(value for value in episodes if value.split == "train")
    query = tuple(value for value in episodes if value.split == args.split)
    if not train or not query:
        raise ValueError("training or query partition is empty")
    intent = CompositionalIntentSpace.from_segments(
        segment for episode in train for segment in episode.segments
    )
    labels = {name: intent.labels[name] for name in EVENT_FACTORS}
    if any(
        tuple(source["factor_labels"][name]) != tuple(labels[name])  # type: ignore[index]
        for name in EVENT_FACTORS
    ):
        raise ValueError("query intent vocabulary differs from development")

    feature = _feature_provenance(args.feature_root)
    development_feature = source.get("cache")
    if (
        not isinstance(development_feature, Mapping)
        or feature.get("contains_annotations") is not False
        or feature.get("test_split_accessed") is not True
        or feature.get("partitions") != [args.split]
        or int(feature.get("fold", -1)) != fold
        or feature.get("backbone") != development_feature.get("backbone")
        or feature.get("backbone_checkpoint")
        != development_feature.get("backbone_checkpoint")
        or feature.get("observer_sha256")
        != development_feature.get("observer_sha256")
        or float(feature.get("sample_rate_hz", -1.0))
        != float(development_feature.get("sample_rate_hz", -2.0))
    ):
        raise ValueError("query features do not match the development stream")

    checkpoint_path = Path(str(source["observer_checkpoint"]))
    if _sha256(checkpoint_path) != source.get("observer_checkpoint_sha256"):
        raise ValueError("frozen event observer changed after model selection")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    observer_metadata = checkpoint.get("metadata")
    if (
        not isinstance(observer_metadata, Mapping)
        or observer_metadata.get("observer_architecture")
        not in ROLE_EVENT_OBSERVER_ARCHITECTURES
        or observer_metadata.get("test_split_accessed") is not False
        or int(observer_metadata.get("fold", -1)) != fold
    ):
        raise ValueError("frozen event observer is incompatible")
    observer = build_role_event_observer(observer_metadata, int(source["input_dim"]), labels)
    observer.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    observer.to(device).eval()

    language_path = Path(str(source["procedure_language"]))
    if _sha256(language_path) != source.get("procedure_language_sha256"):
        raise ValueError("frozen procedure language changed after model selection")
    language = IntervalContinuationLanguage.from_dict(
        json.loads(language_path.read_text())
    )
    if any(
        tuple(language.factor_labels[name]) != tuple(labels[name])
        for name in EVENT_FACTORS
    ):
        raise ValueError("procedure language and query vocabulary disagree")

    streams = load_competitive_role_streams(query, args.feature_root, intent)
    priors = _priors(observer_metadata)
    sequences = []
    with torch.inference_mode():
        for stream in streams:
            events = _causal_onset_events(
                observer,
                observer_metadata,
                stream,
                priors,
                labels,
                device,
                attach_targets=False,
            )
            gap_frames = int(source.get("gap_candidates_minimum_frames") or 0)
            if gap_frames > 0:
                # gap candidates (every hand always carries an open step), exactly as in the development examples
                events = _with_gap_candidates(stream, events, priors, labels, gap_frames)
            sequence = _sequence(stream, events)
            if sequence is not None:
                sequences.append(sequence)
    if not sequences:
        raise ValueError("frozen observer decoded no usable query candidates")

    metadata = {
        "schema": ROLE_QUERY_SCHEMA,
        "source_examples": str(args.source_examples.resolve()),
        "source_examples_sha256": _sha256(args.source_examples),
        "manifest": str(manifest.resolve()),
        "feature_root": str(args.feature_root.resolve()),
        "feature_manifest_sha256": _sha256(args.feature_root / "manifest.json"),
        "observer_checkpoint": str(checkpoint_path.resolve()),
        "observer_checkpoint_sha256": _sha256(checkpoint_path),
        "procedure_language": str(language_path.resolve()),
        "procedure_language_sha256": _sha256(language_path),
        "fold": fold,
        "split": args.split,
        "parameters_fitted": False,
        "query_annotations_used_by_candidate_decoder": False,
        "query_annotations_used_for_evaluation_only": True,
        "decoded_candidates": int(
            sum(len(value["origin_target"]) for value in sequences)
        ),
        "test_split_accessed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    torch.save({"metadata": metadata, args.split: sequences}, temporary)
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sequences": len(sequences),
                "decoded_candidates": metadata["decoded_candidates"],
                "gap_candidates_minimum_frames": int(source.get("gap_candidates_minimum_frames") or 0),
            }
        )
    )


if __name__ == "__main__":
    main()
