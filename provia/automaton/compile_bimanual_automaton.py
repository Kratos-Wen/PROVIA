"""Induce a predictive-state automaton from a bimanual procedure language."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch

from provia.observer.factors import EVENT_ROLES
from provia.training.common import _sha256
from provia.model.predictive_state_induction import (
    JEFFREYS_CONCENTRATION,
    quotient_interval_language,
)
from provia.model.interval_transport import IntervalContinuationLanguage

from .compile_languages import _atomic_json


def _mapping(path: Path) -> Mapping[str, object]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain one mapping")
    return value


def _decoded_trace(
    path: Path, language: IntervalContinuationLanguage
) -> tuple[int, ...]:
    payload = _mapping(path)
    decoder = payload.get("decoder")
    events = payload.get("events")
    if (
        payload.get("schema") != "provia_uncertain_expert_reference"
        or not isinstance(decoder, Mapping)
        or decoder.get("uses_expert_annotations") is not False
        or not isinstance(events, list)
        or not events
    ):
        raise ValueError(f"{path}: expert reference is malformed or annotated")
    indexed = []
    for index, raw in enumerate(events):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path}: event {index} is malformed")
        role = str(raw.get("hand"))
        argmax = raw.get("argmax")
        if role not in EVENT_ROLES or not isinstance(argmax, Mapping):
            raise ValueError(f"{path}: event {index} lacks a role or mark")
        start = float(raw.get("start_frame", 0.0))
        commitment = float(raw.get("onset_frame", start))
        end = float(raw.get("end_frame", commitment))
        if not 0.0 <= start <= commitment <= end:
            raise ValueError(f"{path}: event {index} has invalid timing")
        identity = f"{role}|{argmax.get('verb')}|{argmax.get('noun')}"
        indexed.append(
            (
                commitment,
                start,
                end,
                EVENT_ROLES.index(role),
                index,
                identity,
            )
        )
    node_index = {value: index for index, value in enumerate(language.node_ids)}
    result = []
    for *_, identity in sorted(indexed):
        node = node_index.get(identity)
        if node is None:
            raise ValueError(f"{path}: decoded event {identity} is outside the language")
        result.append(node)
    return tuple(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--reference-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    artifact = torch.load(args.examples, map_location="cpu", weights_only=False)
    source = artifact.get("metadata") if isinstance(artifact, Mapping) else None
    if (
        not isinstance(source, Mapping)
        or source.get("schema") != "role_windowed_examples"
        or source.get("test_split_accessed") is not False
    ):
        raise ValueError("role examples are malformed")
    source_path = Path(str(source["procedure_language"]))
    if (
        not source_path.is_file()
        or _sha256(source_path) != source.get("procedure_language_sha256")
    ):
        raise ValueError("source role language changed after example compilation")
    source_language = IntervalContinuationLanguage.from_dict(_mapping(source_path))

    audit = _mapping(args.reference_audit)
    raw_paths = audit.get("reference_files")
    expected_counts = audit.get("reference_events")
    if (
        audit.get("uses_expert_annotations") is not False
        or audit.get("thresholds") != 0
        or not isinstance(raw_paths, list)
        or not isinstance(expected_counts, list)
        or len(raw_paths) != source_language.source_trace_count
        or len(raw_paths) != len(expected_counts)
    ):
        raise ValueError("reference audit is malformed or not annotation-free")
    paths = tuple(Path(str(value)) for value in raw_paths)
    traces = tuple(_decoded_trace(path, source_language) for path in paths)
    if tuple(len(trace) for trace in traces) != tuple(
        int(value) for value in expected_counts
    ):
        raise ValueError("decoded expert event counts changed after compilation")

    language, quotient = quotient_interval_language(source_language, traces)
    language_path = args.output_dir / "continuation_language.json"
    _atomic_json(language_path, language.to_dict())
    registry = {
        "schema": "bayesian_predictive_role_language_registry",
        "method": "factorized Jeffreys-evidence predictive-state quotient",
        "source_examples": str(args.examples.resolve()),
        "source_examples_sha256": _sha256(args.examples),
        "source_language": str(source_path.resolve()),
        "source_language_sha256": _sha256(source_path),
        "reference_audit": str(args.reference_audit.resolve()),
        "reference_audit_sha256": _sha256(args.reference_audit),
        "references": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in paths
        ],
        "fold": int(source["fold"]),
        "language": language_path.name,
        "language_sha256": _sha256(language_path),
        "source_process_states": source_language.state_count,
        "predictive_states": quotient.state_count,
        "merged_source_states": quotient.merged_state_count,
        "log_bayes_evidence_gain": quotient.log_bayes_evidence_gain,
        "source_traces": quotient.source_trace_count,
        "model_prior_odds": 1.0,
        "categorical_prior": {
            "name": "Jeffreys",
            "concentration_per_outcome": JEFFREYS_CONCENTRATION,
        },
        "unvisited_valid_source_states_preserved": True,
        "repetition_or_revisit_cycles_preserved": True,
        "similarity_threshold": False,
        "selected_state_count": False,
        "anomaly_labels_used": False,
        "expert_event_annotations_used": False,
        "validation_videos_used": False,
        "test_videos_used": False,
    }
    _atomic_json(args.output_dir / "registry.json", registry)
    _atomic_json(
        args.output_dir / "complete.json",
        {"status": "complete", "test_split_accessed": False},
    )
    print(json.dumps(registry, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
