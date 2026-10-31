"""Compile decoded expert videos into a role-interval continuation language."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from provia.observer.factors import EVENT_FACTORS
from provia.model.intents import CompositionalIntentSpace
from provia.model.interval_transport import (
    CommittedReferenceAction,
    IntervalContinuationLanguage,
    compile_interval_continuations,
)
from provia.model.reference_event import UncertainReferenceEvent

from .reference_distribution import _normalized, project_reference_distribution


def compile_interval_reference_files(
    paths: Sequence[Path],
    intent_space: CompositionalIntentSpace,
) -> tuple[IntervalContinuationLanguage, dict[str, object]]:
    """Compile independently decoded, annotation-free successful videos."""

    if not paths:
        raise ValueError("at least one expert reference is required")
    traces = []
    source_videos = []
    confidence = {name: [] for name in EVENT_FACTORS}
    unknown_mass = {name: [] for name in EVENT_FACTORS}
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != "provia_uncertain_expert_reference"
        ):
            raise ValueError(f"{path}: unknown expert-reference schema")
        decoder = payload.get("decoder")
        if not isinstance(decoder, Mapping) or decoder.get("uses_expert_annotations"):
            raise ValueError(f"{path}: expert events must be decoded without labels")
        raw_events = payload.get("events")
        if not isinstance(raw_events, list) or not raw_events:
            raise ValueError(f"{path}: decoded expert video contains no events")
        trace = []
        for raw in raw_events:
            if not isinstance(raw, Mapping):
                raise ValueError(f"{path}: every decoded event must be an object")
            role = _normalized(raw.get("hand"))
            distributions = raw.get("factor_distributions")
            if role not in {"left", "right"} or not isinstance(
                distributions, Mapping
            ):
                raise ValueError(f"{path}: invalid event role or distributions")
            projected = {}
            for name in EVENT_FACTORS:
                values, statistics = project_reference_distribution(
                    distributions.get(name), name, path, intent_space
                )
                projected[name] = values
                confidence[name].append(float(statistics["top1_probability"]))
                unknown_mass[name].append(
                    float(statistics["unknown_probability_mass"])
                )
            start = float(raw.get("start_frame", 0))
            commitment = float(raw.get("onset_frame", start))
            end = float(raw.get("end_frame", commitment))
            trace.append(
                CommittedReferenceAction(
                    role=role,
                    start=start,
                    commitment=commitment,
                    end=end,
                    event=UncertainReferenceEvent(
                        verb=projected["verb"],
                        noun=projected["noun"],
                        tool=projected["tool"],
                    ),
                )
            )
        traces.append(tuple(trace))
        source_videos.append(str(payload.get("video_id") or path.stem))
    language = compile_interval_continuations(
        traces,
        factor_labels={
            name: intent_space.labels[name] for name in EVENT_FACTORS
        },
    )
    audit = {
        "reference_files": [str(path.resolve()) for path in paths],
        "reference_videos": source_videos,
        "reference_events": [len(trace) for trace in traces],
        "event_types": len(language.node_ids),
        "necessary_event_types": sum(
            language.node_is_globally_necessary(node)
            for node in range(len(language.node_ids))
        ),
        "optional_event_types": list(language.optional_nodes()),
        "states": language.state_count,
        "transitions": len(language.transitions),
        "accepting_states": len(language.accepting_states),
        "mean_decoder_top1_probability": {
            name: sum(values) / len(values) for name, values in confidence.items()
        },
        "mean_unknown_probability_mass": {
            name: sum(values) / len(values) for name, values in unknown_mass.items()
        },
        "uses_expert_annotations": False,
        "thresholds": 0,
    }
    return language, audit


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intent-space", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.audit.exists():
        raise FileExistsError("refusing to overwrite a compiled reference")
    with open(args.intent_space, encoding="utf-8") as handle:
        intent_space = CompositionalIntentSpace.from_dict(json.load(handle))
    language, audit = compile_interval_reference_files(
        args.reference, intent_space
    )
    _atomic_json(args.output, language.to_dict())
    _atomic_json(args.audit, audit)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
