"""Project unhanded interval annotations to causal single-actor completions.

Public procedural-error benchmarks commonly annotate complete action intervals
but neither hand identity nor a causal commitment/onset.  Such intervals can
overlap, so assigning them to the fixed left/right slots would invent hand
labels.  This module instead creates one active actor and one structural null
role.  Each annotated action becomes a completion event at its observed end;
actions completing on the same sampled frame are represented by one compound
set-valued event.

The projection is intended for the fully predicted completion-stream protocol.
Native segment-given evaluation must retain the original interval manifest.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

from .io import read_jsonl, write_jsonl
from .schema import CanonicalEpisode, Segment


_LABELED_CORRECTNESS = frozenset({"correct", "error", "correction"})


def _semantic_value(values: Iterable[str | None], *, fallback: str) -> str:
    materialized = sorted({str(value).strip() for value in values if value and str(value).strip()})
    return "+".join(materialized) if materialized else fallback


def _completion_frame(segment: Segment, sample_rate_hz: float) -> int:
    if segment.end_sec is None:
        raise ValueError("a completion event requires an observed end time")
    return int(round(float(segment.end_sec) * float(sample_rate_hz)))


def _compound_completion(
    episode: CanonicalEpisode,
    frame: int,
    segments: Sequence[Segment],
    sample_rate_hz: float,
    action_marks: bool = False,
) -> Segment:
    if not segments:
        raise ValueError("cannot create an empty compound completion")
    time_sec = float(frame) / float(sample_rate_hz)
    action_ids = sorted(
        {str(item.action_id) for item in segments if item.action_id is not None}
    )
    action_text = _semantic_value(
        (item.action_text for item in segments), fallback="unknown_action"
    )
    action_key = "+".join(action_ids) if action_ids else action_text
    # Preserve every factor that is completely annotated for this compound.
    # Missing factors remain explicit nulls rather than being hallucinated.
    # If no native factor is available, the native action class is encoded
    # exactly once in the required verb slot.  This deterministic rule lets a
    # verb-only dataset share visual evidence across actions without claiming
    # object or tool annotations that the source does not provide.
    native_factor_available = {
        "verb": all(item.verb for item in segments),
        "noun": all(item.noun for item in segments),
        "tool": all(item.tool for item in segments),
    }
    if not action_marks and any(native_factor_available.values()):
        verb = (
            _semantic_value((item.verb for item in segments), fallback="<unk>")
            if native_factor_available["verb"]
            else f"action_{action_key}"
        )
        noun = (
            _semantic_value((item.noun for item in segments), fallback="<unk>")
            if native_factor_available["noun"]
            else "<none>"
        )
        tool = (
            _semantic_value((item.tool for item in segments), fallback="<unk>")
            if native_factor_available["tool"]
            else "<none>"
        )
        semantic_encoding = (
            "complete_native_factorization"
            if native_factor_available["verb"] and native_factor_available["noun"]
            else "available_native_factorization"
        )
    else:
        verb = f"action_{action_key}"
        noun = "<none>"
        tool = "<none>"
        semantic_encoding = "atomic_native_action_class"
    correctness_values = {str(item.correctness) for item in segments}
    correctness = (
        "error"
        if "error" in correctness_values
        else "correction" if "correction" in correctness_values else "correct"
    )
    error_types = tuple(
        sorted({value for item in segments for value in item.error_types})
    )
    source_ids = tuple(sorted(item.segment_id for item in segments))
    return Segment(
        segment_id=f"completion_{frame:08d}",
        start_sec=time_sec,
        onset_sec=time_sec,
        end_sec=time_sec,
        observed=True,
        action_id="+".join(action_ids) if action_ids else None,
        action_text=action_text,
        verb=verb,
        noun=noun,
        tool=tool,
        hand="left",
        correctness=correctness,
        error_types=error_types,
        native_action_type="compound_completion",
        attributes={
            "projection": "observed_interval_end",
            "source_segment_ids": source_ids,
            "source_segment_count": len(source_ids),
            "source_dataset": episode.dataset,
            "semantic_encoding": semantic_encoding,
            "native_factors_preserved": tuple(
                name for name, available in native_factor_available.items() if available
            ),
        },
    )


def project_episode_to_single_actor_completions(
    episode: CanonicalEpisode,
    *,
    sample_rate_hz: float,
    action_marks: bool = False,
) -> CanonicalEpisode:
    """Return an audited completion stream without inventing hand identity."""

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    grouped: dict[int, list[Segment]] = {}
    omitted: list[str] = []
    for segment in episode.segments:
        if (
            not segment.observed
            or segment.start_sec is None
            or segment.end_sec is None
            or segment.correctness not in _LABELED_CORRECTNESS
        ):
            omitted.append(segment.segment_id)
            continue
        grouped.setdefault(
            _completion_frame(segment, sample_rate_hz), []
        ).append(segment)
    completions = tuple(
        _compound_completion(
            episode, frame, tuple(grouped[frame]), sample_rate_hz, action_marks
        )
        for frame in sorted(grouped)
    )
    if not completions:
        raise ValueError(f"{episode.episode_id}: no labeled completion is available")
    normalized_split = (
        "validation" if episode.split.lower() == "val" else episode.split.lower()
    )
    return replace(
        episode,
        split=normalized_split,
        segments=completions,
        attributes={
            **episode.attributes,
            "single_actor_projection": {
                "schema": "single_actor_completion_projection",
                "active_role": "left",
                "inactive_role": "right_structural_null",
                "completion_time": "observed_interval_end",
                "simultaneous_events": "compound_set_event",
                "sample_rate_hz": float(sample_rate_hz),
                "source_segments": len(episode.segments),
                "projected_completions": len(completions),
                "omitted_unlabeled_or_untimed_segments": omitted,
            },
        },
    )


def project_manifest(
    episodes: Iterable[CanonicalEpisode],
    *,
    sample_rate_hz: float,
    action_marks: bool = False,
) -> tuple[CanonicalEpisode, ...]:
    return tuple(
        project_episode_to_single_actor_completions(
            episode, sample_rate_hz=sample_rate_hz, action_marks=action_marks
        )
        for episode in episodes
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rate-hz", type=float, required=True)
    parser.add_argument(
        "--action-marks",
        action="store_true",
        help="use one atomic action class per step instead of verb, noun and tool factors",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    projected = project_manifest(
        tuple(read_jsonl(args.input)),
        sample_rate_hz=args.sample_rate_hz,
        action_marks=args.action_marks,
    )
    report = write_jsonl(projected, args.output, require_paths=True)
    audit_path = args.output.with_name(args.output.stem + ".projection.json")
    audit = {
        "schema": "single_actor_completion_manifest",
        "source_manifest": str(args.input.resolve()),
        "output_manifest": str(args.output.resolve()),
        "sample_rate_hz": float(args.sample_rate_hz),
        "episodes": report.episodes,
        "completion_events": report.segments,
        "errors": report.errors,
        "corrections": report.corrections,
        "query_annotations_used_at_inference": False,
        "training_annotations_used_for_event_supervision": True,
        "intended_protocol": "fully_predicted_completion_stream",
        "native_segment_protocol_uses_original_manifest": True,
    }
    with open(audit_path, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
