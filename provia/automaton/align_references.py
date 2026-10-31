"""Bayes-factor alignment of uncertain role-specific expert event streams.

The compiler aligns each hand independently across successful demonstrations.
Pairwise match weight is the log Bayes factor for one shared latent semantic
mark against independent marks under the empirical expert background.  Free
insertions represent incidental actions; no similarity threshold, gap cost,
or manually selected reference video is introduced.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from provia.observer.factors import EVENT_FACTORS, EVENT_ROLES


@dataclass(frozen=True)
class ReferenceEvent:
    source_index: int
    role: str
    probability: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class Alignment:
    pairs: tuple[tuple[int, int], ...]
    log_bayes_factor: float


def _normalized_probability(raw: object, factor: str) -> np.ndarray:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{factor} distribution must be an object")
    values = raw.get("log_probability")
    labels = raw.get("labels")
    if not isinstance(values, list) or not isinstance(labels, list) or not values:
        raise ValueError(f"{factor} distribution is incomplete")
    log_probability = np.asarray(values, dtype=np.float64)
    if log_probability.shape != (len(labels),) or not np.isfinite(
        log_probability
    ).all():
        raise ValueError(f"{factor} log probabilities are invalid")
    shifted = log_probability - log_probability.max()
    probability = np.exp(shifted)
    return probability / probability.sum()


def _events(payload: Mapping[str, object]) -> tuple[ReferenceEvent, ...]:
    raw_events = payload.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("an expert reference must contain decoded events")
    result = []
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping):
            raise ValueError("expert events must be objects")
        role = str(raw.get("hand"))
        distributions = raw.get("factor_distributions")
        if role not in EVENT_ROLES or not isinstance(distributions, Mapping):
            raise ValueError("expert event role or factors are invalid")
        result.append(
            ReferenceEvent(
                source_index=index,
                role=role,
                probability={
                    name: _normalized_probability(distributions.get(name), name)
                    for name in EVENT_FACTORS
                },
            )
        )
    return tuple(result)


def empirical_background(
    traces: Sequence[Sequence[ReferenceEvent]],
) -> dict[str, np.ndarray]:
    events = [event for trace in traces for event in trace]
    if not events:
        raise ValueError("alignment requires expert events")
    background = {}
    for name in EVENT_FACTORS:
        matrix = np.stack([event.probability[name] for event in events])
        if any(row.shape != matrix[0].shape for row in matrix):
            raise ValueError(f"expert {name} vocabularies disagree")
        value = matrix.mean(axis=0)
        background[name] = value / value.sum()
    return background


def semantic_log_bayes_factor(
    first: ReferenceEvent,
    second: ReferenceEvent,
    background: Mapping[str, np.ndarray],
) -> float:
    if first.role != second.role:
        return -math.inf
    result = 0.0
    tiny = np.finfo(np.float64).tiny
    for name in EVENT_FACTORS:
        prior = np.asarray(background[name], dtype=np.float64)
        left = np.asarray(first.probability[name], dtype=np.float64)
        right = np.asarray(second.probability[name], dtype=np.float64)
        if prior.shape != left.shape or right.shape != left.shape:
            raise ValueError(f"{name} alignment distributions disagree")
        shared_to_independent = float(np.sum(left * right / np.maximum(prior, tiny)))
        result += math.log(max(shared_to_independent, tiny))
    return result


def align_sequences(
    anchor: Sequence[ReferenceEvent],
    candidate: Sequence[ReferenceEvent],
    background: Mapping[str, np.ndarray],
) -> Alignment:
    """Maximum-evidence ordered matching with Bayes-neutral free insertions."""

    rows, columns = len(anchor), len(candidate)
    value = np.zeros((rows + 1, columns + 1), dtype=np.float64)
    count = np.zeros((rows + 1, columns + 1), dtype=np.int64)
    back = np.zeros((rows + 1, columns + 1), dtype=np.int8)
    # back: 1 skips anchor, 2 skips candidate, 3 matches.
    for row in range(1, rows + 1):
        back[row, 0] = 1
    for column in range(1, columns + 1):
        back[0, column] = 2
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            alternatives = [
                (value[row - 1, column], count[row - 1, column], 1),
                (value[row, column - 1], count[row, column - 1], 2),
            ]
            gain = semantic_log_bayes_factor(
                anchor[row - 1], candidate[column - 1], background
            )
            if gain > 0.0:
                alternatives.append(
                    (
                        value[row - 1, column - 1] + gain,
                        count[row - 1, column - 1] + 1,
                        3,
                    )
                )
            best = max(alternatives, key=lambda item: (item[0], item[1], item[2]))
            value[row, column], count[row, column], back[row, column] = best
    pairs = []
    row, column = rows, columns
    while row or column:
        move = int(back[row, column])
        if move == 3:
            pairs.append((row - 1, column - 1))
            row -= 1
            column -= 1
        elif move == 1:
            row -= 1
        elif move == 2:
            column -= 1
        else:  # pragma: no cover - only the origin has no predecessor
            raise RuntimeError("expert alignment backtrace is incomplete")
    pairs.reverse()
    return Alignment(tuple(pairs), float(value[rows, columns]))


def align_reference_payloads(
    payloads: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if len(payloads) < 2:
        raise ValueError("cross-demonstration alignment requires at least two videos")
    decoded = [_events(payload) for payload in payloads]
    background = empirical_background(decoded)
    output = [json.loads(json.dumps(payload)) for payload in payloads]
    role_audit = {}
    for role in EVENT_ROLES:
        role_traces = [tuple(event for event in trace if event.role == role) for trace in decoded]
        if any(not trace for trace in role_traces):
            raise ValueError(f"every expert video needs at least one {role} event")
        pairwise = np.zeros((len(role_traces), len(role_traces)), dtype=np.float64)
        for first in range(len(role_traces)):
            for second in range(first + 1, len(role_traces)):
                score = align_sequences(
                    role_traces[first], role_traces[second], background
                ).log_bayes_factor
                pairwise[first, second] = score
                pairwise[second, first] = score
        totals = pairwise.sum(axis=1)
        medoid = max(range(len(role_traces)), key=lambda index: (totals[index], -index))
        clusters: list[list[ReferenceEvent]] = [
            [event] for event in role_traces[medoid]
        ]
        matched_source: list[set[int]] = [set() for _ in role_traces]
        matched_source[medoid].update(event.source_index for event in role_traces[medoid])
        alignments = {}
        for trace_index, trace in enumerate(role_traces):
            if trace_index == medoid:
                pairs = tuple((index, index) for index in range(len(trace)))
                score = float(pairwise[medoid, trace_index])
            else:
                aligned = align_sequences(role_traces[medoid], trace, background)
                pairs, score = aligned.pairs, aligned.log_bayes_factor
            alignments[str(trace_index)] = {
                "matches": len(pairs),
                "log_bayes_factor": score,
            }
            for anchor_index, trace_event_index in pairs:
                event = trace[trace_event_index]
                if trace_index != medoid:
                    clusters[anchor_index].append(event)
                matched_source[trace_index].add(event.source_index)
        pooled = []
        for cluster in clusters:
            pooled.append(
                {
                    name: np.stack(
                        [event.probability[name] for event in cluster]
                    ).mean(axis=0)
                    for name in EVENT_FACTORS
                }
            )
        for trace_index, trace in enumerate(role_traces):
            aligned = (
                tuple((index, index) for index in range(len(trace)))
                if trace_index == medoid
                else align_sequences(role_traces[medoid], trace, background).pairs
            )
            for anchor_index, trace_event_index in aligned:
                source_index = trace[trace_event_index].source_index
                raw_event = output[trace_index]["events"][source_index]  # type: ignore[index]
                distributions = raw_event["factor_distributions"]  # type: ignore[index]
                argmax = raw_event["argmax"]  # type: ignore[index]
                for name in EVENT_FACTORS:
                    probability = pooled[anchor_index][name]
                    probability = probability / probability.sum()
                    distributions[name]["log_probability"] = [  # type: ignore[index]
                        float(math.log(max(value, np.finfo(np.float64).tiny)))
                        for value in probability
                    ]
                    labels = distributions[name]["labels"]  # type: ignore[index]
                    argmax[name] = str(labels[int(probability.argmax())])  # type: ignore[index]
        role_audit[role] = {
            "medoid_reference_index": medoid,
            "medoid_video_id": str(payloads[medoid].get("video_id")),
            "medoid_events": len(role_traces[medoid]),
            "alignment": alignments,
            "unmatched_events": [
                len(role_traces[index]) - len(matched_source[index])
                for index in range(len(role_traces))
            ],
        }
    for payload in output:
        decoder = payload.get("decoder")
        if not isinstance(decoder, dict):
            raise ValueError("expert decoder provenance is missing")
        decoder["cross_demonstration_alignment"] = (
            "ordered shared-latent Bayes factors under empirical expert background"
        )
        decoder["alignment_thresholds"] = 0
    audit = {
        "schema": "role_bayes_factor_expert_alignment",
        "videos": [str(payload.get("video_id")) for payload in payloads],
        "roles": role_audit,
        "background_probability": {
            name: [float(value) for value in background[name]]
            for name in EVENT_FACTORS
        },
        "gap_cost": 0,
        "similarity_thresholds": 0,
        "uses_expert_annotations": False,
    }
    return output, audit


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
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite aligned expert references")
    if args.audit.exists():
        raise FileExistsError(f"refusing to overwrite {args.audit}")
    payloads = []
    for path in args.reference:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != "provia_uncertain_expert_reference"
            or not isinstance(payload.get("decoder"), Mapping)
            or payload["decoder"].get("uses_expert_annotations") is not False  # type: ignore[index]
        ):
            raise ValueError(f"{path}: reference provenance is invalid")
        payloads.append(payload)
    aligned, audit = align_reference_payloads(payloads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source, payload in zip(args.reference, aligned):
        _atomic_json(args.output_dir / source.name, payload)
    _atomic_json(args.audit, audit)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
