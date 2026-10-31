"""Compile bimanual training examples at predicted onsets.

The frozen observer emits immutable candidates and soft verb--noun--tool marks
at predicted interaction onset, then causally closes each step at its
posterior-median end.  The current event's onset-to-end visual effect becomes
available only at that end.  Predicted intervals also define the support used
for annotation matching; query annotations never define a candidate or model
input.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from provia.automaton.event_decoding import (
    decode_posterior_median_events,
    decode_posterior_median_onsets,
)
from provia.data.io import read_jsonl
from provia.data.role_stream import CompetitiveRoleStream, load_competitive_role_streams
from provia.observer.factors import EVENT_FACTORS, EVENT_ROLES
from provia.observer.build import ROLE_EVENT_OBSERVER_ARCHITECTURES, build_role_event_observer
from provia.observer.marked_renewal import MarkedRenewalRoleObserver
from provia.observer.role_statistics import role_interval_trajectory_statistics
from provia.training.common import (
    _cache_provenance,
    _partitions,
    _sha256,
)
from provia.training.role_transport_fit import (
    TransportEvent,
    _decoded_events,
    _fit_role_execution_mode_dynamics,
    _fit_role_observation_channel,
    _priors,
)
from provia.model.intents import CompositionalIntentSpace
from provia.model.interval_transport import IntervalContinuationLanguage


def _ordered_overlap_pairs(
    predicted: Sequence[tuple[int, int]],
    target: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Maximum monotone positive-overlap matching with stable tie breaks."""

    indexed_predicted = tuple(
        sorted(enumerate(predicted), key=lambda item: (item[1], item[0]))
    )
    indexed_target = tuple(
        sorted(enumerate(target), key=lambda item: (item[1], item[0]))
    )
    predicted_values = tuple(value for _, value in indexed_predicted)
    target_values = tuple(value for _, value in indexed_target)
    cardinality = np.zeros(
        (len(predicted_values) + 1, len(target_values) + 1), dtype=np.int64
    )
    intersection = np.zeros_like(cardinality)
    decision = np.zeros_like(cardinality, dtype=np.int8)
    for predicted_index, (predicted_start, predicted_end) in enumerate(
        predicted_values, start=1
    ):
        for target_index, (target_start, target_end) in enumerate(
            target_values, start=1
        ):
            candidates = [
                (
                    int(cardinality[predicted_index - 1, target_index]),
                    int(intersection[predicted_index - 1, target_index]),
                    0,
                ),
                (
                    int(cardinality[predicted_index, target_index - 1]),
                    int(intersection[predicted_index, target_index - 1]),
                    1,
                ),
            ]
            overlap = max(
                0,
                min(predicted_end, target_end)
                - max(predicted_start, target_start)
                + 1,
            )
            if overlap:
                candidates.append(
                    (
                        int(cardinality[predicted_index - 1, target_index - 1])
                        + 1,
                        int(intersection[predicted_index - 1, target_index - 1])
                        + overlap,
                        2,
                    )
                )
            best = max(candidates, key=lambda value: (value[0], value[1], value[2]))
            cardinality[predicted_index, target_index] = best[0]
            intersection[predicted_index, target_index] = best[1]
            decision[predicted_index, target_index] = best[2]
    pairs = []
    predicted_index = len(predicted_values)
    target_index = len(target_values)
    while predicted_index and target_index:
        choice = int(decision[predicted_index, target_index])
        if choice == 2:
            pairs.append(
                (
                    indexed_predicted[predicted_index - 1][0],
                    indexed_target[target_index - 1][0],
                )
            )
            predicted_index -= 1
            target_index -= 1
        elif choice == 1:
            target_index -= 1
        else:
            predicted_index -= 1
    return tuple(sorted(pairs))


def _event_targets(
    stream: CompetitiveRoleStream,
    events: Sequence[TransportEvent],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = np.zeros(len(events), dtype=np.bool_)
    mode = np.full(len(events), -1, dtype=np.int64)
    assignment = np.full(len(events), -1, dtype=np.int64)
    for role, role_name in enumerate(EVENT_ROLES):
        event_indices = [
            index for index, event in enumerate(events) if event.key[0] == role_name
        ]
        interval_indices = [
            index
            for index, interval in enumerate(stream.intervals)
            if int(interval.role) == role
        ]
        pairs = _ordered_overlap_pairs(
            tuple(
                (int(events[index].start), int(events[index].end))
                for index in event_indices
            ),
            tuple(
                (
                    int(stream.intervals[index].start),
                    int(stream.intervals[index].end),
                )
                for index in interval_indices
            ),
        )
        for event_position, interval_position in pairs:
            event_index = event_indices[event_position]
            interval_index = interval_indices[interval_position]
            origin[event_index] = True
            mode[event_index] = int(stream.intervals[interval_index].correctness)
            assignment[event_index] = interval_index
    return origin, mode, assignment


def _sequence(
    stream: CompetitiveRoleStream,
    events: Sequence[TransportEvent],
) -> dict[str, object] | None:
    ordered = tuple(
        sorted(
            (event for event in events if not event.right_censored),
            key=lambda event: (
                int(event.onset),
                int(event.start),
                EVENT_ROLES.index(event.key[0]),
            ),
        )
    )
    if not ordered:
        return None
    origin, mode, assignment = _event_targets(stream, ordered)
    onset = torch.tensor([int(event.onset) for event in ordered], dtype=torch.long)
    end = torch.tensor([int(event.end) for event in ordered], dtype=torch.long)
    role_names = [event.key[0] for event in ordered]
    previous, statistic, valid = role_interval_trajectory_statistics(
        stream.role_features,
        stream.timestamps_sec,
        role_names,
        onset,
        end,
    )
    keep = torch.nonzero(valid, as_tuple=False).flatten()
    if not len(keep):
        return None
    keep_numpy = keep.numpy()
    kept_events = tuple(ordered[index] for index in keep_numpy)
    kept_assignment = assignment[keep_numpy]
    native_error_types = tuple(
        ()
        if interval_index < 0
        else tuple(stream.intervals[int(interval_index)].error_types)
        for interval_index in kept_assignment
    )
    return {
        "episode_id": stream.episode_id,
        "video_id": stream.video_id,
        "stream_start_sec": float(stream.timestamps_sec[0]),
        "stream_end_sec": float(stream.timestamps_sec[-1]),
        "start_index": torch.tensor(
            [int(event.start) for event in kept_events], dtype=torch.long
        ),
        "onset_index": onset.index_select(0, keep),
        "end_index": end.index_select(0, keep),
        "onset_time_sec": stream.timestamps_sec.index_select(
            0, onset.index_select(0, keep)
        ).float(),
        "end_time_sec": stream.timestamps_sec.index_select(
            0,
            torch.tensor(
                [int(event.end) for event in kept_events], dtype=torch.long
            ),
        ).float(),
        "role_index": torch.tensor(
            [EVENT_ROLES.index(event.key[0]) for event in kept_events],
            dtype=torch.long,
        ),
        "previous": previous.index_select(0, keep).float(),
        "statistic": statistic.index_select(0, keep).float(),
        **{
            f"factor_{name}": torch.tensor(
                [event.factor_probability[name] for event in kept_events],
                dtype=torch.float32,
            )
            for name in EVENT_FACTORS
        },
        "origin_target": torch.from_numpy(origin[keep_numpy].copy()),
        "mode_target": torch.from_numpy(mode[keep_numpy].copy()),
        "native_assignment": torch.from_numpy(kept_assignment.copy()),
        "native_error_types": native_error_types,
        "native_intervals": tuple(
            {
                "start_index": int(interval.start),
                "onset_index": int(interval.onset),
                "end_index": int(interval.end),
                "start_time_sec": float(stream.timestamps_sec[interval.start]),
                "onset_time_sec": float(stream.timestamps_sec[interval.onset]),
                "end_time_sec": float(stream.timestamps_sec[interval.end]),
                "role_index": int(interval.role),
                "mode": int(interval.correctness),
                "error_types": tuple(interval.error_types),
            }
            for interval in stream.intervals
        ),
    }


def _normalization(
    sequences: Sequence[Mapping[str, object]],
) -> tuple[Tensor, Tensor]:
    values = torch.cat(
        [sequence["statistic"] for sequence in sequences]  # type: ignore[list-item]
    ).float()
    mean = values.mean(dim=0)
    variance = (values - mean).square().mean(dim=0)
    positive = variance[variance > 0]
    reference = positive.mean() if len(positive) else variance.new_tensor(1.0)
    floor = torch.finfo(variance.dtype).eps * reference
    return mean, variance.clamp_min(floor).sqrt()


def _origin_audit(
    sequences: Sequence[Mapping[str, object]],
) -> tuple[float, dict[str, object]]:
    values = torch.cat(
        [sequence["origin_target"] for sequence in sequences]  # type: ignore[list-item]
    ).bool()
    genuine = int(values.sum())
    total = int(len(values))
    probability = (genuine + 0.5) / (total + 1.0)
    return probability, {
        "decoded_candidates": total,
        "genuine_matches": genuine,
        "insertions": total - genuine,
        "genuine_probability": probability,
        "timing_tolerance": None,
        "anomaly_labels_used_for_origin": False,
    }


def _causal_onset_events(
    observer: MarkedRenewalRoleObserver,
    metadata: Mapping[str, object],
    stream: CompetitiveRoleStream,
    priors: Mapping[str, Tensor],
    labels: Mapping[str, Sequence[str]],
    device: torch.device,
    *,
    attach_targets: bool = True,
) -> tuple[TransportEvent, ...]:
    """Decode immutable candidates at onset and their later causal end marks."""

    role_change = stream.role_change_features
    if metadata.get("role_change_evidence") is False:
        role_change = None
    elif metadata.get("role_change_evidence") is True and role_change is None:
        raise ValueError("observer checkpoint requires role-change evidence")
    evidence, posterior = observer(
        stream.global_features.to(device),
        stream.role_features.to(device),
        stream.timestamps_sec.to(device=device, dtype=torch.float32),
        None if role_change is None else role_change.to(device),
    )
    factors = {
        name: values.float().cpu()
        for name, values in evidence.factor_logits.items()
    }
    conditional = posterior.conditional_mark_probability.float().cpu()
    statistic = str(metadata.get("semantic_mark_statistic") or "frame-evidence-mean")
    if statistic == "path-effect":
        statistic = "frame-evidence-mean"
    onsets = decode_posterior_median_onsets(
        conditional,
        factors,
        priors,
        observer.initial_phase_probability.cpu(),
        semantic_statistic=statistic,
    )
    completed = decode_posterior_median_events(
        conditional,
        factors,
        priors,
        observer.initial_phase_probability.cpu(),
        semantic_statistic=statistic,
    )
    completed_transport = _decoded_events(
        stream, completed, labels, attach_targets=attach_targets
    )
    completed_by_onset = {
        (EVENT_ROLES.index(event.key[0]), event.start, event.onset): event
        for event in completed_transport
    }
    onset_keys = {
        (event.role, event.start_index, event.onset_index) for event in onsets
    }
    if onset_keys != set(completed_by_onset):
        raise RuntimeError("causal onset and eventual event decoders disagree")

    result = []
    for event in onsets:
        key = (event.role, event.start_index, event.onset_index)
        complete = completed_by_onset[key]
        log_probability = event.factor_log_probability
        factor_probability = {
            name: tuple(float(np.exp(value)) for value in log_probability[name])
            for name in EVENT_FACTORS
        }
        factors_argmax = tuple(
            str(labels[name][int(np.argmax(factor_probability[name]))])
            for name in EVENT_FACTORS
        )
        result.append(
            TransportEvent(
                key=(EVENT_ROLES[event.role], *factors_argmax),
                factor_probability=factor_probability,
                start=event.start_index,
                onset=event.onset_index,
                end=complete.end,
                correctness=complete.correctness,
                error_types=complete.error_types,
                commitment_probability=event.commitment_probability,
                right_censored=complete.right_censored,
            )
        )
    return tuple(
        sorted(result, key=lambda item: (item.onset, item.start, item.key[0]))
    )


def _with_gap_candidates(
    stream: CompetitiveRoleStream,
    events: Sequence[TransportEvent],
    priors: Mapping[str, Tensor],
    labels: Mapping[str, Sequence[str]],
    minimum_frames: int,
) -> tuple[TransportEvent, ...]:
    """Every hand always carries an open step: add one candidate per role for
    each interval the observer left uncovered (leading gap and gaps between
    consecutive decoded candidates) of at least ``minimum_frames`` frames.
    Gap candidates carry the observer's prior mark distribution, onset equal
    to start, and zero commitment; their execution status is read from the
    frame-level evidence like any other candidate."""

    result = list(events)
    for role, role_name in enumerate(EVENT_ROLES):
        decoded = sorted(
            (event for event in events if event.key[0] == role_name),
            key=lambda event: (int(event.start), int(event.end)),
        )
        cursor = 0
        for event in decoded:
            gap_end = int(event.start) - 1
            if gap_end - cursor + 1 >= minimum_frames:
                result.append(
                    TransportEvent(
                        key=(
                            role_name,
                            *(
                                str(labels[name][int(torch.argmax(priors[name]))])
                                for name in EVENT_FACTORS
                            ),
                        ),
                        factor_probability={
                            name: tuple(float(value) for value in priors[name])
                            for name in EVENT_FACTORS
                        },
                        start=cursor,
                        onset=cursor,
                        end=gap_end,
                        correctness=-1,
                        error_types=(),
                        commitment_probability=0.0,
                        right_censored=False,
                    )
                )
            cursor = max(cursor, int(event.end) + 1)
    return tuple(sorted(result, key=lambda item: (item.onset, item.start, item.key[0])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--observer-checkpoint", type=Path, required=True)
    parser.add_argument("--procedure-language", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gap-candidates", type=int, default=0, help="minimum frames: add one open-step candidate per role for every uncovered interval of at least this many frames (0 = off)")
    parser.add_argument("--gap-base-examples", type=Path, default=None, help="compiled examples file whose registries (prefix evidence, effect, language) the gap examples inherit; its sha256 is recorded")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    episodes = tuple(read_jsonl(args.manifest))
    train_native, validation_native = _partitions(episodes, args.fold)
    intent_space = CompositionalIntentSpace.from_segments(
        segment for episode in train_native for segment in episode.segments
    )
    labels = {name: intent_space.labels[name] for name in EVENT_FACTORS}
    payload = torch.load(
        args.observer_checkpoint, map_location="cpu", weights_only=False
    )
    metadata = payload.get("metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("test_split_accessed") is not False
        or int(metadata.get("fold", -1)) != args.fold
        or metadata.get("observer_architecture")
        not in ROLE_EVENT_OBSERVER_ARCHITECTURES
    ):
        raise ValueError("observer checkpoint does not belong to this fold")
    train_streams = load_competitive_role_streams(
        train_native, args.feature_root, intent_space
    )
    validation_streams = load_competitive_role_streams(
        validation_native, args.feature_root, intent_space
    )
    input_dim = int(train_streams[0].role_features.shape[-1])
    observer = build_role_event_observer(metadata, input_dim, labels)
    observer.load_state_dict(payload["model"], strict=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    observer.to(device).eval()
    priors = _priors(metadata)

    with open(args.procedure_language, encoding="utf-8") as handle:
        language = IntervalContinuationLanguage.from_dict(json.load(handle))
    if any(
        tuple(language.factor_labels[name]) != tuple(labels[name])
        for name in EVENT_FACTORS
    ):
        raise ValueError("procedure language and fold-training labels disagree")

    decoded_partitions: dict[
        str, list[tuple[CompetitiveRoleStream, tuple[TransportEvent, ...]]]
    ] = {"train": [], "validation": []}
    with torch.inference_mode():
        for split, streams in (
            ("train", train_streams),
            ("validation", validation_streams),
        ):
            for stream in streams:
                decoded = _causal_onset_events(
                    observer, metadata, stream, priors, labels, device
                )
                decoded_partitions[split].append((stream, decoded))

    observation_channel, observation_audit = _fit_role_observation_channel(
        decoded_partitions["train"], labels
    )
    transition, initial, mode_audit = _fit_role_execution_mode_dynamics(
        train_streams
    )
    if args.gap_candidates > 0:
        decoded_partitions = {
            split: [
                (stream, _with_gap_candidates(stream, events, priors, labels, args.gap_candidates))
                for stream, events in pairs
            ]
            for split, pairs in decoded_partitions.items()
        }
    train = [
        value
        for stream, events in decoded_partitions["train"]
        if (value := _sequence(stream, events)) is not None
    ]
    validation = [
        value
        for stream, events in decoded_partitions["validation"]
        if (value := _sequence(stream, events)) is not None
    ]
    if not train or not validation:
        raise ValueError("observer decoded no usable onset sequences")
    mean, scale = _normalization(train)
    genuine_probability, origin_audit = _origin_audit(train)
    metadata_out = {
        "schema": "role_examples",
        "gap_candidates_minimum_frames": int(args.gap_candidates),
        "gap_base_examples_sha256": (None if args.gap_base_examples is None else _sha256(args.gap_base_examples)),
        "fold": args.fold,
        "manifest": str(args.manifest.resolve()),
        "feature_root": str(args.feature_root.resolve()),
        "cache": _cache_provenance(args.feature_root, args.fold),
        "observer_checkpoint": str(args.observer_checkpoint.resolve()),
        "observer_checkpoint_sha256": _sha256(args.observer_checkpoint),
        "procedure_language": str(args.procedure_language.resolve()),
        "procedure_language_sha256": _sha256(args.procedure_language),
        "input_dim": input_dim,
        "factor_labels": {name: list(labels[name]) for name in EVENT_FACTORS},
        "trajectory_normalization": {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "source": (
                "fold-training predicted current-event paths from interaction "
                "onset to causally decoded non-censored end"
            ),
        },
        "effect_statistic_interval": "current predicted onset-to-end interval",
        "origin_channel": origin_audit,
        "observation_channel": observation_audit,
        "missed_event_probability": [
            float(value) for value in observation_channel.missed_event_probability
        ],
        "mode_transition_probability": transition.tolist(),
        "mode_initial_probability": initial.tolist(),
        "mode_dynamics": mode_audit,
        "causal_checkpoint": (
            "candidate identity and semantic mark are fixed at frozen-observer "
            "interaction onset; current-event effect is released only at its "
            "causally decoded non-censored end"
        ),
        "independent_commitment_annotations_available": False,
        "train_sequences": len(train),
        "validation_sequences": len(validation),
        "train_decoded_candidates": int(
            sum(len(value["origin_target"]) for value in train)
        ),
        "validation_decoded_candidates": int(
            sum(len(value["origin_target"]) for value in validation)
        ),
        "test_split_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    torch.save(
        {"metadata": metadata_out, "train": train, "validation": validation},
        temporary,
    )
    os.replace(temporary, args.output)
    print(
        {
            "output": str(args.output),
            "train_sequences": len(train),
            "validation_sequences": len(validation),
            "train_candidates": metadata_out["train_decoded_candidates"],
            "validation_candidates": metadata_out[
                "validation_decoded_candidates"
            ],
            "genuine_probability": genuine_probability,
            "missed_event_probability": metadata_out[
                "missed_event_probability"
            ],
        }
    )


if __name__ == "__main__":
    main()
