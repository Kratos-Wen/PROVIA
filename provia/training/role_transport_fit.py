"""Fit the bimanual observation channel on the step observer.

Two protocols are emitted from one frozen observer. ``annotation_support`` uses
annotated query intervals but predicts every event semantic factor. It isolates
the transport from event localization. ``predicted_support`` decodes both
intervals and semantics causally and is the complete query protocol. Neither
protocol observes validation anomaly labels during fitting or model selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from provia.automaton.decoded_events import DecodedEvent
from provia.automaton.event_decoding import decode_posterior_median_events
from provia.data.role_stream import CompetitiveRoleStream
from provia.data.streaming_events import CORRECTNESS_INDEX
from provia.observer.factors import EVENT_FACTORS, EVENT_ROLES
from provia.observer.marked_renewal import MarkedRenewalRoleObserver
from provia.model.role_state_transport import EventKey
from provia.model.factor_transport import FACTOR_ORDER
from provia.model.execution_mode import ERROR, EXECUTION_MODES
from provia.model.observation_channel import (
    FactorizedRoleObservationChannel,
)


@dataclass(frozen=True)
class TransportEvent:
    key: EventKey
    factor_probability: Mapping[str, tuple[float, ...]]
    start: int
    onset: int
    end: int
    correctness: int
    error_types: tuple[str, ...]
    commitment_probability: float = 1.0
    right_censored: bool = False


def _priors(metadata: Mapping[str, object]) -> dict[str, torch.Tensor]:
    raw = metadata.get("event_balanced_factor_counts")
    if not isinstance(raw, Mapping):
        raise ValueError("observer checkpoint has no event-balanced factor counts")
    result = {}
    for name in EVENT_FACTORS:
        counts = torch.tensor(raw[name], dtype=torch.float32)  # type: ignore[arg-type]
        posterior = counts + 0.5
        result[name] = posterior / posterior.sum()
    return result


def _target_for_decoded(
    event: DecodedEvent, stream: CompetitiveRoleStream
) -> tuple[int, tuple[str, ...]]:
    """Assign training supervision by positive temporal intersection.

    The maximum-intersection interval of the same role is used when one exists;
    no IoU or duration threshold is introduced. A hallucinated event with no
    annotated support remains a normal/background observation.
    """

    candidates = []
    for interval in stream.intervals:
        if interval.role != event.role:
            continue
        overlap = max(
            0,
            min(event.end_index, interval.end)
            - max(event.start_index, interval.start)
            + 1,
        )
        if overlap:
            candidates.append((overlap, -abs(event.onset_index - interval.onset), interval))
    if not candidates:
        return CORRECTNESS_INDEX["correct"], ()
    interval = max(candidates, key=lambda value: (value[0], value[1]))[2]
    return int(interval.correctness), tuple(interval.error_types)


def _ordered_positive_overlap_matching(
    predicted: Sequence[tuple[int, int]],
    target: Sequence[tuple[int, int]],
) -> tuple[int, ...]:
    """Maximum monotone matching under positive temporal intersection.

    Cardinality is optimized first and total intersection is only a
    deterministic tie-break.  A one-frame intersection is sufficient, so the
    estimator introduces no IoU or duration threshold.
    """

    indexed_predicted = tuple(
        sorted(enumerate(predicted), key=lambda item: (item[1], item[0]))
    )
    predicted = tuple(interval for _, interval in indexed_predicted)
    target = tuple(sorted(target))
    cardinality = np.zeros(
        (len(predicted) + 1, len(target) + 1), dtype=np.int64
    )
    intersection = np.zeros_like(cardinality)
    decision = np.zeros_like(cardinality, dtype=np.int8)
    for predicted_index, (predicted_start, predicted_end) in enumerate(
        predicted, start=1
    ):
        for target_index, (target_start, target_end) in enumerate(
            target, start=1
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
                        int(
                            cardinality[predicted_index - 1, target_index - 1]
                        )
                        + 1,
                        int(
                            intersection[predicted_index - 1, target_index - 1]
                        )
                        + overlap,
                        2,
                    ),
                )
            best = max(candidates, key=lambda value: (value[0], value[1], value[2]))
            cardinality[predicted_index, target_index] = best[0]
            intersection[predicted_index, target_index] = best[1]
            decision[predicted_index, target_index] = best[2]
    matches = []
    predicted_index = len(predicted)
    target_index = len(target)
    while predicted_index and target_index:
        choice = int(decision[predicted_index, target_index])
        if choice == 2:
            matches.append(indexed_predicted[predicted_index - 1][0])
            predicted_index -= 1
            target_index -= 1
        elif choice == 1:
            target_index -= 1
        else:
            predicted_index -= 1
    return tuple(sorted(matches))


def _fit_role_observation_channel(
    episodes: Sequence[
        tuple[CompetitiveRoleStream, Sequence[TransportEvent]]
    ],
    labels: Mapping[str, Sequence[str]],
) -> tuple[FactorizedRoleObservationChannel, dict[str, object]]:
    """Fit one conjugate deletion/insertion channel on training data only."""

    totals = np.zeros(len(EVENT_ROLES), dtype=np.int64)
    matched = np.zeros(len(EVENT_ROLES), dtype=np.int64)
    predicted_totals = np.zeros(len(EVENT_ROLES), dtype=np.int64)
    origin_counts = np.zeros(
        (len(EVENT_ROLES), 2), dtype=np.int64
    )
    factor_counts = {
        name: np.full(
            (len(EVENT_ROLES), 2, len(labels[name])),
            0.5,
            dtype=np.float64,
        )
        for name in EVENT_FACTORS
    }
    for stream, events in episodes:
        for role, role_name in enumerate(EVENT_ROLES):
            target_intervals = tuple(
                (int(interval.start), int(interval.end))
                for interval in stream.intervals
                if int(interval.role) == role
            )
            role_events = tuple(
                event
                for event in events
                if event.key[0] == role_name
            )
            predicted_intervals = tuple(
                (int(event.start), int(event.end)) for event in role_events
            )
            matched_indices = set(
                _ordered_positive_overlap_matching(
                    predicted_intervals, target_intervals
                )
            )
            totals[role] += len(target_intervals)
            predicted_totals[role] += len(predicted_intervals)
            matched[role] += len(matched_indices)
            for event_index, event in enumerate(role_events):
                origin = 0 if event_index in matched_indices else 1
                origin_counts[role, origin] += 1
                for name in EVENT_FACTORS:
                    probability = np.asarray(
                        event.factor_probability[name], dtype=np.float64
                    )
                    if (
                        probability.shape != (len(labels[name]),)
                        or not np.isfinite(probability).all()
                        or np.any(probability < 0)
                        or probability.sum() <= 0
                    ):
                        raise ValueError(
                            f"invalid {name} probability in observation channel"
                        )
                    factor_counts[name][role, origin] += (
                        probability / probability.sum()
                    )
    if np.any(totals == 0):
        raise ValueError("each role needs training events for deletion estimation")
    if np.any(origin_counts == 0):
        raise ValueError(
            "each role needs genuine and inserted observations for channel fitting"
        )
    missed = totals - matched
    missed_probability = (missed.astype(np.float64) + 0.5) / (
        totals.astype(np.float64) + 1.0
    )
    origin_probability = (origin_counts.astype(np.float64) + 0.5) / (
        origin_counts.sum(axis=-1, keepdims=True).astype(np.float64) + 1.0
    )
    factor_probability = {
        name: values / values.sum(axis=-1, keepdims=True)
        for name, values in factor_counts.items()
    }
    channel = FactorizedRoleObservationChannel(
        factor_labels={
            name: tuple(str(value) for value in labels[name])
            for name in EVENT_FACTORS
        },
        missed_event_probability=tuple(
            float(value) for value in missed_probability
        ),  # type: ignore[arg-type]
        origin_probability=tuple(
            tuple(float(value) for value in row)
            for row in origin_probability
        ),  # type: ignore[arg-type]
        origin_factor_probability={
            name: tuple(
                tuple(
                    tuple(float(value) for value in distribution)
                    for distribution in role
                )
                for role in values
            )
            for name, values in factor_probability.items()
        },  # type: ignore[arg-type]
    )
    channel.validate()
    audit = {
        "schema": "factorized_role_observation_channel_fit",
        "estimators": {
            
        },
        "source": "fold-training event intervals and frozen observer output",
        "anomaly_labels_used": False,
        "role_statistics": {
            role: {
                "annotated_events": int(totals[index]),
                "decoded_events": int(predicted_totals[index]),
                "matched_events": int(matched[index]),
                "missed_events": int(missed[index]),
                "missed_event_probability": float(
                    missed_probability[index]
                ),
                "inserted_events": int(origin_counts[index, 1]),
                "insertion_probability": float(
                    origin_probability[index, 1]
                ),
            }
            for index, role in enumerate(EVENT_ROLES)
        },
    }
    return channel, audit


def _predicted_key(
    event: DecodedEvent,
    labels: Mapping[str, Sequence[str]],
) -> EventKey:
    values = []
    for name in EVENT_FACTORS:
        index = int(np.argmax(event.factor_log_probability[name]))
        values.append(str(labels[name][index]))
    return str(EVENT_ROLES[event.role]), values[0], values[1], values[2]


def _decoded_factor_probability(
    event: DecodedEvent,
) -> dict[str, tuple[float, ...]]:
    return {
        name: tuple(float(np.exp(value)) for value in event.factor_log_probability[name])
        for name in FACTOR_ORDER
    }


def _decoded_events(
    stream: CompetitiveRoleStream,
    decoded: Sequence[DecodedEvent],
    labels: Mapping[str, Sequence[str]],
    *,
    attach_targets: bool = True,
) -> tuple[TransportEvent, ...]:
    result = []
    for event in decoded:
        correctness, error_types = (
            _target_for_decoded(event, stream) if attach_targets else (-1, ())
        )
        result.append(
            TransportEvent(
                key=_predicted_key(event, labels),
                factor_probability=_decoded_factor_probability(event),
                start=event.start_index,
                onset=event.onset_index,
                end=event.end_index,
                correctness=correctness,
                error_types=error_types,
                right_censored=event.right_censored,
            )
        )
    return tuple(sorted(result, key=lambda item: (item.onset, item.start, item.key[0])))


def _observer_events(
    model: object,
    metadata: Mapping[str, object],
    stream: CompetitiveRoleStream,
    priors: Mapping[str, torch.Tensor],
    labels: Mapping[str, Sequence[str]],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], tuple[DecodedEvent, ...]]:
    """Run the checkpoint-declared causal observer without protocol guessing."""

    role_change = stream.role_change_features
    if metadata.get("role_change_evidence") is False:
        role_change = None
    elif metadata.get("role_change_evidence") is True and role_change is None:
        raise ValueError("observer checkpoint requires role-change evidence")
    architecture = metadata.get(
        "observer_architecture", "role_event_observer"
    )
    if architecture == "marked_renewal_observer":
        if not isinstance(model, MarkedRenewalRoleObserver):
            raise TypeError("checkpoint architecture and observer type disagree")
        evidence, posterior = model(
            stream.global_features.to(device),
            stream.role_features.to(device),
            stream.timestamps_sec.to(device=device, dtype=torch.float32),
            None if role_change is None else role_change.to(device),
        )
        factors = {
            name: values.float().cpu()
            for name, values in evidence.factor_logits.items()
        }
        decoded = decode_posterior_median_events(
            posterior.conditional_mark_probability.cpu(),
            factors,
            priors,
            model.initial_phase_probability.cpu(),
        )
        return factors, decoded


def _execution_mode(correctness: int) -> int:
    if correctness == CORRECTNESS_INDEX["error"]:
        return ERROR
    if correctness == CORRECTNESS_INDEX["correction"]:
        return 2
    return 0


def _fit_role_execution_mode_dynamics(
    streams: Sequence[CompetitiveRoleStream],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Estimate one role-symmetric mode law from role-local event chains.

    Counts are pooled across roles, but transitions are formed only between
    consecutive events of the same role.  This keeps statistical strength
    while preventing interleaved actions by the other hand from becoming
    spurious execution-mode transitions.
    """

    initial_counts = np.zeros(len(EXECUTION_MODES), dtype=np.int64)
    transition_counts = np.zeros(
        (len(EXECUTION_MODES), len(EXECUTION_MODES)), dtype=np.int64
    )
    mode_counts = np.zeros(len(EXECUTION_MODES), dtype=np.int64)
    role_sequences = 0
    for stream in streams:
        for role in range(len(EVENT_ROLES)):
            intervals = sorted(
                (
                    interval
                    for interval in stream.intervals
                    if int(interval.role) == role
                ),
                key=lambda interval: (
                    int(interval.onset),
                    int(interval.start),
                ),
            )
            if not intervals:
                continue
            role_sequences += 1
            previous = None
            for interval in intervals:
                mode = _execution_mode(int(interval.correctness))
                mode_counts[mode] += 1
                if previous is None:
                    initial_counts[mode] += 1
                else:
                    transition_counts[previous, mode] += 1
                previous = mode
    if role_sequences == 0 or np.any(mode_counts == 0):
        raise ValueError("role-local training chains must contain every execution mode")
    initial = initial_counts.astype(np.float64) + 0.5
    initial /= initial.sum()
    transition = transition_counts.astype(np.float64) + 0.5
    transition /= transition.sum(axis=-1, keepdims=True)
    return transition, initial, {
        "schema": "role_execution_mode_dynamics",
        "source": "complete fold-training event sequence grouped by role",
        "role_parameters_shared": True,
        "role_sequences": int(role_sequences),
        "observer_conditioned": False,
        "validation_labels_used": False,
        "mode_counts": {
            name: int(mode_counts[index])
            for index, name in enumerate(EXECUTION_MODES)
        },
        "initial_counts": initial_counts.tolist(),
        "transition_counts": transition_counts.tolist(),
        "initial_probability": initial.tolist(),
        "transition_probability": transition.tolist(),
    }
