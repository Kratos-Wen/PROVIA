"""Threshold-free causal decoding for marked-renewal event observations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from torch import Tensor

from provia.automaton.decoded_events import DecodedEvent
from provia.observer.factors import EVENT_FACTORS, EVENT_PHASES, EVENT_ROLES
from provia.observer.marked_renewal import (
    END,
    MARK_DESTINATION,
    NONE,
    ONSET,
    RENEW_ONSET,
    RENEW_START,
    RENEWAL_MARKS,
    START,
    START_ONSET,
)


@dataclass(frozen=True)
class CausalDecodedOnset:
    """An event identity available when the interaction onset is decoded."""

    role: int
    start_index: int
    onset_index: int
    factor_log_probability: Mapping[str, tuple[float, ...]]
    commitment_probability: float = 1.0

    def __post_init__(self) -> None:
        value = float(self.commitment_probability)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("onset commitment probability must lie in [0,1]")


SEMANTIC_STATISTICS = ("frame-evidence-mean", "onset-frame", "end-frame")


def _event_distribution(
    factor_logits: Mapping[str, Tensor],
    factor_priors: Mapping[str, Tensor],
    start: int,
    onset: int,
    role: int,
    statistic: str = "frame-evidence-mean",
) -> dict[str, tuple[float, ...]]:
    """Semantic mark of one event.

    ``frame-evidence-mean``: mean frame evidence from start to onset;
    ``onset-frame``: the onset frame alone; ``end-frame``: the prior at
    emission (the mark is read at the end frame when the event closes, see
    :func:`decode_posterior_median_events`).
    """

    if statistic not in SEMANTIC_STATISTICS:
        raise ValueError(f"unknown semantic mark statistic {statistic!r}")
    result = {}
    for name in EVENT_FACTORS:
        if statistic == "end-frame":
            prior = factor_priors[name]
            result[name] = tuple(float(value) for value in prior.log().cpu())
            continue
        if statistic == "onset-frame":
            logits = factor_logits[name][onset, role]
        else:
            logits = factor_logits[name][start : onset + 1, role].mean(dim=0)
        prior = factor_priors[name].to(logits)
        probability = (logits + prior.log()).log_softmax(dim=-1)
        result[name] = tuple(float(value) for value in probability.cpu())
    return result


def decode_posterior_median_events(
    conditional_mark_probability: Tensor,
    factor_logits: Mapping[str, Tensor],
    factor_priors: Mapping[str, Tensor],
    initial_phase_probability: Tensor,
    semantic_statistic: str = "frame-evidence-mean",
) -> tuple[DecodedEvent, ...]:
    """Decode each next boundary at its causal posterior median.

    The survival probability is reset after every decoded renewal.  Crossing
    one half is therefore the definition of a posterior median, not a
    validation-tuned confidence or duration threshold.  Compound renewal
    marks can close one event and open the next at the same sampled frame.
    Event identity is fixed at onset using only the observed prefix.
    """

    if conditional_mark_probability.ndim != 4 or conditional_mark_probability.shape[1:] != (
        len(EVENT_ROLES), len(EVENT_PHASES), len(RENEWAL_MARKS)
    ):
        raise ValueError("conditional marks must have shape [time,2,3,7]")
    length = len(conditional_mark_probability)
    if length <= 0:
        raise ValueError("cannot decode an empty renewal stream")
    if tuple(factor_logits) != EVENT_FACTORS or tuple(factor_priors) != EVENT_FACTORS:
        raise ValueError("factor mappings do not follow the canonical order")
    if any(
        value.ndim != 3 or value.shape[:2] != (length, len(EVENT_ROLES))
        for value in factor_logits.values()
    ):
        raise ValueError("semantic logits do not align with renewal marks")
    if initial_phase_probability.shape != (len(EVENT_PHASES),):
        raise ValueError("initial phase law has an incompatible shape")
    events: list[DecodedEvent] = []
    idle, approach, interaction = range(len(EVENT_PHASES))

    for role in range(len(EVENT_ROLES)):
        state = int(initial_phase_probability.argmax())
        start: int | None = 0 if state != idle else None
        onset: int | None = 0 if state == interaction else None
        distribution = (
            _event_distribution(factor_logits, factor_priors, 0, 0, role, semantic_statistic)
            if onset is not None
            else None
        )
        survival = conditional_mark_probability.new_tensor(1.0)
        cause_mass = conditional_mark_probability.new_zeros(len(RENEWAL_MARKS))

        def close(end: int, *, right_censored: bool = False) -> None:
            nonlocal start, onset, distribution
            if start is not None and onset is not None and distribution is not None:
                mark = distribution
                if semantic_statistic == "end-frame":
                    mark = _event_distribution(
                        factor_logits, factor_priors, start, max(end, onset), role, "onset-frame"
                    )
                events.append(
                    DecodedEvent(
                        role=role,
                        start_index=start,
                        onset_index=onset,
                        end_index=max(end, onset),
                        factor_log_probability=mark,
                        right_censored=right_censored,
                    )
                )
            start, onset, distribution = None, None, None

        for index in range(1, length):
            row = conditional_mark_probability[index, role, state]
            event_probability = row.clone()
            event_probability[NONE] = 0.0
            cause_mass = cause_mass + survival * event_probability
            survival = survival * row[NONE]
            if float(survival) > 0.5:
                continue
            mark = int(cause_mass.argmax())
            if mark == NONE:
                raise RuntimeError("posterior median crossed without a boundary cause")
            if mark == START:
                start, onset, distribution = index, None, None
            elif mark == START_ONSET:
                start, onset = index, index
                distribution = _event_distribution(
                    factor_logits, factor_priors, start, onset, role, semantic_statistic
                )
            elif mark == ONSET:
                if start is None:
                    raise RuntimeError("an onset was decoded without an open event")
                onset = index
                distribution = _event_distribution(
                    factor_logits, factor_priors, start, onset, role, semantic_statistic
                )
            elif mark == END:
                close(index - 1)
            elif mark == RENEW_START:
                close(index - 1)
                start, onset, distribution = index, None, None
            elif mark == RENEW_ONSET:
                close(index - 1)
                start, onset = index, index
                distribution = _event_distribution(
                    factor_logits, factor_priors, start, onset, role, semantic_statistic
                )
            else:
                raise RuntimeError(f"unknown renewal mark {mark}")
            following = MARK_DESTINATION[state][mark]
            if following < 0:
                raise RuntimeError("decoder selected an illegal renewal mark")
            state = int(following)
            survival = conditional_mark_probability.new_tensor(1.0)
            cause_mass.zero_()
        if state == interaction:
            close(length - 1, right_censored=True)
        elif state == approach:
            start, onset, distribution = None, None, None

    return tuple(
        sorted(
            events,
            key=lambda value: (value.start_index, value.onset_index, value.role),
        )
    )


def decode_posterior_median_onsets(
    conditional_mark_probability: Tensor,
    factor_logits: Mapping[str, Tensor],
    factor_priors: Mapping[str, Tensor],
    initial_phase_probability: Tensor,
    semantic_statistic: str = "frame-evidence-mean",
) -> tuple[CausalDecodedOnset, ...]:
    """Emit each event at its causal posterior-median interaction onset.

    Unlike :func:`decode_posterior_median_events`, this decoder does not wait
    for an end boundary. Its state, survival mass, semantic mark, and emitted
    candidates at index ``t`` are functions only of rows ``0..t``. The full
    event decoder remains useful for evaluation assignment, but cannot create
    or alter a candidate that was not already emitted here.
    """

    if conditional_mark_probability.ndim != 4 or conditional_mark_probability.shape[1:] != (
        len(EVENT_ROLES), len(EVENT_PHASES), len(RENEWAL_MARKS)
    ):
        raise ValueError("conditional marks must have shape [time,2,3,7]")
    length = len(conditional_mark_probability)
    if length <= 0:
        raise ValueError("cannot decode an empty renewal stream")
    if tuple(factor_logits) != EVENT_FACTORS or tuple(factor_priors) != EVENT_FACTORS:
        raise ValueError("factor mappings do not follow the canonical order")
    if any(
        value.ndim != 3 or value.shape[:2] != (length, len(EVENT_ROLES))
        for value in factor_logits.values()
    ):
        raise ValueError("semantic logits do not align with renewal marks")
    if initial_phase_probability.shape != (len(EVENT_PHASES),):
        raise ValueError("initial phase law has an incompatible shape")

    onsets: list[CausalDecodedOnset] = []
    idle, approach, interaction = range(len(EVENT_PHASES))
    for role in range(len(EVENT_ROLES)):
        state = int(initial_phase_probability.argmax())
        start: int | None = 0 if state != idle else None
        if state == interaction:
            onsets.append(
                CausalDecodedOnset(
                    role=role,
                    start_index=0,
                    onset_index=0,
                    factor_log_probability=_event_distribution(
                        factor_logits, factor_priors, 0, 0, role, semantic_statistic
                    ),
                    commitment_probability=float(
                        initial_phase_probability[interaction]
                    ),
                )
            )
        survival = conditional_mark_probability.new_tensor(1.0)
        cause_mass = conditional_mark_probability.new_zeros(len(RENEWAL_MARKS))
        for index in range(1, length):
            row = conditional_mark_probability[index, role, state]
            event_probability = row.clone()
            event_probability[NONE] = 0.0
            cause_mass = cause_mass + survival * event_probability
            survival = survival * row[NONE]
            if float(survival) > 0.5:
                continue
            mark = int(cause_mass.argmax())
            if mark == NONE:
                raise RuntimeError("posterior median crossed without a boundary cause")
            if mark == START:
                start = index
            elif mark == START_ONSET:
                start = index
                onsets.append(
                    CausalDecodedOnset(
                        role=role,
                        start_index=index,
                        onset_index=index,
                        factor_log_probability=_event_distribution(
                            factor_logits, factor_priors, index, index, role, semantic_statistic
                        ),
                        commitment_probability=float(cause_mass[mark]),
                    )
                )
            elif mark == ONSET:
                if start is None:
                    raise RuntimeError("an onset was decoded without an open event")
                onsets.append(
                    CausalDecodedOnset(
                        role=role,
                        start_index=start,
                        onset_index=index,
                        factor_log_probability=_event_distribution(
                            factor_logits, factor_priors, start, index, role, semantic_statistic
                        ),
                        commitment_probability=float(cause_mass[mark]),
                    )
                )
            elif mark == END:
                start = None
            elif mark == RENEW_START:
                start = index
            elif mark == RENEW_ONSET:
                start = index
                onsets.append(
                    CausalDecodedOnset(
                        role=role,
                        start_index=index,
                        onset_index=index,
                        factor_log_probability=_event_distribution(
                            factor_logits, factor_priors, index, index, role, semantic_statistic
                        ),
                        commitment_probability=float(cause_mass[mark]),
                    )
                )
            else:
                raise RuntimeError(f"unknown renewal mark {mark}")
            following = MARK_DESTINATION[state][mark]
            if following < 0:
                raise RuntimeError("decoder selected an illegal renewal mark")
            state = int(following)
            survival = conditional_mark_probability.new_tensor(1.0)
            cause_mass.zero_()
    return tuple(
        sorted(
            onsets,
            key=lambda value: (value.start_index, value.onset_index, value.role),
        )
    )
