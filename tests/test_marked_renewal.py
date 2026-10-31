import torch

from provia.automaton.event_decoding import (
    decode_posterior_median_events,
    decode_posterior_median_onsets,
)
from provia.data.streaming_events import StreamingEventInterval
from provia.observer.factors import EVENT_FACTORS
from provia.observer.marked_renewal import (
    END,
    NONE,
    ONSET,
    RENEW_ONSET,
    START,
    START_ONSET,
    MarkedRenewalRoleObserver,
    renewal_mark_targets,
)


def _model() -> MarkedRenewalRoleObserver:
    labels = {name: ("only",) for name in EVENT_FACTORS}
    return MarkedRenewalRoleObserver(
        2,
        labels,
        torch.ones(6),
        torch.tensor((0.8, 0.1, 0.1)),
    )


def test_targets_preserve_adjacent_start_and_onset_as_a_renewal_mark():
    intervals = (
        StreamingEventInterval("a", 0, 0, 1, 2, 0, ()),
        StreamingEventInterval("b", 0, 3, 3, 4, 0, ()),
    )
    target = renewal_mark_targets(intervals, 6)
    assert int(target[1, 0]) == ONSET
    assert int(target[3, 0]) == RENEW_ONSET
    assert int(target[5, 0]) == END


def test_competing_risk_rows_are_normalized_and_illegal_marks_are_zero():
    model = _model()
    intensity = torch.ones(4, 2, 6)
    conditional = model.conditional_mark_probabilities(
        intensity, torch.tensor((0.0, 0.1, 0.2, 0.3))
    )
    assert torch.allclose(
        conditional.sum(dim=-1), torch.ones_like(conditional[..., 0])
    )
    # An idle source cannot emit a bare onset or end.
    assert torch.count_nonzero(conditional[:, :, 0, ONSET]) == 0
    assert torch.count_nonzero(conditional[:, :, 0, END]) == 0
    posterior = model.filter(conditional)
    assert torch.allclose(
        posterior.state_probability.sum(dim=-1),
        torch.ones_like(posterior.state_probability[..., 0]),
    )




def test_posterior_median_decoder_splits_adjacent_interactions_without_gap_rule():
    length = 6
    conditional = torch.zeros(length, 2, 3, 7)
    conditional[..., NONE] = 1.0

    def boundary(index: int, role: int, state: int, mark: int) -> None:
        conditional[index, role, state] = 0.0
        conditional[index, role, state, NONE] = 0.25
        conditional[index, role, state, mark] = 0.75

    boundary(1, 0, 0, START)
    boundary(2, 0, 1, ONSET)
    boundary(3, 0, 2, RENEW_ONSET)
    boundary(5, 0, 2, END)
    factor_logits = {
        name: torch.zeros(length, 2, 1) for name in EVENT_FACTORS
    }
    priors = {name: torch.ones(1) for name in EVENT_FACTORS}
    events = decode_posterior_median_events(
        conditional,
        factor_logits,
        priors,
        torch.tensor((0.8, 0.1, 0.1)),
    )
    left = [event for event in events if event.role == 0]
    assert [(event.start_index, event.onset_index, event.end_index) for event in left] == [
        (1, 2, 2),
        (3, 3, 4),
    ]
    onsets = decode_posterior_median_onsets(
        conditional,
        factor_logits,
        priors,
        torch.tensor((0.8, 0.1, 0.1)),
    )
    left_onsets = [event for event in onsets if event.role == 0]
    assert [(event.start_index, event.onset_index) for event in left_onsets] == [
        (1, 2),
        (3, 3),
    ]
    assert [event.commitment_probability for event in left_onsets] == [0.75, 0.75]


def test_stream_termination_marks_an_open_interaction_as_right_censored():
    length = 5
    conditional = torch.zeros(length, 2, 3, 7)
    conditional[..., NONE] = 1.0
    conditional[1, 0, 0] = 0.0
    conditional[1, 0, 0, NONE] = 0.25
    conditional[1, 0, 0, START_ONSET] = 0.75
    factors = {name: torch.zeros(length, 2, 1) for name in EVENT_FACTORS}
    priors = {name: torch.ones(1) for name in EVENT_FACTORS}
    events = decode_posterior_median_events(
        conditional,
        factors,
        priors,
        torch.tensor((0.8, 0.1, 0.1)),
    )
    left = [event for event in events if event.role == 0]
    assert len(left) == 1
    assert left[0].end_index == length - 1
    assert left[0].right_censored is True


def test_onset_decoder_is_invariant_to_all_future_rows():
    length = 7
    conditional = torch.zeros(length, 2, 3, 7)
    conditional[..., NONE] = 1.0
    conditional[1, 0, 0, NONE] = 0.25
    conditional[1, 0, 0, START] = 0.75
    conditional[2, 0, 1, NONE] = 0.25
    conditional[2, 0, 1, ONSET] = 0.75
    factors = {name: torch.randn(length, 2, 2) for name in EVENT_FACTORS}
    priors = {name: torch.full((2,), 0.5) for name in EVENT_FACTORS}
    initial = torch.tensor((0.8, 0.1, 0.1))
    before = decode_posterior_median_onsets(
        conditional, factors, priors, initial
    )[0]
    conditional[3, 0, 2] = 0.0
    conditional[3, 0, 2, NONE] = 0.25
    conditional[3, 0, 2, END] = 0.75
    for values in factors.values():
        values[3:] = 1000.0 * torch.randn_like(values[3:])
    after = decode_posterior_median_onsets(
        conditional, factors, priors, initial
    )[0]
    assert (before.role, before.start_index, before.onset_index) == (
        after.role,
        after.start_index,
        after.onset_index,
    )
    assert before.commitment_probability == after.commitment_probability
    for name in EVENT_FACTORS:
        assert before.factor_log_probability[name] == after.factor_log_probability[name]
