import torch

from provia.model.completion_transport import FactorizedCompletionTransport
from provia.model.interval_transport import (IntervalContinuationLanguage)


def _labels():
    return {
        "verb": ("a", "b"),
        "noun": ("x", "y"),
        "tool": ("none",),
    }


def _single_language():
    language = IntervalContinuationLanguage(
        factor_labels=_labels(),
        node_ids=("left|a|x", "left|b|y"),
        node_roles=("left", "left"),
        node_possibility={
            "verb": ((1.0, 0.0), (0.0, 1.0)),
            "noun": ((1.0, 0.0), (0.0, 1.0)),
            "tool": ((1.0,), (1.0,)),
        },
        state_count=3,
        start_state=0,
        accepting_states=(2,),
        transitions=((0, 0, 1), (1, 1, 2)),
        source_trace_count=2,
        transition_probability=(1.0, 1.0),
    )
    language.validate()
    return language


def _role_language():
    language = IntervalContinuationLanguage(
        factor_labels=_labels(),
        node_ids=("left|a|x", "right|b|y"),
        node_roles=("left", "right"),
        node_possibility={
            "verb": ((1.0, 0.0), (0.0, 1.0)),
            "noun": ((1.0, 0.0), (0.0, 1.0)),
            "tool": ((1.0,), (1.0,)),
        },
        state_count=4,
        start_state=0,
        accepting_states=(3,),
        transitions=((0, 0, 1), (0, 1, 2), (1, 1, 3), (2, 0, 3)),
        source_trace_count=2,
        transition_probability=(0.5, 0.5, 1.0, 1.0),
    )
    language.validate()
    return language


def _uniform(reference, length=None):
    value = {
        "verb": reference.new_tensor((0.5, 0.5)),
        "noun": reference.new_tensor((0.5, 0.5)),
        "tool": reference.new_tensor((1.0,)),
    }
    if length is None:
        return value
    return {
        name: probability.unsqueeze(0).expand(length, -1).clone()
        for name, probability in value.items()
    }


def _left(reference, length=None):
    value = {
        "verb": reference.new_tensor((1.0, 0.0)),
        "noun": reference.new_tensor((1.0, 0.0)),
        "tool": reference.new_tensor((1.0,)),
    }
    if length is None:
        return value
    return {
        name: probability.unsqueeze(0).expand(length, -1).clone()
        for name, probability in value.items()
    }




def _initial(model, transport):
    procedure = torch.zeros(transport.language.state_count)
    procedure[transport.language.start_state] = 1.0
    return model.initial_belief(procedure)


def test_uniform_observation_is_exactly_neutral_to_reference_nodes():
    transport = FactorizedCompletionTransport(
        _single_language(), 0.0, posterior_predictive_innovation=True
    )
    node = transport.predictive_node_bayes_factor(_uniform(torch.zeros(1), 3))
    torch.testing.assert_close(node, torch.ones_like(node))
    mixed, reference, _, _ = transport.predictive_reference_statistics(
        _uniform(torch.zeros(1), 3)
    )
    torch.testing.assert_close(reference[:, :2], torch.ones_like(reference[:, :2]))
    torch.testing.assert_close(mixed[:, :2], torch.ones_like(mixed[:, :2]))


def test_semantic_agreement_cannot_certify_correct_execution():
    transport = FactorizedCompletionTransport(
        _single_language(), 0.0, posterior_predictive_innovation=True
    )
    matching, reference, _, _ = transport.predictive_reference_statistics(
        _left(torch.zeros(1))
    )
    assert reference[0] == 1.0
    assert matching[0] == 1.0
    assert torch.all(matching <= 1.0)
    mismatch = {
        "verb": torch.tensor((0.0, 1.0)),
        "noun": torch.tensor((0.0, 1.0)),
        "tool": torch.tensor((1.0,)),
    }
    rejected, _, _, _ = transport.predictive_reference_statistics(mismatch)
    torch.testing.assert_close(rejected[0], transport.innovation_probability)


def test_predictive_conditional_transport_preserves_every_path_mass():
    transport = FactorizedCompletionTransport(
        _single_language(), 0.1, posterior_predictive_innovation=True
    )
    factor = _left(torch.zeros(1), 4)
    mixed, _, _, _ = transport.predictive_reference_statistics(factor)
    raw = torch.rand(4, 3)
    observed = raw * mixed
    destination = transport.conditional_predictive_advance_probability_batch(
        observed, factor
    )
    torch.testing.assert_close(
        destination.sum(dim=-1), observed.sum(dim=-1), atol=3e-5, rtol=3e-5
    )










