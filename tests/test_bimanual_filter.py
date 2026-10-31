import torch

from provia.model.interval_transport import (
    ContinuationTransport,
    IntervalContinuationLanguage,
)


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


def _factor(reference, *, right=False, length=None):
    value = {
        "verb": reference.new_tensor((0.0, 1.0) if right else (1.0, 0.0)),
        "noun": reference.new_tensor((0.0, 1.0) if right else (1.0, 0.0)),
        "tool": reference.new_tensor((1.0,)),
    }
    if length is None:
        return value
    return {
        name: probability.unsqueeze(0).expand(length, -1).clone()
        for name, probability in value.items()
    }












def test_role_conditional_endpoint_transport_preserves_each_other_hand_slice():
    transport = ContinuationTransport(
        _role_language(), posterior_predictive_innovation=True
    )
    source = torch.rand(transport.language.state_count, 3, 6)
    mixed, _, _, _ = transport.predictive_reference_statistics(
        _factor(source), 0
    )
    observed = source * mixed[:, None, None]
    destination = transport.conditional_predictive_advance_probability(
        observed, 0, _factor(source)
    )
    torch.testing.assert_close(
        destination.sum(dim=0), observed.sum(dim=0), atol=4e-5, rtol=4e-5
    )


