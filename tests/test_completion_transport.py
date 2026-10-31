import torch

from provia.model.completion_transport import (FactorizedCompletionTransport)
from provia.model.interval_transport import IntervalContinuationLanguage


def _language() -> IntervalContinuationLanguage:
    language = IntervalContinuationLanguage(
        factor_labels={
            "verb": ("a", "b"),
            "noun": ("x", "y"),
            "tool": ("none",),
        },
        node_ids=("left|a|x", "left|b|y"),
        node_roles=("left", "left"),
        node_possibility={
            "verb": ((1.0, 0.25), (0.2, 1.0)),
            "noun": ((1.0, 0.4), (0.3, 1.0)),
            "tool": ((1.0,), (1.0,)),
        },
        state_count=4,
        start_state=0,
        accepting_states=(3,),
        transitions=((0, 0, 1), (0, 1, 2), (1, 1, 3), (2, 0, 3)),
        source_trace_count=2,
    )
    language.validate()
    return language


























def test_missed_event_closure_exactly_supports_a_revisitable_state() -> None:
    language = IntervalContinuationLanguage(
        factor_labels={
            "verb": ("repeat", "finish"),
            "noun": ("part",),
            "tool": ("none",),
        },
        node_ids=("left|repeat|part", "left|finish|part"),
        node_roles=("left", "left"),
        node_possibility={
            "verb": ((1.0, 0.0), (0.0, 1.0)),
            "noun": ((1.0,), (1.0,)),
            "tool": ((1.0,), (1.0,)),
        },
        state_count=3,
        start_state=0,
        accepting_states=(2,),
        transitions=((0, 0, 1), (1, 0, 0), (1, 1, 2)),
        source_trace_count=2,
        transition_probability=(1.0, 0.5, 0.5),
        state_terminal_probability=(0.1, 0.1, 0.9),
    )
    language.validate()
    transport = FactorizedCompletionTransport(language, 0.2)
    closure = transport.closure.to(torch.float64)
    hidden = torch.tensor(
        ((0.0, 0.2, 0.0), (0.1, 0.0, 0.1), (0.0, 0.0, 0.0)),
        dtype=torch.float64,
    )
    stop = torch.diag(1.0 - hidden.sum(dim=-1))
    torch.testing.assert_close(
        (torch.eye(3, dtype=torch.float64) - hidden) @ closure,
        stop,
        rtol=1e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(
        closure.sum(dim=-1), torch.ones(3, dtype=torch.float64)
    )
