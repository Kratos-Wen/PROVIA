"""Fixtures shared by the model tests."""

import torch




from provia.model.interval_transport import IntervalContinuationLanguage




def _labels():
    return {
        "verb": ("a", "b"),
        "noun": ("x", "y"),
        "tool": ("none",),
    }


def _language():
    return IntervalContinuationLanguage(
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


def _factors(length: int):
    return {
        "verb": torch.softmax(torch.randn(length, 2), dim=-1),
        "noun": torch.softmax(torch.randn(length, 2), dim=-1),
        "tool": torch.ones(length, 1),
    }
