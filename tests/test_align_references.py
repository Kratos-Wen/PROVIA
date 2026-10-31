import numpy as np

from provia.automaton.align_references import (
    ReferenceEvent,
    align_sequences,
    empirical_background,
)


def _event(index, verb, noun):
    return ReferenceEvent(
        source_index=index,
        role="left",
        probability={
            "verb": np.asarray(verb, dtype=np.float64),
            "noun": np.asarray(noun, dtype=np.float64),
            "tool": np.asarray((1.0, 0.0), dtype=np.float64),
        },
    )


def test_bayes_alignment_matches_shared_steps_and_leaves_incidental_event():
    first = (
        _event(0, (0.95, 0.05), (0.9, 0.1)),
        _event(1, (0.05, 0.95), (0.1, 0.9)),
    )
    second = (
        _event(0, (0.9, 0.1), (0.85, 0.15)),
        _event(1, (0.5, 0.5), (0.5, 0.5)),
        _event(2, (0.1, 0.9), (0.05, 0.95)),
    )
    background = empirical_background((first, second))
    result = align_sequences(first, second, background)
    assert result.pairs == ((0, 0), (1, 2))
    assert result.log_bayes_factor > 0.0


def test_different_roles_never_align():
    first = (_event(0, (1.0, 0.0), (1.0, 0.0)),)
    right = ReferenceEvent(0, "right", first[0].probability)
    background = empirical_background((first, (right,)))
    assert align_sequences(first, (right,), background).pairs == ()
