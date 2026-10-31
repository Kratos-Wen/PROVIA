import torch

from provia.observer.factors import EVENT_FACTORS
from provia.model.predictive_state_induction import (
    compile_bayesian_predictive_quotient,
    quotient_interval_language,
)
from provia.model.completion_transport import FactorizedCompletionTransport
from provia.model.interval_transport import IntervalContinuationLanguage


def _source_language(nodes: int) -> IntervalContinuationLanguage:
    labels = {
        "verb": tuple(f"v{index}" for index in range(nodes)),
        "noun": tuple(f"n{index}" for index in range(nodes)),
        "tool": tuple(f"t{index}" for index in range(nodes)),
    }
    possibility = {
        name: tuple(
            tuple(float(column == row) for column in range(nodes))
            for row in range(nodes)
        )
        for name in EVENT_FACTORS
    }
    return IntervalContinuationLanguage(
        factor_labels=labels,
        node_ids=tuple(f"left|v{index}|n{index}" for index in range(nodes)),
        node_roles=tuple("left" for _ in range(nodes)),
        node_possibility=possibility,
        state_count=7,
        start_state=0,
        accepting_states=(3, 6),
        transitions=(
            (0, 0, 1),
            (0, 3, 4),
            (1, 1, 2),
            (2, 2, 3),
            (4, 1, 5),
            (5, 4, 6),
        ),
        source_trace_count=2,
        transition_probability=(0.5, 0.5, 1.0, 1.0, 1.0, 1.0),
    )


def test_predictively_exchangeable_prefixes_share_a_latent_state() -> None:
    traces = ((0, 1, 2), (3, 1, 4))
    quotient = compile_bayesian_predictive_quotient(traces, node_count=5)
    assert quotient.state_count < quotient.source_state_count
    assert quotient.log_bayes_evidence_gain > 0.0
    source_node = {}
    for source, node, target in quotient.transitions:
        source_node.setdefault((source, node), set()).add(target)
    # The same observed continuation remains uncertain over two downstream
    # futures rather than choosing one demonstration or inventing a hard edge.
    assert any(len(targets) == 2 for targets in source_node.values())
    assert all(0.0 < value < 1.0 for value in quotient.state_terminal_probability)
    for source in range(quotient.state_count):
        total = sum(
            probability
            for edge, probability in zip(
                quotient.transitions, quotient.transition_probability
            )
            if edge[0] == source
        )
        if total:
            assert abs(total - 1.0) < 1e-8


def test_ancestor_histories_are_never_pooled() -> None:
    quotient = compile_bayesian_predictive_quotient(
        ((0, 0, 0),), node_count=1
    )
    assert quotient.state_count == quotient.source_state_count == 4
    assert quotient.merged_state_count == 0


def test_compilation_is_byte_stable() -> None:
    traces = ((0, 1, 2), (3, 1, 4), (0, 1, 4))
    first = compile_bayesian_predictive_quotient(traces, node_count=5)
    second = compile_bayesian_predictive_quotient(traces, node_count=5)
    assert first == second


def test_quotient_language_supports_exact_probability_transport() -> None:
    source = _source_language(5)
    language, audit = quotient_interval_language(
        source, ((0, 1, 2), (3, 1, 4))
    )
    language.validate()
    assert language.state_terminal_probability == audit.state_terminal_probability
    transport = FactorizedCompletionTransport(
        language, 0.1, posterior_predictive_innovation=True
    )
    belief = transport.initial_belief.clone()
    factors = {
        name: torch.nn.functional.one_hot(
            torch.tensor(0), num_classes=5
        ).to(torch.float32)
        for name in EVENT_FACTORS
    }
    destination = transport.conditional_predictive_advance_probability_batch(
        belief.unsqueeze(0), factors
    )
    assert torch.isfinite(destination).all()
    assert torch.allclose(destination.sum(), belief.sum(), atol=1e-6)


def test_unobserved_valid_source_interleaving_is_preserved() -> None:
    """Successful references must not redefine a richer source procedure.

    The source language allows either event order.  Only the first order is
    present in the decoded expert stream used for predictive-state induction;
    the unvisited alternative must nevertheless remain a goal-reaching path.
    """

    labels = {
        "verb": ("v0", "v1"),
        "noun": ("n0", "n1"),
        "tool": ("t0", "t1"),
    }
    possibility = {
        name: ((1.0, 0.0), (0.0, 1.0)) for name in EVENT_FACTORS
    }
    source = IntervalContinuationLanguage(
        factor_labels=labels,
        node_ids=("left|v0|n0", "right|v1|n1"),
        node_roles=("left", "right"),
        node_possibility=possibility,
        state_count=4,
        start_state=0,
        accepting_states=(3,),
        transitions=((0, 0, 1), (0, 1, 2), (1, 1, 3), (2, 0, 3)),
        source_trace_count=2,
        transition_probability=(0.5, 0.5, 1.0, 1.0),
    )
    language, _ = quotient_interval_language(source, ((0, 1),))

    def reaches_goal(nodes: tuple[int, ...]) -> bool:
        frontier = {language.start_state}
        for node in nodes:
            frontier = {
                target
                for source_state, edge_node, target in language.transitions
                if source_state in frontier and edge_node == node
            }
        return bool(frontier & set(language.accepting_states))

    assert reaches_goal((0, 1))
    assert reaches_goal((1, 0))
