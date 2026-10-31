
from provia.model.interval_transport import (CommittedReferenceAction, compile_interval_continuations)
from provia.model.reference_event import UncertainReferenceEvent


LABELS = {
    "verb": ("assemble", "remove"),
    "noun": ("part",),
    "tool": ("none",),
}


def _event(verb: int) -> UncertainReferenceEvent:
    return UncertainReferenceEvent(
        verb=tuple(float(index == verb) for index in range(2)),
        noun=(1.0,),
        tool=(1.0,),
    )


def _action(
    role: str,
    verb: int,
    start: float,
    commitment: float,
    end: float,
) -> CommittedReferenceAction:
    return CommittedReferenceAction(
        role=role,
        start=start,
        commitment=commitment,
        end=end,
        event=_event(verb),
    )


def test_compiler_preserves_role_order_and_cross_role_concurrency():
    language = compile_interval_continuations(
        (
            (
                _action("left", 0, 0.0, 0.2, 1.0),
                _action("right", 1, 0.1, 0.3, 1.2),
                _action("left", 1, 1.3, 1.4, 2.0),
            ),
        ),
        factor_labels=LABELS,
    )
    start_edges = [
        edge for edge in language.transitions if edge[0] == language.start_state
    ]
    assert {language.node_roles[node] for _, node, _ in start_edges} == {
        "left",
        "right",
    }
    assert language.state_count == 5
    assert len(language.accepting_states) == 1


def test_compiler_fits_a_jeffreys_predictive_branch_law() -> None:
    assemble = (_action("left", 0, 0.0, 0.2, 1.0),)
    remove = (_action("left", 1, 0.0, 0.2, 1.0),)
    language = compile_interval_continuations(
        (assemble, assemble, assemble, remove), factor_labels=LABELS
    )
    assert language.transition_probability is not None
    start = [
        (language.node_ids[node], language.transition_probability[index])
        for index, (source, node, _) in enumerate(language.transitions)
        if source == language.start_state
    ]
    probability = dict(start)
    # Three versus one traversal with Jeffreys half-counts gives 3.5/5 and
    # 1.5/5. No validation-set branch threshold or temperature is involved.
    assert probability["left|assemble|part"] == 0.7
    assert probability["left|remove|part"] == 0.3
    restored = type(language).from_dict(language.to_dict())
    assert restored == language






def test_exact_language_marks_demonstrated_branch_as_optional():
    common_start = _action("left", 0, 0.0, 0.2, 0.8)
    optional = _action("right", 1, 0.9, 1.0, 1.2)
    common_end = _action("left", 1, 1.3, 1.4, 2.0)
    language = compile_interval_continuations(
        (
            (common_start, common_end),
            (common_start, optional, common_end),
        ),
        factor_labels=LABELS,
    )
    optional_index = language.node_ids.index("right|remove|part")
    assert not language.node_is_globally_necessary(optional_index)
    assert "right|remove|part" in language.optional_nodes()
    assert language.node_is_globally_necessary(
        language.node_ids.index("left|assemble|part")
    )
    assert language.node_is_globally_necessary(
        language.node_ids.index("left|remove|part")
    )




































