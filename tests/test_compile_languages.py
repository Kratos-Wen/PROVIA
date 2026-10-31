from provia.automaton.compile_languages import (
    _medoid_index,
    _normalized_sequence_distance,
)


def test_normalized_sequence_distance_is_symmetric_and_bounded() -> None:
    first = ("a", "b", "c")
    second = ("a", "x")
    forward = _normalized_sequence_distance(first, second)
    backward = _normalized_sequence_distance(second, first)
    assert forward == backward
    assert 0.0 <= forward <= 1.0
    assert _normalized_sequence_distance(first, first) == 0.0


def test_medoid_selects_the_observed_consensus_trace() -> None:
    video_ids = ("outlier", "center", "near")
    signatures = (
        ("x", "y", "z"),
        ("a", "b", "c"),
        ("a", "b", "d"),
    )
    selected, distances = _medoid_index(video_ids, signatures)
    assert video_ids[selected] == "center"
    assert distances[selected] == min(distances)


def test_medoid_tie_break_is_video_id_not_input_order() -> None:
    selected, _ = _medoid_index(("z", "a"), (("x",), ("y",)))
    assert selected == 1
