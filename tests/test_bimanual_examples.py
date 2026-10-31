from provia.training.prepare_bimanual_examples import _ordered_overlap_pairs


def test_role_matching_is_one_to_one_monotone_without_timing_threshold() -> None:
    predicted = ((0, 4), (3, 8), (10, 12))
    target = ((2, 5), (7, 11))
    pairs = _ordered_overlap_pairs(predicted, target)
    assert len(pairs) == 2
    assert len({predicted_index for predicted_index, _ in pairs}) == 2
    assert len({target_index for _, target_index in pairs}) == 2
    assert pairs == tuple(sorted(pairs))


def test_role_matching_requires_positive_intersection_only() -> None:
    assert _ordered_overlap_pairs(((0, 1), (4, 5)), ((2, 3),)) == ()

