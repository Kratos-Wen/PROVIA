import numpy as np

from provia.training.prepare_single_actor_examples import _matched_candidates


def test_matching_selects_exactly_one_nearest_candidate_per_occupied_cell() -> None:
    timestamps = np.arange(12, dtype=np.float64)
    assignment, genuine = _matched_candidates(
        np.asarray((2, 6, 10)),
        np.asarray((1, 3, 4, 6, 7, 11)),
        timestamps,
    )
    np.testing.assert_array_equal(assignment, (0, 0, 1, 1, 1, 2))
    np.testing.assert_array_equal(genuine, (True, False, False, True, False, True))
    assert genuine.sum() == len(np.unique(assignment))


def test_matching_uses_earlier_candidate_for_an_exact_distance_tie() -> None:
    timestamps = np.arange(8, dtype=np.float64)
    assignment, genuine = _matched_candidates(
        np.asarray((4,)), np.asarray((3, 5)), timestamps
    )
    np.testing.assert_array_equal(assignment, (0, 0))
    np.testing.assert_array_equal(genuine, (True, False))
