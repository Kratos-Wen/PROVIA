import json

import numpy as np
import pytest
import torch

from provia.data.completion_stream import (
    CompletionStream,
    completion_inference_stream,
)
from provia.evaluation.evaluate_baseline_single_actor import (
    FROC_BUDGETS,
    _candidate_slices,
    _causal_window_batches,
    _frozen_thresholds,
    _load_frozen_validation,
)
from provia.baselines.run_single_actor import _causal_window


def test_candidate_slices_match_nonoverlapping_provia_lifecycles():
    lifecycle = _candidate_slices(np.asarray((3, 6, 9), dtype=np.int64))
    assert lifecycle == (slice(1, 4), slice(4, 7), slice(7, 10))
    covered = np.concatenate(
        [np.arange(value.start, value.stop) for value in lifecycle]
    )
    np.testing.assert_array_equal(covered, np.arange(1, 10))


def test_candidate_slices_reject_empty_or_noncausal_completions():
    with pytest.raises(ValueError):
        _candidate_slices(np.empty(0, dtype=np.int64))
    with pytest.raises(ValueError):
        _candidate_slices(np.asarray((3, 3), dtype=np.int64))
    with pytest.raises(ValueError):
        _candidate_slices(np.asarray((0, 2), dtype=np.int64))


def test_frozen_thresholds_are_read_without_refitting():
    artifact = {
        "metrics": {
            "checkpoints": {
                "anytime": {
                    "free_response": {
                        "operating_points": {
                            budget: {"validation_threshold": index / 10.0}
                            for index, budget in enumerate(FROC_BUDGETS, start=1)
                        }
                    }
                }
            }
        }
    }
    assert _frozen_thresholds(artifact, "anytime") == {
        budget: index / 10.0
        for index, budget in enumerate(FROC_BUDGETS, start=1)
    }


def _validation_artifact() -> dict[str, object]:
    checkpoints = {}
    for checkpoint in ("completion", "anytime"):
        checkpoints[checkpoint] = {
            "free_response": {
                "threshold_source": "selected on validation free-response curve",
                "operating_points": {
                    budget: {"validation_threshold": index / 10.0}
                    for index, budget in enumerate(FROC_BUDGETS, start=1)
                },
            }
        }
    return {
        "schema": "streaming_neural_anytime_evaluation",
        "method": "causal_tcn",
        "split": "validation",
        "test_split_accessed": False,
        "future_frames_used": False,
        "query_boundaries_or_labels_used": False,
        "baseline_checkpoint_sha256": "checkpoint",
        "observer_checkpoint_sha256": "observer",
        "frozen_validation": None,
        "metrics": {"checkpoints": checkpoints},
    }


def test_control_test_thresholds_require_matching_validation(tmp_path):
    path = tmp_path / "validation.json"
    path.write_text(json.dumps(_validation_artifact()))
    loaded = _load_frozen_validation(
        path,
        method="causal_tcn",
        checkpoint_sha256="checkpoint",
        observer_checkpoint_sha256="observer",
    )
    assert loaded["split"] == "validation"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("method", "prego"),
        ("split", "test"),
        ("test_split_accessed", True),
        ("baseline_checkpoint_sha256", "different"),
        ("observer_checkpoint_sha256", "different"),
        ("frozen_validation", "nested.json"),
    ),
)
def test_control_test_thresholds_reject_mismatched_provenance(
    tmp_path, field, value
):
    artifact = _validation_artifact()
    artifact[field] = value
    path = tmp_path / "validation.json"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="incompatible"):
        _load_frozen_validation(
            path,
            method="causal_tcn",
            checkpoint_sha256="checkpoint",
            observer_checkpoint_sha256="observer",
        )


def test_vectorized_windows_equal_the_frozen_baseline_definition():
    torch.manual_seed(91)
    feature = torch.randn(13, 5)
    batches = list(_causal_window_batches(feature, window=7, batch_size=4))
    actual = torch.cat([value for _, value in batches], dim=0)
    expected = torch.stack(
        [_causal_window(feature, index, 7) for index in range(len(feature))]
    )
    torch.testing.assert_close(actual, expected)


def test_baseline_query_view_is_invariant_to_all_native_annotations():
    visual = {
        "episode_id": "dataset:video",
        "video_id": "video",
        "task_id": "task",
        "global_features": torch.randn(7, 5),
        "change_features": torch.randn(7, 5),
        "frame_ids": torch.arange(7),
        "timestamps_sec": torch.arange(7, dtype=torch.float64),
    }
    first = CompletionStream(
        **visual,
        event_indices=torch.tensor([2, 5]),
        factor_targets={"verb": torch.tensor([0, 1]), "noun": torch.tensor([1, 0]), "tool": torch.tensor([0, 0])},
        correctness=torch.tensor([0, 1]),
        error_types=((), ("wrong_tool",)),
    )
    changed = CompletionStream(
        **visual,
        event_indices=torch.tensor([1, 6]),
        factor_targets={"verb": torch.tensor([9, 8]), "noun": torch.tensor([7, 6]), "tool": torch.tensor([5, 4])},
        correctness=torch.tensor([2, 2]),
        error_types=(("arbitrary",), ("changed",)),
    )
    assert completion_inference_stream(first) == completion_inference_stream(changed)


def test_continuous_readout_requires_matching_frozen_validation(tmp_path):
    artifact = _validation_artifact()
    path = tmp_path / "validation.json"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="incompatible"):
        _load_frozen_validation(
            path,
            method="causal_tcn",
            checkpoint_sha256="checkpoint",
            observer_checkpoint_sha256="observer",
            readout="posterior_continuous",
        )
    artifact["readout"] = "posterior_continuous"
    path = tmp_path / "validation.json"
    path.write_text(json.dumps(artifact))
    loaded = _load_frozen_validation(
        path,
        method="causal_tcn",
        checkpoint_sha256="checkpoint",
        observer_checkpoint_sha256="observer",
        readout="posterior_continuous",
    )
    assert loaded["readout"] == "posterior_continuous"
