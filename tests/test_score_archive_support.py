from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from provia.data.streaming_events import CORRECTNESS_INDEX
from provia.evaluation.score_archive import (
    read_native_event_score_archive,
    write_native_event_score_archive,
)


def test_archive_retains_uncovered_native_event_zero_scores(tmp_path) -> None:
    mode = np.asarray(
        [
            CORRECTNESS_INDEX["correct"],
            CORRECTNESS_INDEX["error"],
            CORRECTNESS_INDEX["error"],
            CORRECTNESS_INDEX["correct"],
        ],
        dtype=np.int64,
    )
    score = np.asarray([0.1, 0.8, 0.0, 0.2], dtype=np.float64)
    video_id = np.asarray(["v1", "v1", "v2", "v2"], dtype=np.str_)
    output = tmp_path / "native_scores.npz"

    write_native_event_score_archive(
        output,
        native_mode=mode,
        native_score=score,
        video_id=video_id,
    )

    with np.load(output, allow_pickle=False) as archive:
        target = archive["target"]
        archived_mode = archive["mode"]
        archived_score = archive["score"]
        archived_video = archive["video_id"]
    expected_target = (mode == CORRECTNESS_INDEX["error"]).astype(np.int64)
    assert np.array_equal(target, expected_target)
    assert np.array_equal(archived_mode, mode)
    assert np.array_equal(archived_score, score)
    assert np.array_equal(archived_video, video_id)
    assert len(target) == len(mode)
    assert average_precision_score(target, archived_score) == average_precision_score(
        expected_target, score
    )
    assert roc_auc_score(target, archived_score) == roc_auc_score(
        expected_target, score
    )
    loaded_target, loaded_score, loaded_video, has_mode = (
        read_native_event_score_archive(output)
    )
    assert has_mode is True
    assert np.array_equal(loaded_target, expected_target)
    assert np.array_equal(loaded_score, score)
    assert np.array_equal(loaded_video, video_id)


def test_archive_audit_rejects_mode_target_disagreement(tmp_path) -> None:
    output = tmp_path / "malformed_native_scores.npz"
    np.savez_compressed(
        output,
        target=np.asarray([0, 1], dtype=np.int64),
        mode=np.asarray(
            [CORRECTNESS_INDEX["error"], CORRECTNESS_INDEX["correct"]],
            dtype=np.int64,
        ),
        score=np.asarray([0.1, 0.9], dtype=np.float64),
        video_id=np.asarray(["v1", "v2"], dtype=np.str_),
    )
    with pytest.raises(
        ValueError,
        match="correctness modes disagree with binary error targets",
    ):
        read_native_event_score_archive(output)


def test_archive_reader_accepts_binary_support(tmp_path) -> None:
    output = tmp_path / "native_scores.npz"
    target = np.asarray([0, 1], dtype=np.int64)
    score = np.asarray([0.2, 0.8], dtype=np.float64)
    video_id = np.asarray(["v1", "v2"], dtype=np.str_)
    np.savez_compressed(
        output,
        target=target,
        score=score,
        video_id=video_id,
    )
    loaded_target, loaded_score, loaded_video, has_mode = (
        read_native_event_score_archive(output)
    )
    assert has_mode is False
    assert np.array_equal(loaded_target, target)
    assert np.array_equal(loaded_score, score)
    assert np.array_equal(loaded_video, video_id)
