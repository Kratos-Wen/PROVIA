from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from provia.data.streaming_events import CORRECTNESS_INDEX
from provia.evaluation.score_archive import (
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
        archived_score = archive["score"]
        archived_video = archive["video_id"]
    expected_target = (mode == CORRECTNESS_INDEX["error"]).astype(np.int64)
    assert np.array_equal(target, expected_target)
    assert np.array_equal(archived_score, score)
    assert np.array_equal(archived_video, video_id)
    assert len(target) == len(mode)
    assert average_precision_score(target, archived_score) == average_precision_score(
        expected_target, score
    )
    assert roc_auc_score(target, archived_score) == roc_auc_score(
        expected_target, score
    )
