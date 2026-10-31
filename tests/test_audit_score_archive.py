from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from provia.evaluation.audit_score_archive import audit
from provia.data.streaming_events import CORRECTNESS_INDEX
from provia.evaluation.score_archive import write_native_event_score_archive


def _fixture(tmp_path):
    scores = tmp_path / "scores.npz"
    mode = np.asarray(
        [
            CORRECTNESS_INDEX["correct"],
            CORRECTNESS_INDEX["error"],
            CORRECTNESS_INDEX["correct"],
        ],
        dtype=np.int64,
    )
    values = np.asarray([0.1, 0.8, 0.2], dtype=np.float64)
    write_native_event_score_archive(
        scores,
        native_mode=mode,
        native_score=values,
        video_id=np.asarray(["a", "a", "b"]),
    )
    target = (mode == CORRECTNESS_INDEX["error"]).astype(np.int64)
    digest = hashlib.sha256(scores.read_bytes()).hexdigest()
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "evaluation_device": "cuda",
                "split": "test",
                "test_split_accessed": True,
                "metrics": {
                    "primary_streaming_checkpoint": "anytime",
                    "score_archive_sha256": digest,
                    "checkpoints": {
                        "anytime": {
                            "native_event": {
                                "support": 3,
                                "positives": 1,
                                "average_precision": average_precision_score(
                                    target, values
                                ),
                                "auroc": roc_auc_score(target, values),
                            }
                        }
                    },
                },
            }
        )
    )
    return metrics, scores


def test_audit_accepts_matching_native_scores(tmp_path) -> None:
    metrics, scores = _fixture(tmp_path)
    result = audit(
        metrics_path=metrics, scores_path=scores, expected_device="cuda"
    )
    assert result["audit"] == "passed"
    assert result["support"] == 3
    assert result["video_clusters"] == 2


def test_audit_rejects_wrong_device(tmp_path) -> None:
    metrics, scores = _fixture(tmp_path)
    with pytest.raises(ValueError, match="device"):
        audit(metrics_path=metrics, scores_path=scores, expected_device="cpu")
