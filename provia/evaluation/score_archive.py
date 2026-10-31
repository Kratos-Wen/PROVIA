"""Write score archives on the exact native-event support used by main metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.streaming_events import CORRECTNESS_INDEX


def read_native_event_score_archive(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Load and validate native-event score support."""

    with np.load(path, allow_pickle=False) as archive:
        required = {"target", "score", "video_id"}
        available = set(archive.files)
        if frozenset(available) not in {
            frozenset(required),
            frozenset(required | {"mode"}),
        }:
            raise ValueError("unexpected native-event archive schema")
        target = np.asarray(archive["target"], dtype=np.int64)
        score = np.asarray(archive["score"], dtype=np.float64)
        video_id = np.asarray(archive["video_id"], dtype=np.str_)
        mode = (
            np.asarray(archive["mode"], dtype=np.int64)
            if "mode" in available
            else None
        )
    if not (
        target.ndim == score.ndim == video_id.ndim == 1
        and target.shape == score.shape == video_id.shape
        and len(target)
        and np.isin(target, (0, 1)).all()
        and np.isfinite(score).all()
        and np.all(video_id != "")
        and len(np.unique(target)) == 2
    ):
        raise ValueError("malformed native-event score archive")
    if mode is not None and not (
        mode.ndim == 1
        and mode.shape == target.shape
        and np.isin(mode, tuple(CORRECTNESS_INDEX.values())).all()
        and np.array_equal(target, mode == CORRECTNESS_INDEX["error"])
    ):
        raise ValueError("correctness modes disagree with binary error targets")
    return target, score, video_id, mode is not None


def write_native_event_score_archive(
    path: Path,
    *,
    native_mode: np.ndarray,
    native_score: np.ndarray,
    video_id: np.ndarray,
) -> str:
    """Atomically archive every native event, including uncovered zero scores.

    Native-event AP and AUROC penalize missed event proposals by retaining their
    zero scores.  Statistical exports must use that same support; coverage is a
    diagnostic and must never filter this archive.
    """

    mode = np.asarray(native_mode, dtype=np.int64)
    score = np.asarray(native_score, dtype=np.float64)
    videos = np.asarray(video_id, dtype=np.str_)
    if not (
        mode.ndim == score.ndim == videos.ndim == 1
        and mode.shape == score.shape == videos.shape
        and len(mode)
        and np.isfinite(score).all()
        and np.all(videos != "")
    ):
        raise ValueError("native-event score archive fields are malformed")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            target=(mode == CORRECTNESS_INDEX["error"]).astype(np.int64),
            mode=mode,
            score=score,
            video_id=videos,
        )
    temporary.replace(path)
    return str(path.resolve())
