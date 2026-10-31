from __future__ import annotations

import json

import pytest

from provia.data.io import read_jsonl, validate_corpus, write_jsonl
from provia.data.schema import (
    CAPABILITIES,
    CanonicalEpisode,
    LabelAvailability,
    SchemaError,
    Segment,
)


def _episode(subject: str = "p1", split: str = "train") -> CanonicalEpisode:
    return CanonicalEpisode(
        dataset="fixture",
        video_id=f"{subject}_{split}",
        task_id="task",
        split=split,
        video_path=None,
        fps=10.0,
        frame_count=100,
        duration_sec=10.0,
        subject_id=subject,
        environment_id=None,
        segments=(
            Segment(
                segment_id="observed",
                start_sec=1.0,
                onset_sec=1.5,
                end_sec=2.0,
                action_id="1",
                correctness="correct",
            ),
            Segment(
                segment_id="omitted",
                observed=False,
                action_id="2",
                correctness="error",
                error_types=("omission",),
            ),
        ),
        state_observations=(),
        procedure=None,
        interventions=(),
        decision_windows=(),
        hand_tracks_path=None,
        object_tracks_path=None,
        availability=LabelAvailability.from_partial(
            available=("action_semantics", "error_binary"),
            partial=("temporal_segments",),
        ),
        source_files=(),
    )


def test_schema_round_trip_and_atomic_jsonl(tmp_path):
    path = tmp_path / "manifest.jsonl"
    report = write_jsonl([_episode()], path)
    assert report.episodes == 1
    assert report.untimed_segments == 1
    loaded = list(read_jsonl(path))
    assert loaded == [_episode()]
    summary = json.loads((tmp_path / "manifest.jsonl.summary.json").read_text())
    assert summary["segments"] == 2


def test_unobserved_segment_rejects_invented_interval():
    invalid = Segment(
        segment_id="missing",
        start_sec=1.0,
        end_sec=2.0,
        observed=False,
    )
    assert any("invented interval" in issue for issue in invalid.validate())


def test_availability_requires_every_capability():
    availability = LabelAvailability(status={"error_binary": "available"})
    issues = availability.validate()
    assert any("missing capability" in issue for issue in issues)
    complete = LabelAvailability.from_partial()
    assert set(complete.status) == set(CAPABILITIES)


def test_subject_leakage_is_detected():
    report = validate_corpus(
        [_episode("same", "train"), _episode("same", "test")],
        enforce_subject_disjoint=True,
    )
    assert any("subject leakage" in issue for issue in report.issues)


def test_duplicate_episode_fails_write(tmp_path):
    with pytest.raises(SchemaError, match="duplicate episode_id"):
        write_jsonl([_episode(), _episode()], tmp_path / "bad.jsonl")
