from __future__ import annotations

from dataclasses import replace

from provia.data.project_single_actor import project_episode_to_single_actor_completions
from provia.data.schema import CanonicalEpisode, LabelAvailability, Segment


def _episode() -> CanonicalEpisode:
    return CanonicalEpisode(
        dataset="public",
        video_id="v1",
        task_id="task",
        split="val",
        video_path=__file__,
        fps=30.0,
        frame_count=300,
        duration_sec=10.0,
        subject_id=None,
        environment_id=None,
        segments=(
            Segment(
                segment_id="a",
                start_sec=0.0,
                end_sec=2.01,
                action_id="1",
                action_text="pick part",
                correctness="correct",
            ),
            Segment(
                segment_id="b",
                start_sec=1.0,
                end_sec=2.04,
                action_id="2",
                action_text="place part",
                correctness="error",
                error_types=("wrong_order",),
            ),
        ),
        state_observations=(),
        procedure=None,
        interventions=(),
        decision_windows=(),
        hand_tracks_path=None,
        object_tracks_path=None,
        availability=LabelAvailability.from_partial(
            available=("temporal_segments", "action_semantics", "error_binary")
        ),
        source_files=(__file__,),
    )


def test_single_actor_projection_compounds_same_sampled_completion() -> None:
    value = project_episode_to_single_actor_completions(
        _episode(), sample_rate_hz=4.0
    )
    assert value.split == "validation"
    assert len(value.segments) == 1
    event = value.segments[0]
    assert event.hand == "left"
    assert event.start_sec == event.onset_sec == event.end_sec == 2.0
    assert event.correctness == "error"
    assert event.error_types == ("wrong_order",)
    assert event.attributes["source_segment_ids"] == ("a", "b")
    assert event.attributes["semantic_encoding"] == "atomic_native_action_class"
    assert value.attributes["single_actor_projection"]["inactive_role"] == "right_structural_null"


def test_single_actor_projection_preserves_complete_native_factors() -> None:
    source = _episode()
    segments = tuple(
        replace(
            segment,
            verb="pick" if index == 0 else "place",
            noun="part",
            tool=None,
        )
        for index, segment in enumerate(source.segments)
    )
    value = project_episode_to_single_actor_completions(
        replace(source, segments=segments),
        sample_rate_hz=4.0,
    )
    event = value.segments[0]
    assert event.verb == "pick+place"
    assert event.noun == "part"
    assert event.tool == "<none>"
    assert event.attributes["semantic_encoding"] == "complete_native_factorization"


def test_single_actor_projection_preserves_complete_verb_without_inventing_noun() -> None:
    source = _episode()
    source = replace(
        source,
        segments=tuple(
            replace(segment, verb="pick" if index == 0 else "place")
            for index, segment in enumerate(source.segments)
        ),
    )
    value = project_episode_to_single_actor_completions(
        source, sample_rate_hz=4.0
    )
    event = value.segments[0]
    assert event.verb == "pick+place"
    assert event.noun == "<none>"
    assert event.tool == "<none>"
    assert event.attributes["semantic_encoding"] == "available_native_factorization"
    assert event.attributes["native_factors_preserved"] == ("verb",)
