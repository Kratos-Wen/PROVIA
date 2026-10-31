"""HoloAssist adapter for released processed action/intervention labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..schema import (
    CanonicalEpisode,
    Intervention,
    LabelAvailability,
    Segment,
    clean_text,
)
from .base import (
    AdapterError,
    DatasetAdapter,
    load_json,
    load_text_split_files,
    require,
    resolve_video_path,
)


def _lookup(values: Sequence[Any], index: Any, name: str) -> str:
    position = int(index)
    if not 0 <= position < len(values):
        raise AdapterError(f"HoloAssist {name} index {position} outside [0,{len(values)-1}]")
    return str(values[position])


class HoloAssistAdapter(DatasetAdapter):
    dataset_name = "holoassist"

    def convert(self) -> Iterable[CanonicalEpisode]:
        label_path = Path(require(self.config, "label_file"))
        class_path = Path(require(self.config, "class_file"))
        labels = load_json(label_path)
        classes = load_json(class_path)
        if not isinstance(labels, Mapping) or not isinstance(classes, Mapping):
            raise AdapterError("HoloAssist processed label/class files must be objects")
        fine = labels.get("fine_grained_action")
        if not isinstance(fine, Mapping):
            raise AdapterError("HoloAssist label file lacks fine_grained_action")
        split_files = require(self.config, "split_files")
        if not isinstance(split_files, Mapping):
            raise AdapterError("HoloAssist split_files must be an object")
        split_index = load_text_split_files(
            {str(key): str(value) for key, value in split_files.items()}
        )
        intervention_key = str(
            self.config.get("intervention_key", "intervention_prediction_1")
        )
        intervention_labels = labels.get(intervention_key, {})
        if not isinstance(intervention_labels, Mapping):
            raise AdapterError(f"HoloAssist {intervention_key} must be an object")
        action_names = classes.get("fine_grained_action", ())
        verbs = classes.get("fine_grained_verb", ())
        nouns = classes.get("fine_grained_noun", ())
        correctness_names = classes.get("fine_grained_correctness", ())
        task_names = classes.get("task_type", ())
        intervention_names = classes.get("intervention_prediction", ())
        for name, values in (
            ("fine_grained_action", action_names),
            ("fine_grained_verb", verbs),
            ("fine_grained_noun", nouns),
            ("fine_grained_correctness", correctness_names),
            ("task_type", task_names),
            ("intervention_prediction", intervention_names),
        ):
            if not isinstance(values, list):
                raise AdapterError(f"HoloAssist class map {name} must be a list")
        fps = float(self.config.get("fps", 30.0))
        video_patterns = tuple(
            self.config.get(
                "video_patterns", ("{video_id}/Export_py/Video_compress.mp4",)
            )
        )
        include_unassigned = bool(self.config.get("include_unassigned", False))
        invalid_interval_policy = str(
            self.config.get("invalid_interval_policy", "error")
        )
        if invalid_interval_policy not in {"error", "quarantine"}:
            raise AdapterError(
                "HoloAssist invalid_interval_policy must be 'error' or 'quarantine'"
            )

        for video_id in sorted(fine):
            if video_id not in split_index and not include_unassigned:
                continue
            raw_segments = fine[video_id]
            if not isinstance(raw_segments, list):
                raise AdapterError(f"HoloAssist {video_id}: actions must be a list")
            segments: list[Segment] = []
            task_ids: set[int] = set()
            quarantined_segments: list[dict[str, Any]] = []
            for index, item in enumerate(raw_segments):
                if not isinstance(item, list) or len(item) != 3:
                    raise AdapterError(
                        f"HoloAssist {video_id}: action {index} must be [start,end,labels]"
                    )
                start, end, target = item
                if not isinstance(target, list) or len(target) < 5:
                    raise AdapterError(
                        f"HoloAssist {video_id}: action {index} target must have five IDs"
                    )
                task_id, action_id, verb_id, noun_id, correctness_id = map(
                    int, target[:5]
                )
                task_ids.add(task_id)
                start_sec = float(start)
                end_sec = float(end)
                if end_sec < start_sec:
                    issue = {
                        "native_index": index,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "target": list(target),
                        "reason": "end_precedes_start",
                    }
                    if invalid_interval_policy == "error":
                        raise AdapterError(
                            f"HoloAssist {video_id}: action {index} has end "
                            f"{end_sec} before start {start_sec}; set "
                            "invalid_interval_policy='quarantine' only after "
                            "auditing the source annotation"
                        )
                    quarantined_segments.append(issue)
                    continue
                correctness_name = _lookup(
                    correctness_names, correctness_id, "correctness"
                )
                correctness = (
                    "correct"
                    if correctness_name.strip().lower() == "correct action"
                    else "error"
                )
                segments.append(
                    Segment(
                        segment_id=f"fine_{index:06d}",
                        start_sec=start_sec,
                        end_sec=end_sec,
                        action_id=str(action_id),
                        action_text=_lookup(action_names, action_id, "action"),
                        verb=clean_text(_lookup(verbs, verb_id, "verb")),
                        noun=clean_text(_lookup(nouns, noun_id, "noun")),
                        hand=None,
                        correctness=correctness,
                        error_types=("wrong_action",) if correctness == "error" else (),
                        native_action_type=correctness_name,
                        attributes={
                            "task_id": task_id,
                            "verb_id": verb_id,
                            "noun_id": noun_id,
                        },
                    )
                )
            if len(task_ids) > 1:
                raise AdapterError(
                    f"HoloAssist {video_id}: fine actions contain multiple task IDs {task_ids}"
                )
            task_id = next(iter(task_ids), -1)
            task_name = (
                _lookup(task_names, task_id, "task") if task_id >= 0 else "unknown"
            )
            interventions: list[Intervention] = []
            seen_interventions: set[tuple[float, int]] = set()
            for index, item in enumerate(intervention_labels.get(video_id, ())):
                if not isinstance(item, list) or len(item) != 3:
                    raise AdapterError(
                        f"HoloAssist {video_id}: intervention {index} is malformed"
                    )
                _, observed_onset, target = item
                if not isinstance(target, list) or len(target) < 2:
                    raise AdapterError(
                        f"HoloAssist {video_id}: intervention target is malformed"
                    )
                native_task, intervention_id = map(int, target[:2])
                key = (float(observed_onset), intervention_id)
                if key in seen_interventions:
                    continue
                seen_interventions.add(key)
                interventions.append(
                    Intervention(
                        intervention_id=f"human_{len(interventions):05d}",
                        start_sec=float(observed_onset),
                        end_sec=None,
                        intervention_type=_lookup(
                            intervention_names, intervention_id, "intervention"
                        ),
                        text=None,
                        linked_segment_id=None,
                        source=f"official_processed:{intervention_key}",
                    )
                )
                if task_id >= 0 and native_task != task_id:
                    raise AdapterError(
                        f"HoloAssist {video_id}: action/intervention task IDs disagree"
                    )
            last_time = max(
                [item.end_sec or item.start_sec for item in segments]
                + [item.start_sec for item in interventions]
                + [0.0]
            )
            video_path = resolve_video_path(
                self.config.get("dataset_root"),
                video_patterns,
                video_id=video_id,
            )
            hand_tracks = None
            if self.config.get("dataset_root"):
                candidate = (
                    Path(self.config["dataset_root"])
                    / video_id
                    / "Export_py"
                    / "Hands"
                )
                hand_tracks = str(candidate.resolve()) if candidate.exists() else None
            availability = LabelAvailability.from_partial(
                available=("action_semantics", "error_binary")
                + (() if quarantined_segments else ("temporal_segments",)),
                partial=tuple(
                    capability
                    for capability, present in (
                        ("temporal_segments", bool(quarantined_segments)),
                        ("human_intervention", bool(interventions)),
                        ("hand_tracks", hand_tracks is not None),
                    )
                    if present
                ),
                notes={
                    "temporal_segments": (
                        f"{len(quarantined_segments)} source intervals with end before "
                        "start were quarantined and retained in episode attributes."
                        if quarantined_segments
                        else "All released fine-action intervals passed temporal validation."
                    ),
                    "error_type": "The processed release collapses fine-action correctness to correct/wrong.",
                    "correction": "Corrected-by details require raw annotations; no recovery interval is inferred here.",
                    "procedure_graph": "No complete normative procedure graph is supplied by this adapter.",
                    "human_intervention": "Intervention onset is observed human behavior, not an optimal PNR.",
                    "prospective_pnr": "No latest-safe-intervention ground truth is released.",
                },
            )
            source_files = [str(label_path.resolve()), str(class_path.resolve())]
            source_files.extend(str(Path(path).resolve()) for path in split_files.values())
            yield CanonicalEpisode(
                dataset=self.dataset_name,
                video_id=str(video_id),
                task_id=str(task_id),
                split=split_index.get(str(video_id), "unspecified"),
                video_path=video_path,
                fps=fps,
                frame_count=int(last_time * fps) + 1,
                duration_sec=last_time,
                subject_id=None,
                environment_id=None,
                segments=tuple(segments),
                state_observations=(),
                procedure=None,
                interventions=tuple(interventions),
                decision_windows=(),
                hand_tracks_path=hand_tracks,
                object_tracks_path=None,
                availability=availability,
                source_files=tuple(source_files),
                attributes={
                    "task_name": task_name,
                    "intervention_reference": "observed_human_onset_not_pnr",
                    "quarantined_native_segments": quarantined_segments,
                },
            )
