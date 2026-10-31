"""CaptainCook4D adapter following the official annotation repository."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from ..schema import (
    CanonicalEpisode,
    LabelAvailability,
    ProcedureEdge,
    ProcedureNode,
    ProcedureStructure,
    Segment,
)
from .base import (
    AdapterError,
    DatasetAdapter,
    build_split_index,
    load_json,
    parse_action_text,
    require,
    resolve_video_path,
)


def _slug(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _index_error_annotations(payload: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(payload, list):
        raise AdapterError("CaptainCook4D error_annotations.json must be a list")
    result: dict[str, Mapping[str, Any]] = {}
    for record in payload:
        recording_id = str(record.get("recording_id", "")).strip()
        if not recording_id:
            raise AdapterError("CaptainCook4D error record lacks recording_id")
        if recording_id in result:
            raise AdapterError(f"Duplicate CaptainCook4D error record {recording_id}")
        result[recording_id] = record
    return result


def _load_durations(path: Optional[str]) -> dict[str, float]:
    if not path:
        return {}
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    result: dict[str, float] = {}
    with open(source, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            recording_id = str(row.get("recording_id", "")).strip()
            if recording_id:
                result[recording_id] = float(row["duration(sec)"])
    return result


def _load_graph(path: Path, activity_name: str) -> ProcedureStructure:
    payload = load_json(path)
    steps = payload.get("steps")
    edges = payload.get("edges")
    if not isinstance(steps, Mapping) or not isinstance(edges, list):
        raise AdapterError(f"Malformed CaptainCook4D task graph: {path}")
    nodes = tuple(
        ProcedureNode(node_id=str(node_id), text=str(text), action_id=str(node_id))
        for node_id, text in sorted(steps.items(), key=lambda item: int(item[0]))
    )
    parsed_edges: list[ProcedureEdge] = []
    for index, edge in enumerate(edges):
        if not isinstance(edge, list) or len(edge) != 2:
            raise AdapterError(f"Malformed edge {index} in {path}: {edge!r}")
        parsed_edges.append(
            ProcedureEdge(source=str(edge[0]), target=str(edge[1]))
        )
    return ProcedureStructure(
        structure_id=f"captaincook4d:{_slug(activity_name)}",
        nodes=nodes,
        edges=tuple(parsed_edges),
        source="official_task_graph",
        verified=None,
        attributes={"native_path": str(path.resolve())},
    )


class CaptainCook4DAdapter(DatasetAdapter):
    dataset_name = "captaincook4d"

    def convert(self) -> Iterable[CanonicalEpisode]:
        annotations_root = Path(require(self.config, "annotations_root"))
        complete_path = Path(
            self.config.get(
                "complete_annotations",
                annotations_root / "annotation_json/complete_step_annotations.json",
            )
        )
        error_path = Path(
            self.config.get(
                "error_annotations",
                annotations_root / "annotation_json/error_annotations.json",
            )
        )
        split_path = Path(require(self.config, "split_file"))
        graph_root = Path(
            self.config.get("task_graph_root", annotations_root / "task_graphs")
        )
        metadata_path = self.config.get(
            "video_metadata", annotations_root / "metadata/video_information.csv"
        )

        complete = load_json(complete_path)
        if not isinstance(complete, Mapping):
            raise AdapterError(
                "CaptainCook4D complete_step_annotations.json must be an object"
            )
        errors = _index_error_annotations(load_json(error_path))
        split_index = build_split_index(load_json(split_path))
        durations = _load_durations(str(metadata_path) if metadata_path else None)
        fps = self.config.get("fps")
        fps = float(fps) if fps is not None else None
        video_patterns = tuple(
            self.config.get(
                "video_patterns",
                ("{video_id}.mp4", "{video_id}/video.mp4", "{video_id}/rgb.mp4"),
            )
        )

        for recording_id in sorted(complete):
            record = complete[recording_id]
            if not isinstance(record, Mapping):
                raise AdapterError(f"Record {recording_id} is not an object")
            native_error = errors.get(str(recording_id))
            complete_steps = record.get("steps", ())
            detailed_steps = (
                native_error.get("step_annotations", ())
                if native_error is not None
                else complete_steps
            )
            if len(complete_steps) != len(detailed_steps):
                raise AdapterError(
                    f"{recording_id}: complete/error step counts disagree: "
                    f"{len(complete_steps)} != {len(detailed_steps)}"
                )
            segments: list[Segment] = []
            for index, (basic, detailed) in enumerate(zip(complete_steps, detailed_steps)):
                basic_id = str(basic.get("step_id"))
                detailed_id = str(detailed.get("step_id"))
                if basic_id != detailed_id:
                    raise AdapterError(
                        f"{recording_id}: step {index} ID mismatch {basic_id}!={detailed_id}"
                    )
                start = float(detailed.get("start_time", -1))
                end = float(detailed.get("end_time", -1))
                observed = start >= 0 and end >= 0
                if observed != (end >= 0):
                    raise AdapterError(
                        f"{recording_id}: step {index} has one negative boundary"
                    )
                raw_errors = detailed.get("errors", []) or []
                if not isinstance(raw_errors, list):
                    raise AdapterError(
                        f"{recording_id}: step {index} errors must be a list"
                    )
                error_types = tuple(
                    str(item.get("tag", "Other")).strip() for item in raw_errors
                )
                action_text, verb = parse_action_text(detailed.get("description"))
                segments.append(
                    Segment(
                        segment_id=f"step_{index:04d}",
                        start_sec=start if observed else None,
                        end_sec=end if observed else None,
                        observed=observed,
                        action_id=detailed_id,
                        action_text=action_text,
                        verb=verb,
                        hand=None,
                        correctness="error" if error_types else "correct",
                        error_types=error_types,
                        native_action_type="|".join(error_types) or "normal",
                        attributes={
                            "modified_description": detailed.get("modified_description"),
                            "error_descriptions": [
                                str(item.get("description", "")) for item in raw_errors
                            ],
                            "occurrence_index": index,
                        },
                    )
                )

            activity_name = str(record.get("activity_name", record.get("activity_id")))
            graph_path = graph_root / f"{_slug(activity_name)}.json"
            procedure = _load_graph(graph_path, activity_name) if graph_path.is_file() else None
            duration = durations.get(str(recording_id))
            if duration is None:
                timed_ends = [item.end_sec for item in segments if item.end_sec is not None]
                duration = max(timed_ends) if timed_ends else None
            source_files = [str(complete_path.resolve()), str(error_path.resolve()), str(split_path.resolve())]
            if metadata_path:
                source_files.append(str(Path(metadata_path).resolve()))
            if procedure is not None:
                source_files.append(str(graph_path.resolve()))
            availability = LabelAvailability.from_partial(
                available=(
                    "action_semantics",
                    "error_binary",
                    "error_type",
                ),
                partial=("temporal_segments", "procedure_graph")
                if procedure is not None
                else ("temporal_segments",),
                notes={
                    "temporal_segments": "Missing steps are explicit but have no temporal interval.",
                    "correction": "The official schema has no general recovery/correction interval.",
                    "prospective_pnr": "No commitment or latest-safe-intervention annotation is released.",
                },
            )
            yield CanonicalEpisode(
                dataset=self.dataset_name,
                video_id=str(recording_id),
                task_id=str(record.get("activity_id", activity_name)),
                split=split_index.get(str(recording_id), "unspecified"),
                video_path=resolve_video_path(
                    self.config.get("video_root"),
                    video_patterns,
                    video_id=recording_id,
                    activity_id=record.get("activity_id"),
                    activity_name=_slug(activity_name),
                ),
                fps=fps,
                frame_count=None,
                duration_sec=duration,
                subject_id=str(record.get("person_id"))
                if record.get("person_id") is not None
                else None,
                environment_id=str(record.get("environment"))
                if record.get("environment") is not None
                else None,
                segments=tuple(segments),
                state_observations=(),
                procedure=procedure,
                interventions=(),
                decision_windows=(),
                hand_tracks_path=None,
                object_tracks_path=None,
                availability=availability,
                source_files=tuple(source_files),
                attributes={"activity_name": activity_name},
            )
