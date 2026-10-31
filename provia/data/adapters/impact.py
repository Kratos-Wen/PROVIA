"""Adapter for the IMPACT per-hand event streams (two actor streams, one per hand)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from ..schema import (
    CanonicalEpisode,
    DecisionWindow,
    LabelAvailability,
    ProcedureEdge,
    ProcedureNode,
    ProcedureStructure,
    Segment,
    clean_text,
)
from .base import AdapterError, DatasetAdapter, load_json, require, resolve_video_path


def _hand(value: object) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {"left", "left_hand", "lefthand", "l"}:
        return "left"
    if normalized in {"right", "right_hand", "righthand", "r"}:
        return "right"
    raise AdapterError(f"Unsupported bimanual hand label {value!r}")


def _load_cv_roles(path: Optional[str]) -> tuple[dict[str, list[dict[str, Any]]], tuple[str, ...]]:
    if not path:
        return {}, ()
    source = Path(path)
    payload = load_json(source)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("folds"), list):
        raise AdapterError("The split file must contain a folds list")
    roles: dict[str, list[dict[str, Any]]] = {}
    for fold in payload["folds"]:
        fold_id = int(fold["fold"])
        for partition in ("train", "val", "test"):
            for video_id in fold.get(partition, ()):
                roles.setdefault(str(video_id), []).append(
                    {"fold": fold_id, "partition": partition}
                )
    return roles, (str(source.resolve()),)


def _load_procedure(path: Optional[str]) -> Optional[ProcedureStructure]:
    if not path:
        return None
    source = Path(path)
    payload = load_json(source)
    if not isinstance(payload, Mapping):
        raise AdapterError("The procedure graph must be an object")
    raw_nodes = payload.get("nodes", payload.get("steps"))
    raw_edges = payload.get("edges")
    if isinstance(raw_nodes, Mapping):
        nodes = tuple(
            ProcedureNode(node_id=str(key), text=str(value), action_id=str(key))
            for key, value in raw_nodes.items()
        )
    elif isinstance(raw_nodes, list):
        nodes = tuple(
            ProcedureNode(
                node_id=str(item.get("id", index)),
                text=str(item.get("text", item.get("description", item.get("id", index)))),
                action_id=str(item.get("action_id", item.get("id", index))),
                attributes={
                    key: value
                    for key, value in item.items()
                    if key not in {"id", "text", "description", "action_id"}
                },
            )
            for index, item in enumerate(raw_nodes)
        )
    else:
        raise AdapterError("The procedure graph lacks nodes/steps")
    if not isinstance(raw_edges, list):
        raise AdapterError("The procedure graph lacks an edges list")
    edges: list[ProcedureEdge] = []
    for item in raw_edges:
        if isinstance(item, Mapping):
            edges.append(
                ProcedureEdge(
                    source=str(item["source"]),
                    target=str(item["target"]),
                    relation=str(item.get("relation", "precedes")),
                    confidence=float(item["confidence"])
                    if item.get("confidence") is not None
                    else None,
                )
            )
        elif isinstance(item, list) and len(item) == 2:
            edges.append(ProcedureEdge(source=str(item[0]), target=str(item[1])))
        else:
            raise AdapterError(f"Malformed procedure edge {item!r}")
    return ProcedureStructure(
        structure_id=str(
            payload.get("structure_id", "impact:reference")
        ),
        nodes=nodes,
        edges=tuple(edges),
        source=str(payload.get("source", "manual_reference")),
        verified=payload.get("verified"),
        attributes={"native_path": str(source.resolve())},
    )


def _decision_windows(
    path: Optional[str], video_id: str, event_ids: Mapping[str, str], fps: float
) -> tuple[DecisionWindow, ...]:
    if not path:
        return ()
    payload = load_json(path)
    if not isinstance(payload, Mapping):
        raise AdapterError("Decision annotations must be an object by video_id")
    result: list[DecisionWindow] = []
    for index, item in enumerate(payload.get(video_id, ())):
        native_event = str(item.get("event_id", ""))
        if native_event not in event_ids:
            raise AdapterError(
                f"{video_id}: decision {index} references unknown event {native_event!r}"
            )

        def time_value(seconds_key: str, frame_key: str) -> Optional[float]:
            if item.get(seconds_key) is not None:
                return float(item[seconds_key])
            if item.get(frame_key) is not None:
                return float(item[frame_key]) / fps
            return None

        onset = time_value("anomaly_onset_sec", "anomaly_onset_frame")
        commitment = time_value("commitment_sec", "commitment_frame")
        if onset is None or commitment is None:
            raise AdapterError(
                f"{video_id}: decision {index} needs onset and commitment/PNR"
            )
        result.append(
            DecisionWindow(
                window_id=str(item.get("window_id", f"pnr_{index:04d}")),
                segment_id=event_ids[native_event],
                anomaly_onset_sec=onset,
                commitment_sec=commitment,
                recovery_start_sec=time_value("recovery_start_sec", "recovery_start_frame"),
                recovery_end_sec=time_value("recovery_end_sec", "recovery_end_frame"),
                actionable=item.get("actionable"),
                source=str(item.get("source", "manual_annotation")),
            )
        )
    return tuple(result)


class ImpactAdapter(DatasetAdapter):
    dataset_name = "impact"

    def convert(self) -> Iterable[CanonicalEpisode]:
        annotation_dir = Path(require(self.config, "annotation_dir"))
        pattern = str(self.config.get("annotation_glob", "*_parsed_annotations.json"))
        annotation_paths = sorted(annotation_dir.glob(pattern))
        if not annotation_paths:
            raise FileNotFoundError(
                f"No bimanual annotations match {annotation_dir / pattern}"
            )
        dataset_name = self.dataset_name
        cv_roles, split_sources = _load_cv_roles(self.config.get("split_file"))
        procedure = _load_procedure(self.config.get("procedure_graph_file"))
        decision_path = self.config.get("decision_annotations")
        video_patterns = tuple(
            self.config.get("video_patterns", ("{video_id}.mp4", "{video_id}.MP4"))
        )
        task_id = str(self.config.get("task_id", "industrial_assembly"))

        for annotation_path in annotation_paths:
            payload = load_json(annotation_path)
            if not isinstance(payload, Mapping):
                raise AdapterError(
                    f"Bimanual annotation is not an object: {annotation_path}"
                )
            video_id = str(payload.get("video_id") or annotation_path.name[: -len("_parsed_annotations.json")])
            fps = float(payload.get("fps", self.config.get("fps", 15.0)))
            if fps <= 0:
                raise AdapterError(f"{video_id}: invalid fps {fps}")
            frame_count = int(payload["frame_count"]) if payload.get("frame_count") is not None else None
            raw_events = payload.get("events")
            if not isinstance(raw_events, list):
                raise AdapterError(f"{video_id}: events must be a list")
            segments: list[Segment] = []
            event_ids: dict[str, str] = {}
            for index, event in enumerate(raw_events):
                native_event_id = str(event.get("event_id", index))
                segment_id = f"event_{index:05d}"
                if native_event_id in event_ids:
                    raise AdapterError(f"{video_id}: duplicate event_id {native_event_id}")
                event_ids[native_event_id] = segment_id
                start = int(event["start_frame"])
                onset = int(event["contact_onset_frame"])
                end = int(event["end_frame"])
                if not start <= onset <= end:
                    raise AdapterError(
                        f"{video_id}:{native_event_id}: invalid start/onset/end {start}/{onset}/{end}"
                    )
                anomaly_label = str(event.get("anomaly_label", "normal") or "normal").strip().lower()
                has_anomaly = bool(event.get("has_anomaly", anomaly_label not in {"normal", "recovery"}))
                if anomaly_label == "normal" and has_anomaly:
                    raise AdapterError(
                        f"{video_id}:{native_event_id}: normal label with has_anomaly=true"
                    )
                if anomaly_label == "recovery":
                    correctness = "correction"
                    error_types = ()
                elif anomaly_label == "normal":
                    correctness = "correct"
                    error_types = ()
                else:
                    correctness = "error"
                    error_types = (anomaly_label,)
                segments.append(
                    Segment(
                        segment_id=segment_id,
                        start_sec=start / fps,
                        onset_sec=onset / fps,
                        end_sec=end / fps,
                        observed=True,
                        action_id=native_event_id,
                        action_text=None,
                        verb=clean_text(event.get("verb")),
                        noun=clean_text(
                            event.get("target_object_name")
                            or event.get("noun_object_name")
                        ),
                        tool=clean_text(event.get("tool_object_name")),
                        hand=_hand(event.get("hand")),
                        correctness=correctness,
                        error_types=error_types,
                        native_action_type=anomaly_label,
                        attributes={
                            "start_frame": start,
                            "onset_frame": onset,
                            "end_frame": end,
                            "native_event_id": native_event_id,
                        },
                    )
                )
            decision_windows = _decision_windows(
                str(decision_path) if decision_path else None,
                video_id,
                event_ids,
                fps,
            )
            hand_tracks = None
            hand_track_root = self.config.get("hand_tracks_root")
            if hand_track_root:
                hand_pattern = str(
                    self.config.get("hand_tracks_pattern", "{video_id}.handtracks.*.json")
                ).format(video_id=video_id)
                matches = sorted(Path(hand_track_root).glob(hand_pattern))
                if len(matches) > 1:
                    raise AdapterError(
                        f"{video_id}: multiple hand-track files match {hand_pattern}; provide an explicit index"
                    )
                if matches:
                    hand_tracks = str(matches[0].resolve())
            object_tracks = str(annotation_path.resolve()) if payload.get("tracks") else None
            native_video = clean_text(payload.get("video_path"))
            video_path = resolve_video_path(
                self.config.get("video_root"),
                video_patterns,
                video_id=video_id,
            )
            if (
                self.config.get("recursive_video_search", False)
                and self.config.get("video_root")
                and (video_path is None or not Path(video_path).is_file())
            ):
                matches: list[Path] = []
                root = Path(self.config["video_root"])
                for suffix in (".mp4", ".MP4"):
                    matches.extend(root.glob(f"**/{video_id}{suffix}"))
                unique_matches = sorted({path.resolve() for path in matches})
                if (
                    len(unique_matches) > 1
                    and self.config.get("recursive_video_prefer_shallow", False)
                ):
                    root_resolved = root.resolve()
                    relative_depth = {
                        path: len(path.relative_to(root_resolved).parts)
                        for path in unique_matches
                    }
                    minimum_depth = min(relative_depth.values())
                    unique_matches = [
                        path
                        for path in unique_matches
                        if relative_depth[path] == minimum_depth
                    ]
                if len(unique_matches) != 1:
                    raise AdapterError(
                        f"{video_id}: recursive video lookup found "
                        f"{len(unique_matches)} candidates"
                    )
                video_path = str(unique_matches[0])
            if video_path is None and native_video is not None:
                video_path = str(Path(native_video).expanduser())
            roles = cv_roles.get(video_id, [])
            source_files = [str(annotation_path.resolve()), *split_sources]
            if decision_path:
                source_files.append(str(Path(decision_path).resolve()))
            if procedure is not None:
                source_files.append(str(Path(self.config["procedure_graph_file"]).resolve()))
            availability = LabelAvailability.from_partial(
                available=(
                    "temporal_segments",
                    "action_semantics",
                    "error_binary",
                    "error_type",
                    "correction",
                    "hand_identity",
                ),
                partial=tuple(
                    capability
                    for capability, present in (
                        ("hand_tracks", hand_tracks is not None),
                        ("object_tracks", object_tracks is not None),
                        ("procedure_graph", procedure is not None),
                        ("prospective_pnr", bool(decision_windows)),
                    )
                    if present
                ),
                notes={
                    "component_state": "No component-state file was supplied to this adapter.",
                    "prospective_pnr": (
                        "Only explicitly supplied decision annotations are used; contact onset is never treated as PNR."
                    ),
                },
            )
            yield CanonicalEpisode(
                dataset=dataset_name,
                video_id=video_id,
                task_id=task_id,
                split="cross_validation" if roles else str(self.config.get("split", "unspecified")),
                video_path=video_path,
                fps=fps,
                frame_count=frame_count,
                duration_sec=frame_count / fps if frame_count is not None else None,
                subject_id=video_id.split("_", 1)[0],
                environment_id=None,
                segments=tuple(segments),
                state_observations=(),
                procedure=procedure,
                interventions=(),
                decision_windows=decision_windows,
                hand_tracks_path=hand_tracks,
                object_tracks_path=object_tracks,
                availability=availability,
                source_files=tuple(dict.fromkeys(source_files)),
                attributes={"cross_validation_roles": roles},
            )
