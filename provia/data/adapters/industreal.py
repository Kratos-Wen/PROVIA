"""IndustReal adapter for action, procedure-step, and component-state labels."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from ..schema import (
    CanonicalEpisode,
    LabelAvailability,
    Segment,
    StateObservation,
    clean_text,
)
from .base import AdapterError, DatasetAdapter, load_json, require, resolve_video_path


FPS = 10.0


def _frame_index(value: object) -> int:
    text = Path(str(value).strip()).stem
    try:
        frame = int(text)
    except ValueError as exc:
        raise AdapterError(f"Invalid IndustReal frame name {value!r}") from exc
    if frame < 0:
        raise AdapterError(f"Negative IndustReal frame index {frame}")
    return frame


def _read_rows(path: Path) -> list[list[str]]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.reader(handle) if row]


def _official_split_index(annotations: Path) -> dict[str, str]:
    """Recording to split, from the official train, val and test CSV files."""

    return {
        row[0].strip(): split
        for split in ("train", "val", "test")
        for row in _read_rows(annotations / f"{split}.csv")
    }


def _component_names(
    procedure_info: list[Mapping[str, Any]], configured: Optional[list[str]]
) -> list[str]:
    max_index = max((int(item["state_idx"]) for item in procedure_info), default=-1)
    if configured is not None:
        if len(configured) != max_index + 1:
            raise AdapterError(
                f"component_names has {len(configured)} entries, expected {max_index + 1}"
            )
        return [str(item) for item in configured]
    names = [f"component_{index:02d}" for index in range(max_index + 1)]
    for item in procedure_info:
        action_id = int(item["id"])
        if action_id % 3 != 0:
            continue
        state_index = int(item["state_idx"])
        description = str(item.get("description", names[state_index]))
        for prefix in ("Install ", "Assemble "):
            if description.startswith(prefix):
                description = description[len(prefix) :]
        names[state_index] = description.strip().lower().replace(" ", "_")
    return names


def _load_ar_segments(path: Path, fps: float) -> list[Segment]:
    segments: list[Segment] = []
    for row_index, row in enumerate(_read_rows(path)):
        if len(row) != 5:
            if row_index == 0 and any("action" in item.lower() for item in row):
                continue
            raise AdapterError(f"{path}:{row_index + 1}: expected 5 AR columns")
        try:
            action_id = int(row[1])
            start_frame = _frame_index(row[3])
            end_frame = _frame_index(row[4])
        except (AdapterError, ValueError):
            if row_index == 0:
                continue
            raise
        if end_frame < start_frame:
            raise AdapterError(f"{path}:{row_index + 1}: AR end precedes start")
        action_text = clean_text(row[2])
        segments.append(
            Segment(
                segment_id=f"ar_{row_index:05d}",
                start_sec=start_frame / fps,
                end_sec=end_frame / fps,
                action_id=str(action_id),
                action_text=action_text,
                hand=None,
                correctness=None,
                native_action_type="action_recognition",
                attributes={
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "source": "AR_labels.csv",
                },
            )
        )
    return segments


def _load_psr_segments(
    path: Path, fps: float, procedure_by_id: Mapping[int, Mapping[str, Any]]
) -> list[Segment]:
    result: list[Segment] = []
    for row_index, row in enumerate(_read_rows(path)):
        if len(row) < 3:
            if row_index == 0:
                continue
            raise AdapterError(f"{path}:{row_index + 1}: expected >=3 PSR columns")
        try:
            frame = _frame_index(row[0])
            action_id = int(row[1])
        except (AdapterError, ValueError):
            if row_index == 0:
                continue
            raise
        info = procedure_by_id.get(action_id, {})
        action_text = clean_text(row[2]) or clean_text(info.get("description"))
        incorrect = bool(
            str(info.get("description", action_text or ""))
            .strip()
            .lower()
            .startswith("incorrectly")
        )
        result.append(
            Segment(
                segment_id=f"psr_{row_index:05d}",
                start_sec=frame / fps,
                end_sec=frame / fps,
                action_id=str(action_id),
                action_text=action_text,
                noun=(
                    f"component_{int(info['state_idx']):02d}"
                    if "state_idx" in info
                    else None
                ),
                hand=None,
                correctness="error" if incorrect else "correct",
                error_types=("incorrect_completion",) if incorrect else (),
                native_action_type="procedure_step_completion",
                attributes={
                    "frame": frame,
                    "install": info.get("install"),
                    "state_idx": info.get("state_idx"),
                    "expected_in_assy": info.get("expected_in_assy"),
                    "expected_in_main": info.get("expected_in_main"),
                    "source": path.name,
                },
            )
        )
    return result


def _load_states(
    path: Path, fps: float, component_names: list[str]
) -> list[StateObservation]:
    result: list[StateObservation] = []
    for row_index, row in enumerate(_read_rows(path)):
        if len(row) != len(component_names) + 1:
            if row_index == 0:
                continue
            raise AdapterError(
                f"{path}:{row_index + 1}: expected {len(component_names) + 1} state columns, got {len(row)}"
            )
        try:
            frame = _frame_index(row[0])
            state = [int(value) for value in row[1:]]
        except (AdapterError, ValueError):
            if row_index == 0:
                continue
            raise
        if any(value not in {-1, 0, 1} for value in state):
            raise AdapterError(f"{path}:{row_index + 1}: state outside -1/0/1")
        result.append(
            StateObservation(
                observation_id=f"state_{row_index:05d}",
                time_sec=frame / fps,
                frame_index=frame,
                values=dict(zip(component_names, state)),
                source="PSR_labels_raw.csv",
            )
        )
    return result


class IndustRealAdapter(DatasetAdapter):
    dataset_name = "industreal"

    def convert(self) -> Iterable[CanonicalEpisode]:
        dataset_root = Path(require(self.config, "dataset_root"))
        recordings_root = Path(
            self.config.get("recordings_root", dataset_root / "recordings")
        )
        procedure_path = Path(require(self.config, "procedure_info"))
        procedure_info = load_json(procedure_path)
        if not isinstance(procedure_info, list):
            raise AdapterError("IndustReal procedure_info.json must be a list")
        procedure_by_id = {int(item["id"]): item for item in procedure_info}
        component_names = _component_names(
            procedure_info, self.config.get("component_names")
        )
        fps = float(self.config.get("fps", FPS))
        splits = self.config.get("splits", ("train", "val", "test"))
        if isinstance(splits, str):
            splits = (splits,)
        requested_splits = {str(value).lower() for value in splits}
        flat_labels_root = self.config.get("recording_labels_root")
        recordings: list[tuple[str, Path]] = []
        if flat_labels_root is not None:
            labels_root = Path(flat_labels_root)
            if not labels_root.is_dir():
                raise FileNotFoundError(labels_root)
            split_index = _official_split_index(
                Path(self.config.get("annotations_root", dataset_root / "annotations"))
            )
            for recording in sorted(
                path for path in labels_root.iterdir() if path.is_dir()
            ):
                split = split_index.get(recording.name)
                if split is None:
                    if self.config.get("allow_unassigned_recordings", False):
                        continue
                    raise AdapterError(
                        f"IndustReal recording {recording.name} is absent from the official splits"
                    )
                if split in requested_splits:
                    recordings.append((split, recording))
        else:
            for split in sorted(requested_splits):
                split_root = recordings_root / split
                if not split_root.is_dir():
                    if self.config.get("allow_missing_splits", False):
                        continue
                    raise FileNotFoundError(split_root)
                recordings.extend(
                    (split, recording)
                    for recording in sorted(
                        path for path in split_root.iterdir() if path.is_dir()
                    )
                )

        video_patterns = tuple(self.config.get("video_patterns", ("{video_id}.mp4",)))
        for split, recording in recordings:
            ar_path = recording / "AR_labels.csv"
            raw_state_path = recording / "PSR_labels_raw.csv"
            psr_error_path = recording / "PSR_labels_with_errors.csv"
            psr_path = (
                psr_error_path
                if psr_error_path.is_file()
                else recording / "PSR_labels.csv"
            )
            segments = _load_ar_segments(ar_path, fps)
            segments.extend(_load_psr_segments(psr_path, fps, procedure_by_id))
            states = _load_states(raw_state_path, fps, component_names)
            frame_candidates = [
                item.frame_index for item in states if item.frame_index is not None
            ]
            for segment in segments:
                frame_candidates.extend(
                    int(segment.attributes[key])
                    for key in ("start_frame", "end_frame", "frame")
                    if key in segment.attributes
                )
            last_frame = max(frame_candidates) if frame_candidates else None
            hand_path = recording / "hands.csv"
            object_path = recording / "OD_labels.json"
            source_files = [str(procedure_path.resolve())]
            source_files.extend(
                str(path.resolve())
                for path in (ar_path, psr_path, raw_state_path)
                if path.is_file()
            )
            availability = LabelAvailability.from_partial(
                available=tuple(
                    capability
                    for capability, present in (
                        ("component_state", bool(states)),
                        ("hand_tracks", hand_path.is_file()),
                    )
                    if present
                ),
                partial=tuple(
                    capability
                    for capability, present in (
                        ("temporal_segments", bool(segments)),
                        ("action_semantics", bool(segments)),
                        ("error_binary", bool(segments)),
                        ("error_type", bool(segments)),
                        ("object_tracks", object_path.is_file()),
                    )
                    if present
                ),
                notes={
                    "error_binary": "Errors are state/step-completion labels, not dense action-interval labels.",
                    "hand_identity": "Left/right joints exist but semantic actions are not assigned to a hand.",
                    "correction": "State repair can be derived as a transition but no recovery interval is annotated.",
                    "procedure_graph": "Procedure metadata describe outcomes; a complete legal-order graph is not released.",
                    "prospective_pnr": "No prospective commitment annotation is released.",
                },
            )
            yield CanonicalEpisode(
                dataset=self.dataset_name,
                video_id=recording.name,
                task_id="assembly" if "assy" in recording.name else "maintenance",
                split=str(split),
                video_path=(
                    resolve_video_path(
                        self.config.get("video_root"),
                        video_patterns,
                        video_id=recording.name,
                    )
                    if self.config.get("video_root")
                    else str((recording / "rgb").resolve())
                ),
                fps=fps,
                frame_count=last_frame + 1 if last_frame is not None else None,
                duration_sec=last_frame / fps if last_frame is not None else None,
                subject_id=recording.name.split("_", 1)[0],
                environment_id=None,
                segments=tuple(segments),
                state_observations=tuple(states),
                procedure=None,
                interventions=(),
                decision_windows=(),
                hand_tracks_path=(
                    str(hand_path.resolve()) if hand_path.is_file() else None
                ),
                object_tracks_path=(
                    str(object_path.resolve()) if object_path.is_file() else None
                ),
                availability=availability,
                source_files=tuple(source_files),
                attributes={
                    "procedure_info_path": str(procedure_path.resolve()),
                    "component_names": component_names,
                    "state_observation_semantics": "piecewise_constant_snapshot",
                },
            )
