"""Compile visual effect posteriors from successful expert executions."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from provia.observer.factors import EVENT_FACTORS
from provia.observer.role_statistics import role_interval_trajectory_statistics
from provia.training.common import _sha256
from provia.model.step_effects import ExpertEffectPosterior
from provia.model.interval_transport import IntervalContinuationLanguage


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _read_language(path: Path) -> IntervalContinuationLanguage:
    with open(path, encoding="utf-8") as handle:
        return IntervalContinuationLanguage.from_dict(json.load(handle))


def _node_from_probabilities(
    language: IntervalContinuationLanguage,
    role: str,
    factor_probability: Mapping[str, torch.Tensor],
    index: int,
) -> int:
    values = {}
    for name in EVENT_FACTORS:
        probability = factor_probability[name][index]
        labels = language.factor_labels[name]
        if probability.shape != (len(labels),):
            raise ValueError(f"{name} probability and effect language disagree")
        values[name] = labels[int(probability.argmax())]
    node_id = f"{role}|{values['verb']}|{values['noun']}"
    try:
        return language.node_ids.index(node_id)
    except ValueError as error:
        raise ValueError(f"expert observation has unknown node {node_id!r}") from error


def compile_completion_effects(
    examples_path: Path,
    language_registry_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Compile task-local effect laws from the registry's expert videos."""

    if (output_dir / "registry.json").exists() or (
        output_dir / "complete.json"
    ).exists():
        raise FileExistsError(f"refusing to overwrite complete {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = torch.load(examples_path, map_location="cpu", weights_only=False)
    metadata = artifact.get("metadata")
    train = artifact.get("train")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("test_split_accessed") is not False
        or not isinstance(train, Sequence)
    ):
        raise ValueError("completion examples are malformed")
    input_dim = int(metadata.get("input_dim", 0))
    if input_dim <= 0:
        raise ValueError("completion examples lack their visual feature dimension")
    with open(language_registry_path, encoding="utf-8") as handle:
        registry = json.load(handle)
    examples_sha256 = _sha256(examples_path)
    language_registry_sha256 = _sha256(language_registry_path)
    if (
        not isinstance(registry, Mapping)
        or registry.get("test_videos_used") is not False
        or registry.get("validation_videos_used") is not False
        or metadata.get("language_registry_sha256") != language_registry_sha256
    ):
        raise ValueError("language registry and frozen examples disagree")
    root = language_registry_path.parent
    records = {}
    for task, raw_record in sorted(registry["tasks"].items()):
        if not isinstance(raw_record, Mapping):
            raise ValueError("task language record is malformed")
        language_path = root / str(raw_record["language"])
        language = _read_language(language_path)
        expert_videos = {str(value) for value in raw_record["expert_videos"]}
        statistics = []
        node_indices = []
        used_videos = set()
        for sequence in train:
            if not isinstance(sequence, Mapping):
                raise ValueError("completion sequence is malformed")
            if str(sequence["task_id"]) != str(task) or str(
                sequence["video_id"]
            ) not in expert_videos:
                continue
            statistic = sequence["statistic"]
            if not isinstance(statistic, torch.Tensor):
                raise ValueError("completion sequence lacks path statistics")
            factor_probability = {
                name: sequence[f"factor_{name}"] for name in EVENT_FACTORS
            }
            if any(not isinstance(value, torch.Tensor) for value in factor_probability.values()):
                raise ValueError("completion sequence lacks semantic probabilities")
            used_videos.add(str(sequence["video_id"]))
            for event_index in range(len(statistic)):
                statistics.append(statistic[event_index].float())
                node_indices.append(
                    _node_from_probabilities(
                        language,
                        "left",
                        factor_probability,  # type: ignore[arg-type]
                        event_index,
                    )
                )
        if used_videos != expert_videos:
            missing = sorted(expert_videos - used_videos)
            raise ValueError(f"task {task} lacks frozen expert sequences: {missing}")
        if not statistics:
            raise ValueError(f"task {task} has no expert effect observations")
        filename = f"task_{task}.effects.pt"
        effect_path = output_dir / filename
        expected_metadata = {
            "task_id": str(task),
            "source_examples": str(examples_path.resolve()),
            "source_examples_sha256": examples_sha256,
            "source_language": str(language_path.resolve()),
            "source_language_sha256": _sha256(language_path),
            "expert_videos": sorted(expert_videos),
            "expert_observations": len(statistics),
            "input_dim": input_dim,
            "test_split_accessed": False,
        }
        if effect_path.exists():
            existing = torch.load(
                effect_path, map_location="cpu", weights_only=False
            )
            model = ExpertEffectPosterior.from_artifact(existing)
            existing_metadata = existing.get("metadata")
            if not isinstance(existing_metadata, Mapping) or any(
                existing_metadata.get(name) != value
                for name, value in expected_metadata.items()
            ):
                raise ValueError(
                    f"task {task} partial effect artifact has another provenance"
                )
        else:
            sample_tensor = torch.stack(statistics)
            index_tensor = torch.tensor(node_indices, dtype=torch.long)
            model = ExpertEffectPosterior.fit_transition_effects(
                language.node_ids, sample_tensor, index_tensor, input_dim
            )
            effect_metadata = {
                **expected_metadata,
                "nodes_observed": int(
                    (model.node_observation_count > 0).sum()
                ),
            }
            _atomic_torch(effect_path, model.artifact(effect_metadata))
            del sample_tensor, index_tensor
        nodes_observed = int((model.node_observation_count > 0).sum())
        records[str(task)] = {
            "effects": filename,
            "sha256": _sha256(effect_path),
            "expert_videos": sorted(expert_videos),
            "expert_observations": len(statistics),
            "nodes": len(language.node_ids),
            "nodes_observed": nodes_observed,
        }
        del statistics, node_indices, model
        gc.collect()
    result = {
        "schema": "completion_normative_effect_registry",
        "source_examples": str(examples_path.resolve()),
        "source_examples_sha256": examples_sha256,
        "source_language_registry": str(language_registry_path.resolve()),
        "source_language_registry_sha256": language_registry_sha256,
        "tasks": records,
        "validation_videos_used": False,
        "test_videos_used": False,
    }
    _atomic_json(output_dir / "registry.json", result)
    _atomic_json(
        output_dir / "complete.json",
        {"status": "complete", "test_split_accessed": False},
    )
    return result


def _frame_position(frame_ids: np.ndarray, frame: int, path: Path) -> int:
    position = int(np.searchsorted(frame_ids, frame))
    if position >= len(frame_ids) or int(frame_ids[position]) != int(frame):
        raise ValueError(f"{path}: frame {frame} is absent from the expert cache")
    return position


def compile_role_effects(
    language_path: Path,
    references: Sequence[Path],
    feature_root: Path,
    output: Path,
) -> dict[str, object]:
    """Compile role-specific effect laws from separate expert recordings."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    language = _read_language(language_path)
    statistics = []
    node_indices = []
    source_records = []
    input_dim: int | None = None
    for reference_path in references:
        with open(reference_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != "provia_uncertain_expert_reference"
        ):
            raise ValueError(f"{reference_path}: unknown expert-reference schema")
        decoder = payload.get("decoder")
        if not isinstance(decoder, Mapping) or decoder.get("uses_expert_annotations"):
            raise ValueError("expert effects require annotation-free decoded events")
        video_id = str(payload["video_id"])
        cache_path = feature_root / f"{video_id}_stream_features.npz"
        with np.load(cache_path, allow_pickle=False) as cache:
            frame_ids = np.asarray(cache["frame_ids"], dtype=np.int64)
            timestamps = torch.from_numpy(
                np.asarray(cache["timestamps_sec"], dtype=np.float32)
            )
            role_features = torch.from_numpy(
                np.asarray(cache["role_features"], dtype=np.float32)
            )
        current_input_dim = int(role_features.shape[-1])
        if input_dim is None:
            input_dim = current_input_dim
        elif input_dim != current_input_dim:
            raise ValueError("expert references use inconsistent feature dimensions")
        raw_events = payload.get("events")
        if not isinstance(raw_events, list) or not raw_events:
            raise ValueError(f"{reference_path}: expert reference has no events")
        ordered = sorted(
            raw_events,
            key=lambda value: (
                int(value["onset_frame"]),
                int(value["start_frame"]),
                str(value["hand"]),
            ),
        )
        ordered = [
            value
            for value in ordered
            if not bool(value.get("right_censored", False))
        ]
        if not ordered:
            raise ValueError(
                f"{reference_path}: expert reference has no completed events"
            )
        roles = [str(value["hand"]) for value in ordered]
        onset = torch.tensor(
            [
                _frame_position(frame_ids, int(value["onset_frame"]), cache_path)
                for value in ordered
            ],
            dtype=torch.long,
        )
        end = torch.tensor(
            [
                _frame_position(frame_ids, int(value["end_frame"]), cache_path)
                for value in ordered
            ],
            dtype=torch.long,
        )
        _, event_statistics, valid = role_interval_trajectory_statistics(
            role_features, timestamps, roles, onset, end
        )
        for event_index, raw in enumerate(ordered):
            if not bool(valid[event_index]):
                continue
            argmax = raw.get("argmax")
            if not isinstance(argmax, Mapping):
                raise ValueError(f"{reference_path}: event lacks semantic modes")
            node_id = f"{roles[event_index]}|{argmax['verb']}|{argmax['noun']}"
            try:
                node = language.node_ids.index(node_id)
            except ValueError as error:
                raise ValueError(
                    f"{reference_path}: effect event has unknown node {node_id!r}"
                ) from error
            statistics.append(event_statistics[event_index].float())
            node_indices.append(node)
        source_records.append(
            {
                "reference": str(reference_path.resolve()),
                "reference_sha256": _sha256(reference_path),
                "video_id": video_id,
                "feature_cache": str(cache_path.resolve()),
                "feature_cache_sha256": _sha256(cache_path),
                "decoded_events": len(ordered),
                "usable_effects": int(valid.sum()),
            }
        )
    if not statistics:
        raise ValueError("expert references contain no usable visual effects")
    if input_dim is None:
        raise ValueError("expert references lack visual features")
    model = ExpertEffectPosterior.fit_transition_effects(
        language.node_ids,
        torch.stack(statistics),
        torch.tensor(node_indices, dtype=torch.long),
        input_dim,
    )
    metadata = {
        "source_language": str(language_path.resolve()),
        "source_language_sha256": _sha256(language_path),
        "sources": source_records,
        "expert_observations": len(statistics),
        "effect_support_interval": "current predicted onset-to-end interval",
        "right_censored_events_used": False,
        "input_dim": input_dim,
        "nodes_observed": int((model.node_observation_count > 0).sum()),
        "uses_expert_annotations": False,
        "test_split_accessed": False,
    }
    _atomic_torch(output, model.artifact(metadata))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    completion = subparsers.add_parser("completion")
    completion.add_argument("--examples", type=Path, required=True)
    completion.add_argument("--language-registry", type=Path, required=True)
    completion.add_argument("--output-dir", type=Path, required=True)
    role = subparsers.add_parser("role")
    role.add_argument("--language", type=Path, required=True)
    role.add_argument("--reference", type=Path, action="append", required=True)
    role.add_argument("--feature-root", type=Path, required=True)
    role.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "completion":
        result = compile_completion_effects(
            args.examples, args.language_registry, args.output_dir
        )
    else:
        result = compile_role_effects(
            args.language, args.reference, args.feature_root, args.output
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
