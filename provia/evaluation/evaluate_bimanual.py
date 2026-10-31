"""Evaluate PROVIA on bimanual streams at every causal event prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from provia.evaluation.score_archive import (
    write_native_event_score_archive,
)
from provia.evaluation.bimanual_query import (
    load_role_query_artifact,
    load_role_query_prefix_registry,
    role_inference_view,
)

from provia.data.streaming_events import CORRECTNESS_INDEX
from provia.evaluation.operating_points import (
    _checkpoint_metrics,
    _free_response_false_alarm_scores,
)
from provia.observer.factors import EVENT_FACTORS
from provia.training.common import _atomic_json, _intact, _sha256
from provia.training.completion_inputs import (
    _class_probability,
    _correction_is_supervised,
)
from provia.training.role_inputs import (
    _load_expert_effect,
    _load_transport,
)
from provia.training.role_examples import (
    _load_prefix_registry,
    _prefix_evidence,
    _role_stream,
)
from provia.training.train_bimanual import (
    _build_model as _build_bimanual_model,
    _sequence_forward as _bimanual_sequence_forward,
    _source_with_predictive_role_language_registry,
)
from provia.model.bimanual import PROVIA_BIMANUAL_ARCHITECTURE


FROC_BUDGETS = ("0.1", "0.5", "1.0", "2.0")


def _load_frozen_operating_thresholds(
    path: Path,
    *,
    checkpoint_sha256: str,
    examples_sha256: str,
) -> dict[str, dict[str, float]]:
    """Load thresholds selected by the matching validation evaluation."""

    value = json.loads(path.read_text())
    if (
        not isinstance(value, Mapping)
        or value.get("split") != "validation"
        or value.get("test_split_accessed") is not False
        or value.get("checkpoint_sha256") != checkpoint_sha256
        or value.get("examples_sha256") != examples_sha256
    ):
        raise ValueError("frozen validation artifact is incompatible with test run")
    metrics = value.get("metrics")
    checkpoints = metrics.get("checkpoints") if isinstance(metrics, Mapping) else None
    if (
        not isinstance(metrics, Mapping)
        or metrics.get("future_frames_used_by_score") is not False
        or not isinstance(checkpoints, Mapping)
    ):
        raise ValueError("frozen validation artifact lacks causal checkpoint metrics")
    result: dict[str, dict[str, float]] = {}
    for checkpoint in ("onset", "completion", "anytime"):
        entry = checkpoints.get(checkpoint)
        response = entry.get("free_response") if isinstance(entry, Mapping) else None
        points = response.get("operating_points") if isinstance(response, Mapping) else None
        if not isinstance(points, Mapping):
            raise ValueError(f"frozen validation artifact lacks {checkpoint} points")
        thresholds = {}
        for budget in FROC_BUDGETS:
            point = points.get(budget)
            if not isinstance(point, Mapping) or "validation_threshold" not in point:
                raise ValueError(
                    f"frozen validation artifact lacks {checkpoint}/{budget} threshold"
                )
            threshold = float(point["validation_threshold"])
            if np.isnan(threshold):
                raise ValueError("a frozen validation threshold cannot be NaN")
            thresholds[budget] = threshold
        result[checkpoint] = thresholds
    return result


def _native_checkpoint(
    sequence: Mapping[str, object],
    candidate_score: np.ndarray,
    origin_score: np.ndarray,
    detection_options: Sequence[np.ndarray],
) -> dict[str, object]:
    assignment = sequence["native_assignment"].numpy()  # type: ignore[union-attr]
    origin = sequence["origin_target"].numpy().astype(bool)  # type: ignore[union-attr]
    mode = sequence["mode_target"].numpy()  # type: ignore[union-attr]
    candidate_target = origin & (mode == CORRECTNESS_INDEX["error"])
    native = sequence["native_intervals"]
    if not isinstance(native, Sequence):
        raise ValueError("native intervals must be a sequence")
    native_mode = np.asarray(
        [int(interval["mode"]) for interval in native], dtype=np.int64
    )
    native_score = np.zeros(len(native), dtype=np.float64)
    covered = np.zeros(len(native), dtype=bool)
    native_options: list[np.ndarray] = []
    for native_index in range(len(native)):
        candidates = np.flatnonzero(assignment == native_index)
        if len(candidates):
            native_score[native_index] = float(candidate_score[candidates].max())
            covered[native_index] = True
            native_options.append(
                np.concatenate(
                    [detection_options[index] for index in candidates], axis=0
                )
            )
        else:
            native_options.append(np.empty((0, 2), dtype=np.float64))
    false_alarm = _free_response_false_alarm_scores(
        candidate_score, assignment, native_mode
    )
    return {
        "video_id": str(sequence["video_id"]),
        "candidate_score": candidate_score,
        "candidate_target": candidate_target.astype(np.int64),
        "origin_score": origin_score,
        "origin_target": origin.astype(np.int64),
        "native_score": native_score,
        "native_mode": native_mode,
        "covered": covered,
        "false_alarm_score": false_alarm,
        "detection_options": native_options,
        "native_categories": tuple(
            tuple(str(value) for value in interval["error_types"])
            for interval in native
        ),
    }


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    sequences: Sequence[Mapping[str, object]],
    feature_root: Path,
    prefix_root: Path,
    prefix_registry: Mapping[str, object],
    labels: Mapping[str, Sequence[str]],
    transport: torch.nn.Module,
    expert_effect: torch.nn.Module,
    device: torch.device,
    class_probability: torch.Tensor,
    correction_supervised: bool,
    sequence_forward: Callable[..., Mapping[str, object]],
    score_output: Path | None = None,
    frozen_operating_thresholds: Mapping[str, Mapping[str, float]] | None = None,
    candidate_dump: list | None = None,
) -> dict[str, object]:
    model.eval()
    endpoint_rows = []
    onset_rows = []
    anytime_rows = []
    duration_minutes = 0.0
    per_video = []
    for sequence in sequences:
        inference = role_inference_view(sequence)
        video_id = str(inference["video_id"])
        frame, timestamp, global_feature, role_feature = _role_stream(
            feature_root, video_id
        )
        event, factor = _prefix_evidence(
            prefix_root,
            prefix_registry,
            video_id,
            frame,
            timestamp,
            labels,
        )
        result = sequence_forward(
            model,
            inference,
            frame,
            timestamp,
            global_feature,
            role_feature,
            event,
            factor,
            labels,
            transport,  # type: ignore[arg-type]
            expert_effect,  # type: ignore[arg-type]
            device,
            class_probability,
            correction_supervised,
            scoring_only=True,
        )
        traces = result["prefix_explanation_probability"]
        start = sequence["start_index"].numpy()  # type: ignore[union-attr]
        onset = sequence["onset_index"].numpy()  # type: ignore[union-attr]
        end = sequence["end_index"].numpy()  # type: ignore[union-attr]
        native = sequence["native_intervals"]
        native_onset = np.asarray(
            [float(interval["onset_time_sec"]) for interval in native],
            dtype=np.float64,
        )
        assignment = sequence["native_assignment"].numpy()  # type: ignore[union-attr]
        endpoint_score = []
        onset_score = []
        anytime_score = []
        endpoint_origin = []
        onset_origin = []
        anytime_origin = []
        endpoint_options = []
        onset_options = []
        anytime_options = []
        for index, probability in enumerate(traces):
            values = probability.numpy()
            anomaly = values[:, 2]
            genuine = values[:, 1:].sum(axis=-1)
            onset_offset = int(onset[index] - start[index])
            endpoint_score.append(float(anomaly[-1]))
            onset_score.append(float(anomaly[onset_offset]))
            anytime_score.append(float(anomaly.max()))
            endpoint_origin.append(float(genuine[-1]))
            onset_origin.append(float(genuine[onset_offset]))
            anytime_origin.append(float(genuine.max()))
            native_index = int(assignment[index])
            reference = (
                native_onset[native_index]
                if native_index >= 0
                else float(timestamp[int(onset[index])])
            )
            times = timestamp[int(start[index]) : int(end[index]) + 1].numpy()
            options = np.column_stack((anomaly, times - reference))
            anytime_options.append(options)
            onset_options.append(options[onset_offset : onset_offset + 1])
            endpoint_options.append(options[-1:])
        endpoint_rows.append(
            _native_checkpoint(
                sequence,
                np.asarray(endpoint_score),
                np.asarray(endpoint_origin),
                endpoint_options,
            )
        )
        onset_rows.append(
            _native_checkpoint(
                sequence,
                np.asarray(onset_score),
                np.asarray(onset_origin),
                onset_options,
            )
        )
        anytime_rows.append(
            _native_checkpoint(
                sequence,
                np.asarray(anytime_score),
                np.asarray(anytime_origin),
                anytime_options,
            )
        )
        if candidate_dump is not None:
            # per-candidate record for alarm-level (untrimmed) protocol evaluation; evaluation bookkeeping only
            native_mode_array = np.asarray([int(interval["mode"]) for interval in native], dtype=np.int64)
            native_end = np.asarray(
                [float(interval.get("end_time_sec", interval["onset_time_sec"])) for interval in native],
                dtype=np.float64,
            )
            assignment_array = np.asarray(assignment, dtype=np.int64)
            candidate_dump.append(
                {
                    "video_id": video_id,
                    "endpoint_score": np.asarray(endpoint_score, dtype=np.float64),
                    "anytime_score": np.asarray(anytime_score, dtype=np.float64),
                    "prefix_options": [np.asarray(o, dtype=np.float64) for o in anytime_options],
                    "assignment": assignment_array,
                    "candidate_mode": np.where(assignment_array >= 0, native_mode_array[np.clip(assignment_array, 0, None)], -1),
                    "origin": assignment_array >= 0,
                    "native_mode": native_mode_array,
                    "native_time": native_end,
                    "decoded_time": np.asarray([float(timestamp[int(e)]) for e in end], dtype=np.float64),
                }
            )
        minutes = (
            float(sequence["stream_end_sec"])
            - float(sequence["stream_start_sec"])
        ) / 60.0
        duration_minutes += minutes
        per_video.append(
            {
                "video_id": video_id,
                "duration_minutes": minutes,
                "decoded_candidates": len(traces),
                "native_events": len(native),
                "native_errors": int(
                    sum(
                        int(interval["mode"]) == CORRECTNESS_INDEX["error"]
                        for interval in native
                    )
                ),
            }
        )

    def combine(
        rows: Sequence[Mapping[str, object]], name: str, checkpoint: str
    ) -> dict[str, object]:
        candidate_score = np.concatenate(
            [row["candidate_score"] for row in rows]
        )
        candidate_target = np.concatenate(
            [row["candidate_target"] for row in rows]
        )
        origin_score = np.concatenate([row["origin_score"] for row in rows])
        origin_target = np.concatenate([row["origin_target"] for row in rows])
        native_score = np.concatenate([row["native_score"] for row in rows])
        native_mode = np.concatenate([row["native_mode"] for row in rows])
        covered = np.concatenate([row["covered"] for row in rows])
        false_alarm = np.concatenate(
            [row["false_alarm_score"] for row in rows]
        )
        options = tuple(
            value for row in rows for value in row["detection_options"]
        )
        categories = tuple(
            value for row in rows for value in row["native_categories"]
        )
        return {
            "causal_checkpoint": name,
            "frames_after_checkpoint_used_by_score": False,
            **_checkpoint_metrics(
                candidate_score=candidate_score,
                candidate_target=candidate_target,
                origin_score=origin_score,
                origin_target=origin_target,
                native_score=native_score,
                native_mode=native_mode,
                covered=covered,
                delay=np.zeros(len(native_score), dtype=np.float64),
                false_alarm_score=false_alarm,
                native_categories=categories,
                duration_minutes=duration_minutes,
                detection_options=options,
                delay_reference="native_onset",
                frozen_operating_thresholds=(
                    None
                    if frozen_operating_thresholds is None
                    else frozen_operating_thresholds[checkpoint]
                ),
            ),
        }

    checkpoints = {
        "onset": combine(onset_rows, "predicted interaction onset", "onset"),
        "completion": combine(endpoint_rows, "predicted event end", "completion"),
        "anytime": combine(
            anytime_rows, "first warning over every causal event prefix", "anytime"
        ),
    }
    score_archive = None
    if score_output is not None:
        native_score = np.concatenate(
            [row["native_score"] for row in anytime_rows]
        )
        native_mode = np.concatenate(
            [row["native_mode"] for row in anytime_rows]
        )
        covered = np.concatenate([row["covered"] for row in anytime_rows])
        native_video_id = np.concatenate(
            [
                np.repeat(str(row["video_id"]), len(row["native_mode"]))
                for row in anytime_rows
            ]
        )
        if not (
            native_score.shape == native_mode.shape == covered.shape
            == native_video_id.shape
        ):
            raise RuntimeError("bimanual score archive fields are not aligned")
        score_archive = write_native_event_score_archive(
            score_output,
            native_mode=native_mode,
            native_score=native_score,
            video_id=native_video_id,
        )
    result = {
        "causal_checkpoint": "every predicted event prefix",
        "future_frames_used_by_score": False,
        "primary_streaming_checkpoint": "anytime",
        "checkpoints": checkpoints,
        "duration_minutes": duration_minutes,
        "per_video": per_video,
    }
    if score_archive is not None:
        assert score_output is not None
        result["score_archive"] = score_archive
        result["score_archive_sha256"] = _sha256(score_output)
    return result


def _checkpoint_relative(checkpoint: Path, value: object) -> Path:
    """A file recorded in checkpoint metadata; released checkpoints record it relative to their own directory."""

    path = Path(str(value))
    return path if path.is_absolute() or path.exists() else checkpoint.parent / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="validation"
    )
    parser.add_argument(
        "--query-examples",
        type=Path,
        help="query-only bimanual candidates linked to frozen development examples",
    )
    parser.add_argument(
        "--query-prefix-evidence",
        type=Path,
        help="annotation-free role evidence compiled for --query-examples",
    )
    parser.add_argument(
        "--frozen-validation",
        type=Path,
        help="matching validation metrics that fix test-time FROC thresholds",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--score-output",
        type=Path,
        help="optional aligned native-event anytime score archive",
    )
    parser.add_argument("--candidate-score-output", type=Path, default=None, help="optional per-candidate dump for alarm-level protocol evaluation")
    parser.add_argument(
        "--rebuilt-inputs",
        action="store_true",
        help="examples and prefix evidence were rebuilt with the released observer; skip their byte-level "
        "bindings to the training-time files (the event ontology is still checked)",
    )
    parser.add_argument(
        "--prefix-evidence",
        type=Path,
        default=None,
        help="role prefix evidence compiled for --examples (validation split with --rebuilt-inputs)",
    )
    args = parser.parse_args()
    if args.rebuilt_inputs and args.split != "test" and args.prefix_evidence is None:
        parser.error("--rebuilt-inputs requires --prefix-evidence outside the test split")
    candidate_dump = [] if args.candidate_score_output is not None else None
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.score_output is not None and args.score_output.exists():
        raise FileExistsError(
            f"refusing to overwrite score archive {args.score_output}"
        )
    if args.split == "test" and args.frozen_validation is None:
        parser.error("test evaluation requires --frozen-validation")
    if args.split != "test" and args.frozen_validation is not None:
        parser.error("--frozen-validation is accepted only for test evaluation")
    artifact = torch.load(args.examples, map_location="cpu", weights_only=False)
    source = artifact.get("metadata")
    train = artifact.get("train")
    query_source = None
    if args.query_examples is None:
        if args.split == "test" or args.query_prefix_evidence is not None:
            parser.error("test evaluation requires query examples and prefix evidence")
        sequences = artifact.get(args.split)
    else:
        if args.split != "test" or args.query_prefix_evidence is None:
            parser.error("query examples are accepted only for a complete test query")
        query_source, sequences = load_role_query_artifact(
            args.query_examples,
            source_examples=args.examples,
            split=args.split,
        )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata")
    checkpoint_sha256 = _sha256(args.checkpoint)
    examples_sha256 = _sha256(args.examples)
    if (
        not isinstance(source, Mapping)
        or source.get("schema")
        != "role_windowed_examples"
        or source.get("test_split_accessed") is not False
        or not isinstance(train, Sequence)
        or not isinstance(sequences, Sequence)
        or not sequences
        or not isinstance(metadata, Mapping)
        or metadata.get("architecture") != PROVIA_BIMANUAL_ARCHITECTURE
        or (not args.rebuilt_inputs and metadata.get("examples_sha256") != examples_sha256)
        or metadata.get("test_split_accessed") is not False
    ):
        raise ValueError("bimanual evaluation artifacts are incompatible")
    frozen_operating_thresholds = (
        None
        if args.frozen_validation is None
        else _load_frozen_operating_thresholds(
            args.frozen_validation,
            checkpoint_sha256=checkpoint_sha256,
            examples_sha256=examples_sha256,
        )
    )
    language_registry = metadata.get("language_registry")
    if language_registry is not None:
        registry_path = _checkpoint_relative(args.checkpoint, language_registry)
        if not _intact(registry_path, metadata.get("language_registry_sha256")):
            raise ValueError("predictive role-language registry changed after training")
        source, _ = _source_with_predictive_role_language_registry(
            source, registry_path, args.examples, rebuilt_examples=args.rebuilt_inputs
        )
    if args.rebuilt_inputs:
        development_prefix_root = args.prefix_evidence
    else:
        development_prefix_root = Path(str(metadata["prefix_evidence"]))
        if _sha256(development_prefix_root / "registry.json") != metadata.get(
            "prefix_evidence_registry_sha256"
        ):
            raise ValueError("role prefix evidence changed after training")
    effect_path = _checkpoint_relative(args.checkpoint, metadata["expert_effect_artifact"])
    if not _intact(effect_path, metadata.get("expert_effect_artifact_sha256")):
        raise ValueError("expert observation posterior changed after training")
    if query_source is None:
        prefix_root = development_prefix_root
        prefix_registry = _load_prefix_registry(prefix_root, args.examples)
    else:
        assert args.query_examples is not None
        assert args.query_prefix_evidence is not None
        prefix_root = args.query_prefix_evidence
        prefix_registry = load_role_query_prefix_registry(
            prefix_root,
            query_examples=args.query_examples,
            source_examples=args.examples,
            split=args.split,
        )
    device = torch.device(args.device)
    correction_supervised = _correction_is_supervised(train)
    model = _build_bimanual_model(
        source,
        device,
        correction_supervised=correction_supervised,
    )
    sequence_forward = _bimanual_sequence_forward
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    transport = _load_transport(
        source, device, posterior_predictive_innovation=True
    )
    expert_effect = _load_expert_effect(source, effect_path, transport, device, rebuilt_examples=args.rebuilt_inputs)
    labels = {
        name: tuple(str(value) for value in source["factor_labels"][name])  # type: ignore[index]
        for name in EVENT_FACTORS
    }
    class_probability = _class_probability(
        train, correction_supervised=correction_supervised
    ).to(device)
    result = {
        "schema": "bimanual_evaluation",
        "split": args.split,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "examples": str(args.examples.resolve()),
        "examples_sha256": examples_sha256,
        "query_examples": (
            None if args.query_examples is None else str(args.query_examples.resolve())
        ),
        "query_examples_sha256": (
            None if args.query_examples is None else _sha256(args.query_examples)
        ),
        "frozen_validation": (
            None
            if args.frozen_validation is None
            else str(args.frozen_validation.resolve())
        ),
        "frozen_validation_sha256": (
            None
            if args.frozen_validation is None
            else _sha256(args.frozen_validation)
        ),
        "frozen_learned_emissions": True,
        "query_annotations_or_future_frames_used": False,
        "query_boundaries_or_labels_used": False,
        "future_frames_used": False,
        "evaluation_device": device.type,
        "evaluation_dtype": "float32",
        "metrics": _evaluate(
            model,
            sequences,
            Path(
                str(
                    source["feature_root"]
                    if query_source is None
                    else query_source["feature_root"]
                )
            ),
            prefix_root,
            prefix_registry,
            labels,
            transport,
            expert_effect,
            device,
            class_probability,
            correction_supervised,
            sequence_forward,
            args.score_output,
            frozen_operating_thresholds,
            candidate_dump=candidate_dump,
        ),
        "test_split_accessed": args.split == "test",
    }
    _atomic_json(args.output, result)
    if candidate_dump is not None:
        np.savez(args.candidate_score_output, videos=np.array(candidate_dump, dtype=object))
    print(json.dumps(result["metrics"]["checkpoints"], sort_keys=True))


if __name__ == "__main__":
    main()
