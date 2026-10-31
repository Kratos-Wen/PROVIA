"""Evaluate bimanual neural baselines at every causal candidate prefix.

Weights, context length, observer candidates, native assignment, and procedure
prior are loaded from artifacts.  No baseline is retrained or selected.
The only added operation is a past-only readout at every frame from the frozen
candidate start through its frozen predicted completion, matching the primary
streaming checkpoint used by the role-aware model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from provia.evaluation.score_archive import (
    write_native_event_score_archive,
)

from provia.baselines.run_single_actor import (
    METHODS,
    _model_and_training,
)
from provia.baselines.run_bimanual import (
    RoleActionVocabulary,
    RolePrefixAnticipator,
)
from provia.data.folds import materialize_cross_validation_fold
from provia.data.io import read_jsonl
from provia.data.role_stream import load_competitive_role_streams
from provia.data.streaming_events import CORRECTNESS_INDEX
from provia.evaluation.operating_points import _checkpoint_metrics
from provia.evaluation.evaluate_bimanual import _native_checkpoint
from provia.evaluation.bimanual_query import load_role_query_artifact
from provia.training.common import _atomic_json, _sha256
from provia.training.role_examples import _role_stream
from provia.model.intents import CompositionalIntentSpace


FROC_BUDGETS = ("0.1", "0.5", "1.0", "2.0")


def _frozen_thresholds(
    artifact: Mapping[str, object] | None, checkpoint: str
) -> Mapping[str, float] | None:
    if artifact is None:
        return None
    metrics = artifact.get("metrics")
    checkpoints = metrics.get("checkpoints") if isinstance(metrics, Mapping) else None
    entry = checkpoints.get(checkpoint) if isinstance(checkpoints, Mapping) else None
    response = entry.get("free_response") if isinstance(entry, Mapping) else None
    points = response.get("operating_points") if isinstance(response, Mapping) else None
    if not isinstance(points, Mapping) or set(points) != set(FROC_BUDGETS):
        raise ValueError(f"frozen validation artifact lacks {checkpoint} points")
    thresholds = {}
    for budget in FROC_BUDGETS:
        point = points[budget]
        if not isinstance(point, Mapping) or "validation_threshold" not in point:
            raise ValueError(f"frozen validation artifact lacks {checkpoint}/{budget}")
        threshold = float(point["validation_threshold"])
        if np.isnan(threshold):
            raise ValueError("a frozen validation threshold cannot be NaN")
        thresholds[budget] = threshold
    return thresholds


def _load_frozen_validation(
    path: Path,
    *,
    method: str,
    checkpoint_sha256: str,
    readout: str = "native_discrete",
) -> Mapping[str, object]:
    value = json.loads(path.read_text())
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != "streaming_role_neural_anytime"
        or value.get("method") != method
        or value.get("split") != "validation"
        or value.get("test_split_accessed") is not False
        or value.get("future_frames_used") is not False
        or value.get("query_boundaries_or_labels_used") is not False
        or value.get("baseline_checkpoint_sha256") != checkpoint_sha256
        or value.get("readout", "native_discrete") != readout
        or value.get("frozen_validation") is not None
    ):
        raise ValueError("frozen role validation artifact is incompatible")
    for checkpoint in ("completion", "anytime"):
        _frozen_thresholds(value, checkpoint)
    return value


def _prefix_windows(
    global_feature: Tensor,
    role_feature: Tensor,
    frame_index: Tensor,
    role: Tensor,
    window: int,
) -> Tensor:
    """Construct exact left-padded causal global/role windows in one batch."""

    if (
        global_feature.ndim != 2
        or role_feature.ndim != 3
        or role_feature.shape[0] != len(global_feature)
        or frame_index.ndim != 1
        or role.shape != frame_index.shape
        or window <= 0
        or torch.any(frame_index < 0)
        or torch.any(frame_index >= len(global_feature))
        or torch.any(role < 0)
        or torch.any(role >= role_feature.shape[1])
    ):
        raise ValueError("malformed causal role-window request")
    offset = torch.arange(window - 1, -1, -1, device=frame_index.device)
    source = (frame_index[:, None] - offset[None]).clamp_min(0)
    global_value = global_feature[source]
    role_value = role_feature[source, role[:, None]]
    return torch.cat((global_value, role_value), dim=-1).to(torch.float32)


@torch.no_grad()
def _candidate_prefix_predictions(
    method: str,
    model: nn.Module,
    global_feature: Tensor,
    role_feature: Tensor,
    timestamp: Tensor,
    start: np.ndarray,
    end: np.ndarray,
    role: np.ndarray,
    window: int,
    short_window: int,
    batch_size: int,
    device: torch.device,
    anticipator: RolePrefixAnticipator | None,
    task: str,
    readout: str = "native_discrete",
) -> tuple[np.ndarray, ...]:
    """Return one score trace per frozen observer candidate."""

    if readout not in {"native_discrete", "posterior_continuous"}:
        raise ValueError(f"unknown baseline readout: {readout}")
    continuous = readout == "posterior_continuous"

    if (
        start.shape != end.shape
        or start.shape != role.shape
        or start.ndim != 1
        or np.any(start < 0)
        or np.any(end < start)
        or np.any(end >= len(global_feature))
    ):
        raise ValueError("candidate lifecycles are malformed")
    candidate_index = np.concatenate(
        [np.full(last - first + 1, index, dtype=np.int64)
         for index, (first, last) in enumerate(zip(start, end))]
    )
    frame_index = np.concatenate(
        [np.arange(first, last + 1, dtype=np.int64)
         for first, last in zip(start, end)]
    )
    role_index = role[candidate_index]
    raw: list[np.ndarray] = []
    model.eval()
    for first in range(0, len(frame_index), batch_size):
        last = min(first + batch_size, len(frame_index))
        windows = _prefix_windows(
            global_feature,
            role_feature,
            torch.from_numpy(frame_index[first:last]),
            torch.from_numpy(role_index[first:last]),
            window,
        ).to(device)
        if method in {"causal_tcn", "mistsense_rgb"}:
            raw.append(torch.sigmoid(model(windows)).cpu().numpy()[:, None])
        elif method == "prego":
            if continuous:
                raw.append(model(windows)[:, -1].softmax(dim=-1).cpu().numpy())
            else:
                raw.append(
                    model(windows)[:, -1].argmax(dim=-1).cpu().numpy()[:, None]
                )
        else:
            raise ValueError(method)
    prediction = np.concatenate(raw, axis=0)
    traces = tuple(prediction[candidate_index == index] for index in range(len(start)))
    if method in {"causal_tcn", "mistsense_rgb"}:
        return tuple(value[:, 0].astype(np.float64, copy=False) for value in traces)
    if method == "prego":
        if anticipator is None:
            raise ValueError("PREGO requires its frozen normal-prefix anticipator")
        completed: list[int] = []
        result = []
        for value in traces:
            expected = anticipator.expected(task, completed)
            if continuous:
                # Recognition--anticipation mismatch as the recognized
                # posterior mass outside the anticipated successor set; the
                # committed prefix still advances by the recognizer's argmax
                # at the candidate end, exactly as in the discrete rule.
                columns = sorted(
                    index
                    for index in expected
                    if 0 <= index < value.shape[1]
                )
                inside = (
                    value[:, columns].sum(axis=-1)
                    if columns
                    else np.zeros(len(value))
                )
                result.append((1.0 - inside).astype(np.float64, copy=False))
                completed.append(int(value[-1].argmax()))
            else:
                result.append(
                    np.asarray(
                        [
                            float(int(action) not in expected)
                            for action in value[:, 0]
                        ],
                        dtype=np.float64,
                    )
                )
                completed.append(int(value[-1, 0]))
        return tuple(result)
    raise ValueError(method)


def _combine(
    rows: Sequence[Mapping[str, object]],
    name: str,
    duration_minutes: float,
    frozen_thresholds: Mapping[str, float] | None = None,
) -> dict[str, object]:
    def concatenate(key: str) -> np.ndarray:
        return np.concatenate([np.asarray(row[key]) for row in rows])

    native_score = concatenate("native_score")
    options = tuple(value for row in rows for value in row["detection_options"])
    categories = tuple(value for row in rows for value in row["native_categories"])
    return {
        "causal_checkpoint": name,
        "frames_after_checkpoint_used_by_score": False,
        **_checkpoint_metrics(
            candidate_score=concatenate("candidate_score"),
            candidate_target=concatenate("candidate_target"),
            origin_score=concatenate("origin_score"),
            origin_target=concatenate("origin_target"),
            native_score=native_score,
            native_mode=concatenate("native_mode"),
            covered=concatenate("covered").astype(bool),
            delay=np.zeros(len(native_score), dtype=np.float64),
            false_alarm_score=concatenate("false_alarm_score"),
            native_categories=categories,
            duration_minutes=duration_minutes,
            detection_options=options,
            delay_reference="native_onset",
            include_origin_metrics=False,
            frozen_operating_thresholds=frozen_thresholds,
        ),
    }


@torch.no_grad()
def _evaluate(
    method: str,
    model: nn.Module,
    sequences: Sequence[Mapping[str, object]],
    feature_root: Path,
    tasks: Mapping[str, str],
    window: int,
    short_window: int,
    batch_size: int,
    device: torch.device,
    anticipator: RolePrefixAnticipator | None,
    frozen_artifact: Mapping[str, object] | None = None,
    score_output: Path | None = None,
    readout: str = "native_discrete",
    candidate_dump: list | None = None,
) -> dict[str, object]:
    completion_rows = []
    anytime_rows = []
    duration_minutes = 0.0
    per_video = []
    for sequence in sequences:
        video_id = str(sequence["video_id"])
        _, timestamp, global_feature, role_feature = _role_stream(
            feature_root, video_id
        )
        start = sequence["start_index"].numpy().astype(np.int64)  # type: ignore[union-attr]
        end = sequence["end_index"].numpy().astype(np.int64)  # type: ignore[union-attr]
        role = sequence["role_index"].numpy().astype(np.int64)  # type: ignore[union-attr]
        traces = _candidate_prefix_predictions(
            method,
            model,
            global_feature,
            role_feature,
            timestamp,
            start,
            end,
            role,
            window,
            short_window,
            batch_size,
            device,
            anticipator,
            tasks[video_id],
            readout,
        )
        native = sequence["native_intervals"]
        assignment = sequence["native_assignment"].numpy().astype(np.int64)  # type: ignore[union-attr]
        native_onset = np.asarray(
            [float(interval["onset_time_sec"]) for interval in native],
            dtype=np.float64,
        )
        completion_score = np.asarray([value[-1] for value in traces])
        anytime_score = np.asarray([value.max() for value in traces])
        completion_options = []
        anytime_options = []
        for index, values in enumerate(traces):
            native_index = int(assignment[index])
            reference = (
                native_onset[native_index]
                if native_index >= 0
                else float(timestamp[int(start[index])])
            )
            times = timestamp[int(start[index]) : int(end[index]) + 1].numpy()
            options = np.column_stack((values, times - reference))
            anytime_options.append(options)
            completion_options.append(options[-1:])
        origin = sequence["origin_target"].numpy().astype(np.float64)  # type: ignore[union-attr]
        completion_rows.append(
            _native_checkpoint(
                sequence, completion_score, origin, completion_options
            )
        )
        anytime_rows.append(
            _native_checkpoint(sequence, anytime_score, origin, anytime_options)
        )
        if candidate_dump is not None:
            # per-candidate record for alarm-level protocol evaluation (bookkeeping only)
            native_mode_array = np.asarray([int(interval["mode"]) for interval in native], dtype=np.int64)
            native_end = np.asarray([float(interval.get("end_time_sec", interval["onset_time_sec"])) for interval in native], dtype=np.float64)
            candidate_dump.append(
                {
                    "video_id": str(video_id),
                    "endpoint_score": np.asarray(completion_score, dtype=np.float64),
                    "anytime_score": np.asarray(anytime_score, dtype=np.float64),
                    "prefix_options": [np.asarray(o, dtype=np.float64) for o in anytime_options],
                    "assignment": assignment.copy(),
                    "candidate_mode": np.where(assignment >= 0, native_mode_array[np.clip(assignment, 0, None)], -1),
                    "origin": assignment >= 0,
                    "native_mode": native_mode_array,
                    "native_time": native_end,
                    "decoded_time": np.asarray([float(timestamp[int(e)]) for e in end], dtype=np.float64),
                    "native_error_types": [tuple(interval.get("error_types", ())) for interval in native],
                }
            )
        minutes = (
            float(sequence["stream_end_sec"]) - float(sequence["stream_start_sec"])
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
    score_archive = None
    if score_output is not None:
        native_score = np.concatenate(
            [np.asarray(row["native_score"]) for row in anytime_rows]
        )
        native_mode = np.concatenate(
            [np.asarray(row["native_mode"]) for row in anytime_rows]
        )
        covered = np.concatenate(
            [np.asarray(row["covered"], dtype=bool) for row in anytime_rows]
        )
        video_id = np.concatenate(
            [
                np.repeat(str(row["video_id"]), len(row["native_mode"]))
                for row in anytime_rows
            ]
        ).astype(np.str_)
        if not (
            native_score.shape == native_mode.shape == covered.shape
            == video_id.shape
        ):
            raise RuntimeError("role score archive is not aligned")
        score_archive = write_native_event_score_archive(
            score_output,
            native_mode=native_mode,
            native_score=native_score,
            video_id=video_id,
        )
    result = {
        "causal_checkpoint": "every frozen predicted-event prefix",
        "future_frames_used_by_score": False,
        "primary_streaming_checkpoint": "anytime",
        "checkpoints": {
            "completion": _combine(
                completion_rows,
                "frozen predicted event completion",
                duration_minutes,
                _frozen_thresholds(frozen_artifact, "completion"),
            ),
            "anytime": _combine(
                anytime_rows,
                "first warning over every causal event prefix",
                duration_minutes,
                _frozen_thresholds(frozen_artifact, "anytime"),
            ),
        },
        "duration_minutes": duration_minutes,
        "per_video": per_video,
    }
    if score_archive is not None:
        assert score_output is not None
        result["score_archive"] = score_archive
        result["score_archive_sha256"] = _sha256(score_output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument(
        "--query-examples",
        type=Path,
        help="query-only bimanual candidates linked to frozen development examples",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument(
        "--training-feature-root",
        type=Path,
        help="development features used only to reconstruct PREGO's frozen prior",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="validation"
    )
    parser.add_argument(
        "--frozen-validation",
        type=Path,
        help="matching validation metrics that fix test-time FROC thresholds",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--score-output",
        type=Path,
        help="optional aligned native-event anytime score archive",
    )
    parser.add_argument(
        "--readout",
        choices=("native_discrete", "posterior_continuous"),
        default="native_discrete",
        help=(
            "frozen-checkpoint score readout: the original discrete rule or "
            "the continuous posterior form of the same native rule"
        ),
    )
    parser.add_argument("--candidate-score-output", type=Path, default=None, help="optional per-candidate dump for alarm-level protocol evaluation")
    args = parser.parse_args()
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
    if args.split == "test" and args.query_examples is None:
        parser.error("test evaluation requires --query-examples")
    if args.split != "test" and args.query_examples is not None:
        parser.error("--query-examples is accepted only for test evaluation")
    checkpoint_path = args.run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata")
    artifact = torch.load(args.examples, map_location="cpu", weights_only=False)
    source = artifact.get("metadata")
    if args.query_examples is None:
        sequences = artifact.get(args.split)
    else:
        _, sequences = load_role_query_artifact(
            args.query_examples,
            source_examples=args.examples,
            split=args.split,
        )
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("schema") != "streaming_role_baseline"
        or metadata.get("method") not in METHODS
        or not isinstance(source, Mapping)
        or source.get("test_split_accessed") is not False
        or not isinstance(sequences, Sequence)
        or not sequences
    ):
        raise ValueError("baseline checkpoint or role examples are malformed")
    method = str(metadata["method"])
    checkpoint_sha256 = _sha256(checkpoint_path)
    frozen_artifact = (
        None
        if args.frozen_validation is None
        else _load_frozen_validation(
            args.frozen_validation,
            method=method,
            checkpoint_sha256=checkpoint_sha256,
            readout=args.readout,
        )
    )
    device = torch.device(args.device)
    first_video = str(sequences[0]["video_id"])
    _, _, first_global, first_role = _role_stream(args.feature_root, first_video)
    input_dim = int(first_global.shape[-1] + first_role.shape[-1])
    action_classes = int(metadata["action_classes"])
    model, _ = _model_and_training(
        method,
        input_dim,
        action_classes if method == "prego" else None,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    episodes = materialize_cross_validation_fold(tuple(read_jsonl(args.manifest)), 0)
    tasks = {value.video_id: value.task_id for value in episodes}
    anticipator = None
    if method == "prego":
        training_episodes = tuple(value for value in episodes if value.split == "train")
        intent_space = CompositionalIntentSpace.from_segments(
            segment for episode in training_episodes for segment in episode.segments
        )
        training_feature_root = args.training_feature_root or args.feature_root
        train = load_competitive_role_streams(
            training_episodes, training_feature_root, intent_space
        )
        vocabulary = RoleActionVocabulary.from_normal_streams(train)
        anticipator = RolePrefixAnticipator(train, tasks, vocabulary)
    metrics = _evaluate(
        method,
        model,
        sequences,
        args.feature_root,
        tasks,
        int(metadata["model_context_frames"]),
        int(metadata["window_selection"]["short_frames"]),
        args.batch_size,
        device,
        anticipator,
        frozen_artifact,
        args.score_output,
        args.readout,
        candidate_dump=candidate_dump,
    )
    result = {
        "schema": "streaming_role_neural_anytime",
        "method": method,
        "readout": args.readout,
        "run_dir": str(args.run_dir.resolve()),
        "baseline_checkpoint": str(checkpoint_path.resolve()),
        "baseline_checkpoint_sha256": checkpoint_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "examples": str(args.examples.resolve()),
        "examples_sha256": _sha256(args.examples),
        "query_examples": (
            None if args.query_examples is None else str(args.query_examples.resolve())
        ),
        "query_examples_sha256": (
            None if args.query_examples is None else _sha256(args.query_examples)
        ),
        "split": args.split,
        "weights_or_architecture_refit": False,
        "validation_or_test_metrics_used_for_selection": False,
        "query_annotations_or_future_frames_used": False,
        "query_boundaries_or_labels_used": False,
        "future_frames_used": False,
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
        "test_split_accessed": args.split == "test",
        "evaluation_device": device.type,
        "evaluation_dtype": "float32",
        "metrics": metrics,
    }
    _atomic_json(args.output, result)
    if candidate_dump is not None:
        np.savez(args.candidate_score_output, videos=np.array(candidate_dump, dtype=object))
    print(json.dumps(metrics["checkpoints"]["anytime"]["native_event"], indent=2))


if __name__ == "__main__":
    main()
