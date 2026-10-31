"""Evaluate frozen causal neural baselines at every available query prefix.

The baseline architecture, weights, context rule, and operating-point policy
are loaded from an already completed baseline run and are never refit.  This
script only changes the readout schedule: a model that is intrinsically causal
is evaluated on every past-only frame ending inside each frozen observer
candidate, in addition to its original completion readout.
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

from provia.baselines.models import MiniROAD
from provia.baselines.run_single_actor import (
    METHODS,
    ActionVocabulary,
    PrefixAnticipator,
    _model_and_training,
)
from provia.data.completion_stream import (
    CompletionInferenceStream,
    CompletionStream,
    completion_inference_stream,
    load_completion_streams,
)
from provia.data.folds import materialize_cross_validation_fold
from provia.data.io import read_jsonl
from provia.data.streaming_events import CORRECTNESS_INDEX
from provia.evaluation.operating_points import (
    _checkpoint_metrics,
    _free_response_false_alarm_scores,
)
from provia.observer.completion_process import build_marked_completion_process
from provia.observer.factors import EVENT_FACTORS
from provia.training.completion_metrics import (
    _decode,
    _pool_predicted_to_targets,
    _priors,
)
from provia.training.common import _atomic_json, _sha256
from provia.model.intents import CompositionalIntentSpace


FROC_BUDGETS = ("0.1", "0.5", "1.0", "2.0")


def _candidate_slices(indices: np.ndarray) -> tuple[slice, ...]:
    """Return the same non-overlapping predicted lifecycles used by PROVIA."""

    value = np.asarray(indices, dtype=np.int64)
    if value.ndim != 1 or not len(value) or np.any(value < 0) or np.any(
        np.diff(value) <= 0
    ):
        raise ValueError("predicted completions must be strictly increasing")
    previous_end = 0
    result = []
    for end in value.tolist():
        start = previous_end + 1
        if start > end:
            raise ValueError("a predicted candidate has no causal prefix")
        result.append(slice(start, end + 1))
        previous_end = end
    return tuple(result)


@torch.no_grad()
def _causal_window_batches(
    feature: Tensor, window: int, batch_size: int
):
    """Yield exact left-padded causal windows without Python frame copies."""

    if feature.ndim != 2 or not len(feature) or window <= 0 or batch_size <= 0:
        raise ValueError("causal window batching received malformed features")
    padded = torch.cat((feature[:1].expand(window - 1, -1), feature), dim=0)
    for start in range(0, len(feature), batch_size):
        count = min(batch_size, len(feature) - start)
        source = padded[start : start + count + window - 1]
        # Tensor.unfold appends the window dimension after the feature
        # dimension, hence the final transpose to [batch,time,feature].
        value = source.unfold(0, window, 1).permute(0, 2, 1).contiguous()
        if value.shape != (count, window, feature.shape[-1]):
            raise RuntimeError("vectorized causal windows have the wrong shape")
        yield start, value


@torch.no_grad()
def _prego_frame_actions(
    model: MiniROAD,
    stream: CompletionInferenceStream,
    window: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    result = []
    model.eval()
    for _, value in _causal_window_batches(
        stream.global_features, window, batch_size
    ):
        value = value.to(device)
        result.append(model(value)[:, -1].argmax(dim=-1).cpu().numpy())
    return np.concatenate(result).astype(np.int64, copy=False)


@torch.no_grad()
def _prego_frame_posteriors(
    model: MiniROAD,
    stream: CompletionInferenceStream,
    window: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    result = []
    model.eval()
    for _, value in _causal_window_batches(
        stream.global_features, window, batch_size
    ):
        value = value.to(device)
        result.append(
            model(value)[:, -1].softmax(dim=-1).cpu().numpy()
        )
    return np.concatenate(result).astype(np.float64, copy=False)


@torch.no_grad()
def _frame_scores(
    method: str,
    model: nn.Module,
    stream: CompletionInferenceStream,
    indices: np.ndarray,
    slices: Sequence[slice],
    long_window: int,
    short_window: int,
    device: torch.device,
    batch_size: int,
    anticipator: PrefixAnticipator | None,
    readout: str = "native_discrete",
) -> np.ndarray:
    if readout not in {"native_discrete", "posterior_continuous"}:
        raise ValueError(f"unknown baseline readout: {readout}")
    continuous = readout == "posterior_continuous"
    if method in {"causal_tcn", "mistsense_rgb"}:
        result = []
        model.eval()
        for _, value in _causal_window_batches(
            stream.global_features, long_window, batch_size
        ):
            result.append(torch.sigmoid(model(value.to(device))).cpu().numpy())
        return np.concatenate(result).astype(np.float64, copy=False)
    if not isinstance(model, MiniROAD) or anticipator is None:
        raise ValueError("PREGO anytime evaluation needs its frozen recognizer")
    if continuous:
        # PREGO's recognition--anticipation mismatch becomes the recognized
        # posterior mass outside the anticipated successor set; committed
        # prefix actions remain the recognizer's argmax at the predicted
        # completion, exactly as in the discrete adaptation.
        posterior = _prego_frame_posteriors(
            model, stream, long_window, device, batch_size
        )
        action = posterior.argmax(axis=-1).astype(np.int64)
        score = np.zeros(len(action), dtype=np.float64)
        completed_prefix: list[int] = []
        for end, lifecycle in zip(indices.tolist(), slices):
            expected = anticipator.expected(stream.task_id, completed_prefix)
            columns = sorted(
                value for value in expected if 0 <= value < posterior.shape[1]
            )
            inside = (
                posterior[lifecycle][:, columns].sum(axis=-1)
                if columns
                else np.zeros(len(posterior[lifecycle]))
            )
            score[lifecycle] = 1.0 - inside
            completed_prefix.append(int(action[end]))
        return score
    action = _prego_frame_actions(
        model, stream, long_window, device, batch_size
    )
    score = np.zeros(len(action), dtype=np.float64)
    completed_prefix = []
    for end, lifecycle in zip(indices.tolist(), slices):
        expected = anticipator.expected(stream.task_id, completed_prefix)
        score[lifecycle] = np.asarray(
            [float(int(value) not in expected) for value in action[lifecycle]],
            dtype=np.float64,
        )
        completed_prefix.append(int(action[end]))
    return score


def _native_options(
    assignment: np.ndarray,
    candidate_options: Sequence[np.ndarray],
    native_count: int,
) -> list[np.ndarray]:
    result = []
    for native_index in range(native_count):
        candidate = np.flatnonzero(assignment == native_index)
        result.append(
            np.concatenate(
                [candidate_options[int(index)] for index in candidate], axis=0
            )
            if len(candidate)
            else np.empty((0, 2), dtype=np.float64)
        )
    return result


def _frozen_thresholds(
    artifact: Mapping[str, object] | None, checkpoint: str
) -> Mapping[str, float] | None:
    if artifact is None:
        return None
    metrics = artifact.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("frozen threshold artifact lacks metrics")
    checkpoints = metrics.get("checkpoints")
    if not isinstance(checkpoints, Mapping) or checkpoint not in checkpoints:
        raise ValueError(f"frozen threshold artifact lacks {checkpoint}")
    entry = checkpoints[checkpoint]
    if not isinstance(entry, Mapping):
        raise ValueError("frozen checkpoint entry is malformed")
    response = entry.get("free_response")
    if not isinstance(response, Mapping):
        raise ValueError("frozen checkpoint lacks a free-response curve")
    points = response.get("operating_points")
    if not isinstance(points, Mapping):
        raise ValueError("frozen checkpoint lacks operating points")
    if set(points) != set(FROC_BUDGETS):
        raise ValueError("frozen validation artifact has the wrong FROC budgets")
    thresholds = {}
    for budget in FROC_BUDGETS:
        value = points[budget]
        if not isinstance(value, Mapping) or "validation_threshold" not in value:
            raise ValueError(f"frozen validation artifact lacks {checkpoint}/{budget}")
        threshold = float(value["validation_threshold"])
        if np.isnan(threshold):
            raise ValueError("a frozen validation threshold cannot be NaN")
        thresholds[budget] = threshold
    return thresholds


def _load_frozen_validation(
    path: Path,
    *,
    method: str,
    checkpoint_sha256: str,
    observer_checkpoint_sha256: str,
    readout: str = "native_discrete",
) -> Mapping[str, object]:
    """Load operating points from the matching validation evaluation only."""

    value = json.loads(path.read_text())
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != "streaming_neural_anytime_evaluation"
        or value.get("method") != method
        or value.get("split") != "validation"
        or value.get("test_split_accessed") is not False
        or value.get("future_frames_used") is not False
        or value.get("query_boundaries_or_labels_used") is not False
        or value.get("baseline_checkpoint_sha256") != checkpoint_sha256
        or value.get("observer_checkpoint_sha256")
        != observer_checkpoint_sha256
        or value.get("readout", "native_discrete") != readout
        or value.get("frozen_validation") is not None
    ):
        raise ValueError("frozen validation artifact is incompatible with test run")
    metrics = value.get("metrics")
    checkpoints = metrics.get("checkpoints") if isinstance(metrics, Mapping) else None
    if not isinstance(checkpoints, Mapping):
        raise ValueError("frozen validation artifact lacks checkpoint metrics")
    for checkpoint in ("completion", "anytime"):
        _frozen_thresholds(value, checkpoint)
        entry = checkpoints.get(checkpoint)
        response = entry.get("free_response") if isinstance(entry, Mapping) else None
        if (
            not isinstance(response, Mapping)
            or response.get("threshold_source")
            != "selected on validation free-response curve"
        ):
            raise ValueError("frozen operating points were not selected on validation")
    return value


_OBSERVER_QUERY: dict = {}


@torch.no_grad()
def _evaluate(
    method: str,
    model: nn.Module,
    observer: nn.Module,
    streams: Sequence[CompletionStream],
    priors: Mapping[str, Tensor],
    device: torch.device,
    long_window: int,
    short_window: int,
    batch_size: int,
    anticipator: PrefixAnticipator | None,
    frozen_artifact: Mapping[str, object] | None,
    score_output: Path | None = None,
    readout: str = "native_discrete",
    candidate_dump: list | None = None,
) -> dict[str, object]:
    accumulated: dict[str, list] = {
        "completion_candidate": [],
        "anytime_candidate": [],
        "candidate_target": [],
        "origin_target": [],
        "completion_native": [],
        "anytime_native": [],
        "native_mode": [],
        "covered": [],
        "completion_false": [],
        "anytime_false": [],
        "completion_options": [],
        "anytime_options": [],
        "categories": [],
        "video_id": [],
    }
    duration_minutes = 0.0
    per_video = []
    for stream in streams:
        # No object containing native events, semantics, correctness, or error
        # categories is admitted to candidate decoding or neural scoring.
        query = completion_inference_stream(stream)
        indices, _ = _decode(observer, _OBSERVER_QUERY.get(query.episode_id, query), priors, device)
        if not len(indices):
            continue
        slices = _candidate_slices(indices)
        frame_score = _frame_scores(
            method,
            model,
            query,
            indices,
            slices,
            long_window,
            short_window,
            device,
            batch_size,
            anticipator,
            readout,
        )
        completion = np.asarray(
            [frame_score[int(index)] for index in indices], dtype=np.float64
        )
        anytime = np.asarray(
            [float(frame_score[value].max()) for value in slices],
            dtype=np.float64,
        )
        completion_native, selected, assignment = _pool_predicted_to_targets(
            stream, indices, completion
        )
        anytime_native, _, anytime_assignment = _pool_predicted_to_targets(
            stream, indices, anytime
        )
        if not np.array_equal(assignment, anytime_assignment):
            raise RuntimeError("native assignment must be score independent")
        native_mode = stream.correctness.numpy().astype(np.int64)
        covered = selected >= 0
        assigned = assignment >= 0
        candidate_target = np.zeros(len(indices), dtype=np.int64)
        candidate_target[assigned] = (
            native_mode[assignment[assigned]] == CORRECTNESS_INDEX["error"]
        )
        native_timestamp = stream.timestamps_sec.numpy()[
            stream.event_indices.numpy()
        ]
        completion_candidate_options = []
        anytime_candidate_options = []
        timestamp = query.timestamps_sec.numpy()
        for candidate_index, (end, lifecycle) in enumerate(
            zip(indices.tolist(), slices)
        ):
            native_index = int(assignment[candidate_index])
            reference = (
                float(native_timestamp[native_index])
                if native_index >= 0
                else float("nan")
            )
            completion_candidate_options.append(
                np.asarray(
                    ((completion[candidate_index], timestamp[end] - reference),),
                    dtype=np.float64,
                )
            )
            anytime_candidate_options.append(
                np.column_stack(
                    (frame_score[lifecycle], timestamp[lifecycle] - reference)
                )
            )
        accumulated["completion_candidate"].append(completion)
        accumulated["anytime_candidate"].append(anytime)
        accumulated["candidate_target"].append(candidate_target)
        accumulated["origin_target"].append(assigned.astype(np.int64))
        accumulated["completion_native"].append(completion_native)
        accumulated["anytime_native"].append(anytime_native)
        accumulated["native_mode"].append(native_mode)
        accumulated["covered"].append(covered)
        accumulated["completion_false"].append(
            _free_response_false_alarm_scores(
                completion, assignment, native_mode
            )
        )
        accumulated["anytime_false"].append(
            _free_response_false_alarm_scores(anytime, assignment, native_mode)
        )
        accumulated["completion_options"].extend(
            _native_options(
                assignment, completion_candidate_options, len(native_mode)
            )
        )
        accumulated["anytime_options"].extend(
            _native_options(
                assignment, anytime_candidate_options, len(native_mode)
            )
        )
        accumulated["categories"].extend(stream.error_types)
        accumulated["video_id"].append(
            np.repeat(stream.video_id, len(native_mode)).astype(np.str_)
        )
        if candidate_dump is not None:
            # per-candidate record for alarm-level protocol evaluation (bookkeeping only)
            assignment_array = np.asarray(assignment, dtype=np.int64)
            candidate_dump.append(
                {
                    "video_id": str(stream.video_id),
                    "endpoint_score": np.asarray(completion, dtype=np.float64),
                    "anytime_score": np.asarray(anytime, dtype=np.float64),
                    "prefix_options": [np.asarray(o, dtype=np.float64) for o in anytime_candidate_options],
                    "assignment": assignment_array,
                    "candidate_mode": np.where(assignment_array >= 0, native_mode[np.clip(assignment_array, 0, None)], -1),
                    "origin": assignment_array >= 0,
                    "native_mode": np.asarray(native_mode, dtype=np.int64),
                    "native_time": np.asarray(native_timestamp, dtype=np.float64),
                    "decoded_time": np.asarray(timestamp[np.asarray(indices, dtype=np.int64)], dtype=np.float64),
                    "native_error_types": [tuple(t) for t in stream.error_types],
                }
            )
        minutes = (
            float(stream.timestamps_sec[-1] - stream.timestamps_sec[0]) / 60.0
            if len(stream.timestamps_sec) > 1
            else 0.0
        )
        duration_minutes += minutes
        per_video.append(
            {
                "video_id": stream.video_id,
                "task_id": stream.task_id,
                "duration_minutes": minutes,
                "predicted_candidates": len(indices),
                "native_events": len(native_mode),
                "native_errors": int(
                    np.sum(native_mode == CORRECTNESS_INDEX["error"])
                ),
                "native_event_coverage": float(covered.mean()),
            }
        )
    if not accumulated["native_mode"]:
        raise ValueError("anytime baseline evaluation produced no candidates")

    def concatenate(name: str) -> np.ndarray:
        return np.concatenate(accumulated[name])

    common = {
        "candidate_target": concatenate("candidate_target"),
        "origin_score": concatenate("origin_target").astype(np.float64),
        "origin_target": concatenate("origin_target"),
        "native_mode": concatenate("native_mode"),
        "covered": concatenate("covered").astype(bool),
        "delay": np.zeros(len(concatenate("native_mode")), dtype=np.float64),
        "native_categories": accumulated["categories"],
        "duration_minutes": duration_minutes,
        "delay_reference": "native_completion",
    }
    native_mode = common["native_mode"]
    covered = common["covered"]
    anytime_native = concatenate("anytime_native")
    video_id = concatenate("video_id").astype(np.str_)
    if not (
        native_mode.shape == covered.shape == anytime_native.shape
        == video_id.shape
    ):
        raise RuntimeError("baseline score archive is not aligned")
    score_archive = None
    if score_output is not None:
        score_archive = write_native_event_score_archive(
            score_output,
            native_mode=native_mode,
            native_score=anytime_native,
            video_id=video_id,
        )
    checkpoints = {}
    for name in ("completion", "anytime"):
        result = _checkpoint_metrics(
            candidate_score=concatenate(f"{name}_candidate"),
            native_score=concatenate(f"{name}_native"),
            false_alarm_score=concatenate(f"{name}_false"),
            detection_options=accumulated[f"{name}_options"],
            frozen_operating_thresholds=_frozen_thresholds(
                frozen_artifact, name
            ),
            include_origin_metrics=False,
            **common,
        )
        checkpoints[name] = result
    result = {
        "checkpoints": checkpoints,
        "primary_streaming_checkpoint": "anytime",
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-feature-root", type=Path)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--observer-feature-root", type=Path, default=None, help="frame-aligned root read by the shared candidate observer on the query split (defaults to --feature-root)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--frozen-validation", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--score-output",
        type=Path,
        help="optional aligned native-event anytime score archive",
    )
    parser.add_argument("--candidate-score-output", type=Path, default=None, help="optional per-candidate dump for alarm-level protocol evaluation")
    parser.add_argument(
        "--readout",
        choices=("native_discrete", "posterior_continuous"),
        default="native_discrete",
        help=(
            "frozen-checkpoint score readout: the original discrete rule or "
            "the continuous posterior form of the same native rule"
        ),
    )
    args = parser.parse_args()
    candidate_dump = [] if args.candidate_score_output is not None else None
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.score_output is not None and args.score_output.exists():
        raise FileExistsError(
            f"refusing to overwrite score archive {args.score_output}"
        )
    if args.split == "test" and args.frozen_validation is None:
        parser.error("test FROC requires --frozen-validation")
    if args.split != "test" and args.frozen_validation is not None:
        parser.error("--frozen-validation is accepted only for test evaluation")

    checkpoint_path = args.run_dir / "best.pt"
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    metadata = checkpoint.get("metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("schema") != "streaming_neural_baseline"
        or metadata.get("method") not in METHODS
    ):
        raise ValueError("baseline checkpoint is not a baseline run")
    method = str(metadata["method"])
    episodes = materialize_cross_validation_fold(
        tuple(read_jsonl(args.manifest)), args.fold
    )
    training_episodes = tuple(
        value for value in episodes if value.split == "train"
    )
    query_episodes = tuple(
        value for value in episodes if value.split == args.split
    )
    if not training_episodes or not query_episodes:
        raise ValueError("manifest lacks the requested fold partitions")
    intent_space = CompositionalIntentSpace.from_segments(
        segment for episode in training_episodes for segment in episode.segments
    )
    training_feature_root = (
        args.feature_root
        if args.training_feature_root is None
        else args.training_feature_root
    )
    train = load_completion_streams(
        training_episodes, training_feature_root, intent_space
    )
    query = load_completion_streams(query_episodes, args.feature_root, intent_space)
    observer_query = {q.episode_id: q for q in (load_completion_streams(query_episodes, args.observer_feature_root, intent_space) if args.observer_feature_root and args.observer_feature_root != args.feature_root else [])}
    global _OBSERVER_QUERY
    _OBSERVER_QUERY = observer_query
    observer_input_dim = int(next(iter(observer_query.values())).global_features.shape[-1]) if observer_query else None
    input_dim = int(train[0].global_features.shape[-1])
    vocabulary = ActionVocabulary.from_normal_streams(train)
    model, _ = _model_and_training(
        method,
        input_dim,
        len(vocabulary.values) if method == "prego" else None,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval().requires_grad_(False)

    observer_path = Path(str(metadata["observer_checkpoint"]))
    if _sha256(observer_path) != metadata.get("observer_checkpoint_sha256"):
        raise ValueError("frozen observer checkpoint changed")
    observer_payload = torch.load(
        observer_path, map_location="cpu", weights_only=False
    )
    observer_metadata = observer_payload.get("metadata")
    if not isinstance(observer_metadata, Mapping):
        raise ValueError("observer checkpoint lacks metadata")
    labels = {name: intent_space.labels[name] for name in EVENT_FACTORS}
    observer = build_marked_completion_process(
        observer_metadata, (observer_input_dim or input_dim), labels
    )
    observer.load_state_dict(observer_payload["model"], strict=True)
    observer.to(device).eval().requires_grad_(False)
    priors = _priors(observer_metadata)
    anticipator = (
        PrefixAnticipator(train, vocabulary) if method == "prego" else None
    )
    checkpoint_sha256 = _sha256(checkpoint_path)
    observer_checkpoint_sha256 = _sha256(observer_path)
    frozen = (
        None
        if args.frozen_validation is None
        else _load_frozen_validation(
            args.frozen_validation,
            method=method,
            checkpoint_sha256=checkpoint_sha256,
            observer_checkpoint_sha256=observer_checkpoint_sha256,
            readout=args.readout,
        )
    )
    window_metadata = metadata.get("window_selection")
    if not isinstance(window_metadata, Mapping):
        raise ValueError("baseline checkpoint lacks its frozen context rule")
    long_window = int(
        metadata.get("model_context_frames", window_metadata["long_frames"])
    )
    short_window = int(window_metadata["short_frames"])
    metrics = _evaluate(
        method,
        model,
        observer,
        query,
        priors,
        device,
        long_window,
        short_window,
        args.batch_size,
        anticipator,
        frozen,
        args.score_output,
        args.readout,
        candidate_dump=candidate_dump,
    )
    result = {
        "schema": "streaming_neural_anytime_evaluation",
        "method": method,
        "split": args.split,
        "readout_change_only": True,
        "readout": args.readout,
        "architecture_or_weights_refit": False,
        "future_frames_used": False,
        "query_boundaries_or_labels_used": False,
        "evaluation_device": device.type,
        "evaluation_dtype": "float32",
        "baseline_checkpoint": str(checkpoint_path.resolve()),
        "baseline_checkpoint_sha256": checkpoint_sha256,
        "observer_checkpoint": str(observer_path.resolve()),
        "observer_checkpoint_sha256": observer_checkpoint_sha256,
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
        "metrics": metrics,
    }
    _atomic_json(args.output, result)
    if candidate_dump is not None:
        np.savez(args.candidate_score_output, videos=np.array(candidate_dump, dtype=object))
    print(json.dumps(metrics["checkpoints"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
