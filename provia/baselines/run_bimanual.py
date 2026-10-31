"""Train streaming baselines on predicted bimanual event streams.

The frozen marked-renewal observer supplies query event support and role.  A
baseline sees the full-frame feature and the feature for that predicted role,
both only up to the event end.  Query annotations, roles, boundaries, and
future frames are evaluation-only and never enter model inference.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from provia.data.folds import materialize_cross_validation_fold
from provia.data.io import read_jsonl
from provia.data.role_stream import (
    CompetitiveRoleStream,
    load_competitive_role_streams,
)
from provia.data.streaming_events import CORRECTNESS_INDEX, StreamingEventInterval
from provia.observer.factors import EVENT_FACTORS
from provia.observer.build import ROLE_EVENT_OBSERVER_ARCHITECTURES, build_role_event_observer
from provia.observer.marked_renewal import MarkedRenewalRoleObserver
from provia.training.completion_metrics import (
    _binary_metrics,
    _category_metrics,
)
from provia.training.common import _sha256
from provia.training.role_transport_fit import (
    _observer_events,
    _priors,
)
from provia.model.intents import CompositionalIntentSpace

from .run_single_actor import (
    CAUSAL_TCN_RECEPTIVE_FIELD,
    METHODS,
    _append_jsonl,
    _atomic_json,
    _epoch_loss,
    _model_and_training,
    _seed_everything,
    _select_threshold,
)


ERROR = CORRECTNESS_INDEX["error"]
CORRECT = CORRECTNESS_INDEX["correct"]


def _event_order(interval: StreamingEventInterval) -> tuple[int, int, int]:
    return interval.onset, interval.start, interval.role


def _action_key(
    stream: CompetitiveRoleStream, interval: StreamingEventInterval
) -> tuple[int, ...]:
    return (
        int(interval.role),
        *(
            int(stream.factor_targets[name][interval.onset, interval.role])
            for name in EVENT_FACTORS
        ),
    )


@dataclass(frozen=True)
class RoleActionVocabulary:
    values: tuple[tuple[int, ...], ...]

    @property
    def lookup(self) -> Mapping[tuple[int, ...], int]:
        return {value: index for index, value in enumerate(self.values)}

    @classmethod
    def from_normal_streams(
        cls, streams: Sequence[CompetitiveRoleStream]
    ) -> "RoleActionVocabulary":
        values = {
            _action_key(stream, interval)
            for stream in streams
            for interval in stream.intervals
            if int(interval.correctness) == CORRECT
        }
        if len(values) < 2:
            raise ValueError("role-action baselines require at least two normal actions")
        return cls(tuple(sorted(values)))


def _derive_role_windows(
    streams: Sequence[CompetitiveRoleStream],
    vocabulary: RoleActionVocabulary,
) -> tuple[int, int, dict[str, object]]:
    """Derive the short and long context windows from action durations on an asynchronous role stream."""

    lookup = vocabulary.lookup
    durations: dict[int, list[int]] = defaultdict(list)
    two_action_spans: list[int] = []
    for stream in streams:
        ordered = sorted(stream.intervals, key=_event_order)
        for index, interval in enumerate(ordered):
            if int(interval.correctness) != CORRECT:
                continue
            key = _action_key(stream, interval)
            if key not in lookup:
                continue
            durations[lookup[key]].append(max(1, interval.end - interval.start + 1))
            if index >= 2 and all(
                int(value.correctness) == CORRECT
                for value in ordered[index - 2 : index + 1]
            ):
                two_action_spans.append(
                    max(1, interval.end - ordered[index - 2].end)
                )
    means = [float(np.mean(values)) for values in durations.values() if values]
    if not means:
        raise ValueError("window derivation needs normal role events")
    short = max(2, int(np.ceil(min(means))))
    long = (
        int(np.ceil(float(np.median(two_action_spans))))
        if two_action_spans
        else 2 * short
    )
    long = max(short + 1, long)
    return short, long, {
        "theta": 0.5,
        "beta": 2,
        "short_frames": short,
        "long_frames": long,
        "two_action_spans": len(two_action_spans),
        "validation_metrics_used": False,
    }


def _role_window(
    stream: CompetitiveRoleStream, end: int, role: int, length: int
) -> Tensor:
    if not 0 <= role < stream.role_features.shape[1]:
        raise ValueError("invalid event role")
    if not 0 <= end < len(stream.global_features) or length <= 0:
        raise ValueError("invalid causal role window")
    start = max(0, end - length + 1)
    value = torch.cat(
        (
            stream.global_features[start : end + 1],
            stream.role_features[start : end + 1, role],
        ),
        dim=-1,
    ).to(torch.float32)
    if len(value) < length:
        value = torch.cat(
            (value[:1].expand(length - len(value), -1), value), dim=0
        )
    return value


class RoleCompletionWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        streams: Sequence[CompetitiveRoleStream],
        window: int,
        target: str,
        vocabulary: RoleActionVocabulary | None = None,
    ) -> None:
        if target not in {"binary", "action"}:
            raise ValueError(target)
        self.streams = tuple(streams)
        self.window = int(window)
        self.samples: list[tuple[int, int, int]] = []
        lookup = {} if vocabulary is None else vocabulary.lookup
        for stream_index, stream in enumerate(self.streams):
            for interval_index, interval in enumerate(stream.intervals):
                if target == "binary":
                    label = int(int(interval.correctness) == ERROR)
                else:
                    if int(interval.correctness) != CORRECT:
                        continue
                    key = _action_key(stream, interval)
                    if key not in lookup:
                        continue
                    label = int(lookup[key])
                self.samples.append((stream_index, interval_index, label))
        if not self.samples:
            raise ValueError(f"empty {target} role-event dataset")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        stream_index, interval_index, label = self.samples[index]
        stream = self.streams[stream_index]
        interval = stream.intervals[interval_index]
        return (
            _role_window(stream, interval.end, interval.role, self.window),
            torch.tensor(label, dtype=torch.long),
        )


class RolePrefixAnticipator:
    """Training-normal role-action trie for the PREGO adaptation."""

    def __init__(
        self,
        streams: Sequence[CompetitiveRoleStream],
        tasks: Mapping[str, str],
        vocabulary: RoleActionVocabulary,
    ) -> None:
        lookup = vocabulary.lookup
        self.roots: dict[str, dict] = {}
        self.initial: dict[str, Counter[int]] = defaultdict(Counter)
        self.transitions: dict[str, dict[int, Counter[int]]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        self.run_count = 0
        for stream in streams:
            task = tasks[stream.video_id]
            root = self.roots.setdefault(task, {"next": Counter(), "child": {}})
            run: list[int] = []

            def commit() -> None:
                if not run:
                    return
                self.run_count += 1
                self.initial[task][run[0]] += 1
                node = root
                for position, action in enumerate(run):
                    node["next"][action] += 1
                    if position:
                        self.transitions[task][run[position - 1]][action] += 1
                    node = node["child"].setdefault(
                        action, {"next": Counter(), "child": {}}
                    )

            for interval in sorted(stream.intervals, key=_event_order):
                key = _action_key(stream, interval)
                if int(interval.correctness) == CORRECT and key in lookup:
                    run.append(int(lookup[key]))
                else:
                    commit()
                    run = []
            commit()
        if not self.run_count:
            raise ValueError("PREGO anticipation needs normal role-action runs")

    def expected(self, task: str, prefix: Sequence[int]) -> set[int]:
        root = self.roots.get(task)
        if root is None:
            return set()
        node = root
        matched = True
        for action in prefix:
            child = node["child"].get(int(action))
            if child is None:
                matched = False
                break
            node = child
        if matched and node["next"]:
            return set(node["next"])
        if prefix:
            transition = self.transitions[task].get(int(prefix[-1]), Counter())
            if transition:
                return set(transition)
        return set(self.initial[task])


@torch.no_grad()
def _score_events(
    method: str,
    model: nn.Module,
    stream: CompetitiveRoleStream,
    events: Sequence[object],
    task: str,
    model_window: int,
    short_window: int,
    device: torch.device,
    batch_size: int,
    anticipator: RolePrefixAnticipator | None,
) -> np.ndarray:
    if not events:
        return np.empty(0, dtype=np.float64)
    windows = [
        _role_window(
            stream,
            int(event.end_index),
            int(event.role),
            model_window,
        )
        for event in events
    ]
    scores: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(windows), batch_size):
        value = torch.stack(windows[start : start + batch_size]).to(device)
        if method in {"causal_tcn", "mistsense_rgb"}:
            scores.append(torch.sigmoid(model(value)).cpu().numpy())
        elif method == "prego":
            predictions.append(model(value)[:, -1].argmax(dim=-1).cpu().numpy())
        else:
            raise ValueError(method)
    if scores:
        return np.concatenate(scores).astype(np.float64, copy=False)
    action = np.concatenate(predictions)
    if method == "prego":
        assert anticipator is not None
        prefix: list[int] = []
        result = []
        for current in action.tolist():
            expected = anticipator.expected(task, prefix)
            result.append(float(int(current) not in expected))
            prefix.append(int(current))
        return np.asarray(result, dtype=np.float64)
    raise ValueError(method)


def _frame_categories(stream: CompetitiveRoleStream) -> tuple[tuple[str, ...], ...]:
    values = [set() for _ in range(len(stream.frame_ids))]
    for interval in stream.intervals:
        if int(interval.correctness) == ERROR:
            for index in range(interval.start, interval.end + 1):
                values[index].update(interval.error_types)
    return tuple(tuple(sorted(value)) for value in values)


def _rasterize_event_scores(
    length: int, events: Sequence[object], scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Use the same predicted onset--end support convention as PROVIA."""

    if scores.shape != (len(events),):
        raise ValueError("event scores and decoded events must align")
    raster = np.zeros(length, dtype=np.float64)
    support = np.zeros(length, dtype=bool)
    for event, score in zip(events, scores):
        onset = int(event.onset_index)
        end = int(event.end_index) + 1
        if not 0 <= onset < end <= length:
            raise ValueError("decoded event support falls outside the stream")
        raster[onset:end] = np.maximum(raster[onset:end], float(score))
        support[onset:end] = True
    return raster, support


def _recovery_evaluation_mask(stream: CompetitiveRoleStream) -> np.ndarray:
    mask = np.zeros(len(stream.frame_ids), dtype=bool)
    for interval in stream.intervals:
        if int(interval.correctness) == CORRECTNESS_INDEX["correction"]:
            mask[interval.onset : interval.end + 1] = True
    return mask


@torch.no_grad()
def _evaluate(
    method: str,
    model: nn.Module,
    observer: MarkedRenewalRoleObserver,
    observer_metadata: Mapping[str, object],
    streams: Sequence[CompetitiveRoleStream],
    tasks: Mapping[str, str],
    labels: Mapping[str, Sequence[str]],
    priors: Mapping[str, Tensor],
    device: torch.device,
    model_window: int,
    short_window: int,
    batch_size: int,
    anticipator: RolePrefixAnticipator | None,
    threshold: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    all_target: list[np.ndarray] = []
    all_score: list[np.ndarray] = []
    all_categories: list[tuple[str, ...]] = []
    all_videos: list[str] = []
    recovery_false = 0
    recovery_total = 0
    predicted_events = 0
    false_warnings = 0
    video_minutes = 0.0
    covered_frames = 0
    anomaly_covered = 0
    anomaly_frames = 0
    per_video = []
    for stream in streams:
        _, decoded = _observer_events(
            observer, observer_metadata, stream, priors, labels, device
        )
        events = tuple(
            sorted(decoded, key=lambda value: (value.end_index, value.role))
        )
        event_score = _score_events(
            method,
            model,
            stream,
            events,
            tasks[stream.video_id],
            model_window,
            short_window,
            device,
            batch_size,
            anticipator,
        )
        frame_score, support = _rasterize_event_scores(
            len(stream.frame_ids), events, event_score
        )
        target = stream.anomaly_target.numpy().astype(np.int64)
        for event, score in zip(events, event_score):
            start = int(event.onset_index)
            end = int(event.end_index) + 1
            if score >= threshold and not np.any(target[start:end]):
                false_warnings += 1
        recovery = _recovery_evaluation_mask(stream)
        recovery_false += int(np.sum((frame_score >= threshold) & recovery))
        recovery_total += int(recovery.sum())
        predicted_events += len(events)
        covered_frames += int(support.sum())
        anomaly_covered += int(np.sum(support & target.astype(bool)))
        anomaly_frames += int(target.sum())
        all_target.append(target)
        all_score.append(frame_score)
        all_categories.extend(_frame_categories(stream))
        all_videos.extend([stream.video_id] * len(target))
        duration = (
            float(stream.timestamps_sec[-1] - stream.timestamps_sec[0]) / 60.0
            if len(stream.timestamps_sec) > 1
            else 0.0
        )
        video_minutes += duration
        per_video.append(
            {
                "video_id": stream.video_id,
                "frames": len(target),
                "anomaly_frames": int(target.sum()),
                "predicted_events": len(events),
                "support_coverage": float(support.mean()),
            }
        )
    target = np.concatenate(all_target)
    score = np.concatenate(all_score)
    metric = _binary_metrics(target, score, threshold)
    native_threshold = 0.5 if method == "prego" else threshold
    result: dict[str, object] = {
        "fully_predicted": metric,
        "strict_frame": metric,
        "native_operating_point": _binary_metrics(target, score, native_threshold),
        "per_error_category": _category_metrics(
            target, score, all_categories, threshold
        ),
        "recovery_false_positive_rate": (
            recovery_false / max(recovery_total, 1)
        ),
        "streaming_operation": {
            "false_warnings_per_minute": false_warnings
            / max(video_minutes, np.finfo(float).tiny),
            "false_warning_events": false_warnings,
            "video_minutes": video_minutes,
        },
        "observation": {
            "frames": len(target),
            "predicted_events": predicted_events,
            "predicted_support_coverage": covered_frames / max(len(target), 1),
            "anomaly_frame_coverage": anomaly_covered / max(anomaly_frames, 1),
        },
        "per_video": per_video,
    }
    return result, {
        "target": target,
        "score": score,
        "video_id": np.asarray(all_videos),
    }


def _validate_cache_pair(
    development: Mapping[str, object], query: Mapping[str, object]
) -> None:
    invariant = ("backbone", "sample_rate_hz", "global_dim", "role_dim", "roles")
    if (
        query.get("contains_annotations") is not False
        or query.get("partitions") != ["test"]
        or any(development.get(name) != query.get(name) for name in invariant)
    ):
        raise ValueError("development and role-stream frontends disagree")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--query-feature-root", type=Path, required=True)
    parser.add_argument("--observer-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    _seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    episodes = materialize_cross_validation_fold(
        tuple(read_jsonl(args.manifest)), args.fold
    )
    training_episodes = tuple(value for value in episodes if value.split == "train")
    validation_episodes = tuple(
        value for value in episodes if value.split == "validation"
    )
    test_episodes = tuple(value for value in episodes if value.split == "test")
    if not training_episodes or not validation_episodes or not test_episodes:
        raise ValueError("train, validation, and test partitions are required")
    tasks = {value.video_id: value.task_id for value in episodes}
    intent_space = CompositionalIntentSpace.from_segments(
        segment for episode in training_episodes for segment in episode.segments
    )
    labels = {name: intent_space.labels[name] for name in EVENT_FACTORS}
    train_streams = load_competitive_role_streams(
        training_episodes, args.feature_root, intent_space
    )
    validation_streams = load_competitive_role_streams(
        validation_episodes, args.feature_root, intent_space
    )

    payload = torch.load(
        args.observer_checkpoint, map_location="cpu", weights_only=False
    )
    observer_metadata = payload.get("metadata")
    if (
        not isinstance(observer_metadata, Mapping)
        or observer_metadata.get("observer_architecture")
        not in ROLE_EVENT_OBSERVER_ARCHITECTURES
    ):
        raise ValueError("role runner requires a role event observer")
    observer = build_role_event_observer(
        observer_metadata, int(train_streams[0].global_features.shape[-1]), labels
    )
    observer.load_state_dict(payload["model"], strict=True)
    observer.to(device).eval()
    priors = _priors(observer_metadata)

    with open(args.feature_root / "manifest.json", encoding="utf-8") as handle:
        development_manifest = json.load(handle)
    with open(args.query_feature_root / "manifest.json", encoding="utf-8") as handle:
        query_manifest = json.load(handle)
    _validate_cache_pair(development_manifest, query_manifest)
    test_streams = load_competitive_role_streams(
        test_episodes, args.query_feature_root, intent_space
    )

    vocabulary = RoleActionVocabulary.from_normal_streams(train_streams)
    short_window, long_window, window_audit = _derive_role_windows(
        train_streams, vocabulary
    )
    model_window = (
        min(long_window, CAUSAL_TCN_RECEPTIVE_FIELD)
        if args.method == "causal_tcn"
        else long_window
    )
    target_kind = "action" if args.method == "prego" else "binary"
    train_data = RoleCompletionWindowDataset(
        train_streams, model_window, target_kind, vocabulary
    )
    validation_data = RoleCompletionWindowDataset(
        validation_streams, model_window, target_kind, vocabulary
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    input_dim = 2 * int(train_streams[0].global_features.shape[-1])
    model, recipe = _model_and_training(
        args.method,
        input_dim,
        len(vocabulary.values) if target_kind == "action" else None,
    )
    model.to(device)
    training_model = (
        torch.compile(model, mode="reduce-overhead")
        if args.method == "causal_tcn" and device.type == "cuda"
        else model
    )
    positive_weight = None
    if target_kind == "binary":
        binary = np.asarray([sample[2] for sample in train_data.samples])
        positives = int(binary.sum())
        negatives = len(binary) - positives
        if not positives or not negatives:
            raise ValueError("binary training requires both classes")
        positive_weight = torch.tensor(
            negatives / positives, dtype=torch.float32, device=device
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(recipe["learning_rate"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    best_loss = float("inf")
    best_state = None
    started = time.perf_counter()
    for epoch in range(1, int(recipe["epochs"]) + 1):
        train_loss = _epoch_loss(
            args.method,
            training_model,
            train_loader,
            device,
            short_window,
            positive_weight,
            optimizer,
        )
        validation_loss = _epoch_loss(
            args.method,
            training_model,
            validation_loader,
            device,
            short_window,
            positive_weight,
            None,
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _append_jsonl(args.output_dir / "training.jsonl", record)
        print(json.dumps(record), flush=True)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.to(device).eval()
    anticipator = (
        RolePrefixAnticipator(train_streams, tasks, vocabulary)
        if args.method == "prego"
        else None
    )

    validation_result, validation_scores = _evaluate(
        args.method,
        model,
        observer,
        observer_metadata,
        validation_streams,
        tasks,
        labels,
        priors,
        device,
        model_window,
        short_window,
        args.batch_size,
        anticipator,
        0.5,
    )
    if args.method == "prego":
        threshold = 0.5
        threshold_source = "PREGO native binary recognition--anticipation mismatch"
    else:
        threshold = _select_threshold(
            validation_scores["target"], validation_scores["score"]
        )
        threshold_source = "validation maximum mean-class F1"
    validation_result, validation_scores = _evaluate(
        args.method,
        model,
        observer,
        observer_metadata,
        validation_streams,
        tasks,
        labels,
        priors,
        device,
        model_window,
        short_window,
        args.batch_size,
        anticipator,
        threshold,
    )
    test_result, test_scores = _evaluate(
        args.method,
        model,
        observer,
        observer_metadata,
        test_streams,
        tasks,
        labels,
        priors,
        device,
        model_window,
        short_window,
        args.batch_size,
        anticipator,
        threshold,
    )

    metadata = {
        "schema": "streaming_role_baseline",
        "method": args.method,
        "query_annotations_or_boundaries_used": False,
        "future_frames_used": False,
        "seed": args.seed,
        "fold": args.fold,
        "recipe": recipe,
        "window_selection": window_audit,
        "model_context_frames": model_window,
        "train_examples": len(train_data),
        "validation_examples": len(validation_data),
        "action_classes": len(vocabulary.values),
        "trainable_parameters": sum(value.numel() for value in model.parameters()),
        "best_validation_loss": best_loss,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "observer_checkpoint": str(args.observer_checkpoint.resolve()),
        "observer_checkpoint_sha256": _sha256(args.observer_checkpoint),
        "test_labels_used_for_fitting_or_selection": False,
        "source_test_split_previously_accessed": bool(
            query_manifest.get("source_test_split_accessed", False)
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    torch.save({"model": best_state, "metadata": metadata}, args.output_dir / "best.pt")
    _atomic_json(args.output_dir / "metadata.json", metadata)
    _atomic_json(args.output_dir / "validation.json", validation_result)
    _atomic_json(args.output_dir / "test.json", test_result)
    np.savez_compressed(args.output_dir / "validation_scores.npz", **validation_scores)
    np.savez_compressed(args.output_dir / "test_scores.npz", **test_scores)
    _atomic_json(
        args.output_dir / "complete.json",
        {"status": "complete", "method": args.method, "test": test_result["fully_predicted"]},
    )
    print(json.dumps(test_result["fully_predicted"], indent=2), flush=True)


if __name__ == "__main__":
    main()
