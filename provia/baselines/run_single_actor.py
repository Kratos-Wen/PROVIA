"""Train and evaluate causal neural baselines on predicted completion streams.

Training may use labels and event times from the training partition.  Query
inference receives only cached visual features up to the current time.  The
shared frozen observer predicts query completion support; no query labels or
boundaries are consumed by any baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from provia.data.completion_stream import CompletionStream, load_completion_streams
from provia.data.folds import materialize_cross_validation_fold
from provia.data.io import read_jsonl
from provia.data.streaming_events import CORRECTNESS_INDEX
from provia.observer.completion_process import build_marked_completion_process
from provia.observer.factors import EVENT_FACTORS
from provia.training.completion_metrics import (
    _binary_metrics,
    _category_metrics,
    _decode,
    _pool_predicted_to_targets,
    _priors,
)
from provia.training.common import _sha256
from provia.model.intents import CompositionalIntentSpace

from .models import CausalTCN, MiniROAD, MistSenseRGB


METHODS = ("causal_tcn", "prego", "mistsense_rgb")
ERROR = CORRECTNESS_INDEX["error"]
CORRECT = CORRECTNESS_INDEX["correct"]
CAUSAL_TCN_RECEPTIVE_FIELD = 1 + 2 * sum((1, 2, 4, 8, 16, 32))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: object) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _action_key(stream: CompletionStream, index: int) -> tuple[int, ...]:
    return tuple(int(stream.factor_targets[name][index]) for name in EVENT_FACTORS)


@dataclass(frozen=True)
class ActionVocabulary:
    values: tuple[tuple[int, ...], ...]

    @property
    def lookup(self) -> Mapping[tuple[int, ...], int]:
        return {value: index for index, value in enumerate(self.values)}

    @classmethod
    def from_normal_streams(
        cls, streams: Sequence[CompletionStream]
    ) -> "ActionVocabulary":
        values = {
            _action_key(stream, index)
            for stream in streams
            for index, correctness in enumerate(stream.correctness.tolist())
            if int(correctness) == CORRECT
        }
        if len(values) < 2:
            raise ValueError("action baselines require at least two normal actions")
        return cls(tuple(sorted(values)))


def _derive_windows(
    streams: Sequence[CompletionStream], vocabulary: ActionVocabulary
) -> tuple[int, int, dict[str, object]]:
    """Derive the short and long context windows from action durations, without a validation search.

    The short context is the shortest class-mean positive inter-completion
    duration.  With beta=2 and theta=50%, the long context is the median span
    covering two complete preceding normal actions.
    """

    lookup = vocabulary.lookup
    durations: dict[int, list[int]] = defaultdict(list)
    two_action_spans = []
    for stream in streams:
        indices = stream.event_indices.numpy()
        correctness = stream.correctness.numpy()
        for index in range(1, len(indices)):
            key = _action_key(stream, index)
            if (
                correctness[index] == CORRECT
                and correctness[index - 1] == CORRECT
                and key in lookup
            ):
                durations[lookup[key]].append(
                    max(1, int(indices[index] - indices[index - 1]))
                )
        for index in range(2, len(indices)):
            if np.all(correctness[index - 2 : index + 1] == CORRECT):
                two_action_spans.append(
                    max(1, int(indices[index] - indices[index - 2]))
                )
    means = [float(np.mean(values)) for values in durations.values() if values]
    if not means:
        raise ValueError("window derivation needs consecutive normal events")
    short = max(2, int(math.ceil(min(means))))
    long = (
        int(math.ceil(float(np.median(two_action_spans))))
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


def _causal_window(features: Tensor, end: int, length: int) -> Tensor:
    if end < 0 or end >= len(features) or length <= 0:
        raise ValueError("invalid causal feature window")
    start = max(0, end - length + 1)
    value = features[start : end + 1].to(dtype=torch.float32)
    if len(value) < length:
        value = torch.cat((value[:1].expand(length - len(value), -1), value), dim=0)
    return value


class CompletionWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        streams: Sequence[CompletionStream],
        window: int,
        target: str,
        vocabulary: ActionVocabulary | None = None,
    ) -> None:
        if target not in {"binary", "action"}:
            raise ValueError(target)
        self.streams = tuple(streams)
        self.window = int(window)
        self.target = target
        self.samples: list[tuple[int, int, int]] = []
        lookup = {} if vocabulary is None else vocabulary.lookup
        for stream_index, stream in enumerate(self.streams):
            for event_index, correctness in enumerate(stream.correctness.tolist()):
                if target == "binary":
                    label = int(int(correctness) == ERROR)
                else:
                    if int(correctness) != CORRECT:
                        continue
                    key = _action_key(stream, event_index)
                    if key not in lookup:
                        continue
                    label = int(lookup[key])
                self.samples.append((stream_index, event_index, label))
        if not self.samples:
            raise ValueError(f"empty {target} completion dataset")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        stream_index, event_index, label = self.samples[index]
        stream = self.streams[stream_index]
        end = int(stream.event_indices[event_index])
        return (
            _causal_window(stream.global_features, end, self.window),
            torch.tensor(label, dtype=torch.long),
        )


class PrefixAnticipator:
    """Normal-training prefix trie used by the PREGO adaptation."""

    def __init__(
        self,
        streams: Sequence[CompletionStream],
        vocabulary: ActionVocabulary,
    ) -> None:
        lookup = vocabulary.lookup
        self.roots: dict[str, dict] = {}
        self.initial: dict[str, Counter[int]] = defaultdict(Counter)
        self.transitions: dict[str, dict[int, Counter[int]]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        self.run_count = 0
        for stream in streams:
            root = self.roots.setdefault(stream.task_id, {"next": Counter(), "child": {}})
            run: list[int] = []

            def commit() -> None:
                if not run:
                    return
                self.run_count += 1
                self.initial[stream.task_id][run[0]] += 1
                node = root
                for position, action in enumerate(run):
                    node["next"][action] += 1
                    if position:
                        self.transitions[stream.task_id][run[position - 1]][action] += 1
                    node = node["child"].setdefault(
                        action, {"next": Counter(), "child": {}}
                    )

            for index, correctness in enumerate(stream.correctness.tolist()):
                key = _action_key(stream, index)
                if int(correctness) == CORRECT and key in lookup:
                    run.append(int(lookup[key]))
                else:
                    commit()
                    run = []
            commit()
        if not self.run_count:
            raise ValueError("PREGO anticipation needs normal training runs")

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


def _model_and_training(
    method: str, input_dim: int, classes: int | None
) -> tuple[nn.Module, dict[str, float | int | str]]:
    if method == "causal_tcn":
        return CausalTCN(input_dim), {
            "epochs": 10,
            "learning_rate": 1e-4,
            "weight_decay": 0.05,
            
        }
    if method == "prego":
        assert classes is not None
        return MiniROAD(input_dim, classes), {
            "epochs": 10,
            "learning_rate": 1e-4,
            "weight_decay": 0.05,
            
        }
    if method == "mistsense_rgb":
        return MistSenseRGB(input_dim), {
            "epochs": 12,
            "learning_rate": 3e-5,
            "weight_decay": 5e-3,
            
        }
    raise ValueError(method)


def _loss(
    method: str,
    model: nn.Module,
    features: Tensor,
    target: Tensor,
    short_window: int,
    positive_weight: Tensor | None,
) -> Tensor:
    if method in {"causal_tcn", "mistsense_rgb"}:
        logits = model(features)
        assert positive_weight is not None
        return nn.functional.binary_cross_entropy_with_logits(
            logits, target.float(), pos_weight=positive_weight
        )
    if method == "prego":
        logits = model(features)[:, -1]
        return nn.functional.cross_entropy(logits, target)
    raise ValueError(method)


@torch.no_grad()
def _epoch_loss(
    method: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    short_window: int,
    positive_weight: Tensor | None,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    examples = 0
    for features, target in loader:
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            value = _loss(
                method, model, features, target, short_window, positive_weight
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                value.backward()
                clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        total += float(value.detach()) * len(target)
        examples += len(target)
    return total / max(examples, 1)


def _select_threshold(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if (
        target.ndim != 1
        or score.shape != target.shape
        or not len(target)
        or not np.isfinite(score).all()
        or np.any((target != 0) & (target != 1))
    ):
        raise ValueError("threshold selection requires finite aligned binary data")
    candidates = np.unique(np.r_[np.nextafter(0.0, 1.0), 0.5, score])
    order = np.argsort(score, kind="stable")
    ordered_score = score[order]
    ordered_target = target[order]
    positive_prefix = np.r_[0, np.cumsum(ordered_target, dtype=np.int64)]
    positives = int(positive_prefix[-1])
    negatives = len(target) - positives
    best_key = (-1.0, -1.0, -1.0)
    selected = 0.5
    for threshold in candidates:
        below = int(np.searchsorted(ordered_score, threshold, side="left"))
        false_negative = int(positive_prefix[below])
        true_positive = positives - false_negative
        true_negative = below - false_negative
        false_positive = negatives - true_negative
        positive_f1 = 2 * true_positive / max(
            2 * true_positive + false_positive + false_negative, 1
        )
        negative_f1 = 2 * true_negative / max(
            2 * true_negative + false_negative + false_positive, 1
        )
        recall = true_positive / positives if positives else 0.0
        key = (
            0.5 * (negative_f1 + positive_f1),
            recall,
            -abs(float(threshold) - 0.5),
        )
        if key > best_key:
            best_key = key
            selected = float(threshold)
    return selected


@torch.no_grad()
def _score_predicted(
    method: str,
    model: nn.Module,
    stream: CompletionStream,
    indices: np.ndarray,
    long_window: int,
    short_window: int,
    device: torch.device,
    batch_size: int,
    anticipator: PrefixAnticipator | None,
) -> np.ndarray:
    if not len(indices):
        return np.empty(0, dtype=np.float64)
    windows = [_causal_window(stream.global_features, int(index), long_window) for index in indices]
    batches = []
    action_predictions: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(windows), batch_size):
        value = torch.stack(windows[start : start + batch_size]).to(device)
        if method in {"causal_tcn", "mistsense_rgb"}:
            batches.append(torch.sigmoid(model(value)).cpu().numpy())
        elif method == "prego":
            action_predictions.append(
                model(value)[:, -1].argmax(dim=-1).cpu().numpy()
            )
        else:
            raise ValueError(method)
    if batches:
        return np.concatenate(batches).astype(np.float64, copy=False)
    predictions = np.concatenate(action_predictions)
    if method == "prego":
        assert anticipator is not None
        prefix: list[int] = []
        scores = []
        for current in predictions.tolist():
            expected = anticipator.expected(stream.task_id, prefix)
            scores.append(float(int(current) not in expected))
            prefix.append(int(current))
        return np.asarray(scores, dtype=np.float64)
    raise ValueError(method)


_OBSERVER_STREAMS: dict = {}


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
    threshold: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    targets = []
    scores = []
    categories: list[tuple[str, ...]] = []
    videos = []
    false_warnings = 0
    minutes = 0.0
    predicted_events = 0
    covered_targets = 0
    per_video = []
    for stream in streams:
        indices, _ = _decode(observer, _OBSERVER_STREAMS.get(stream.episode_id, stream), priors, device)
        event_score = _score_predicted(
            method,
            model,
            stream,
            indices,
            long_window,
            short_window,
            device,
            batch_size,
            anticipator,
        )
        pooled, _, assignment = _pool_predicted_to_targets(stream, indices, event_score)
        target = (stream.correctness.numpy() == ERROR).astype(np.int64)
        targets.append(target)
        scores.append(pooled)
        categories.extend(stream.error_types)
        videos.extend([stream.video_id] * len(target))
        predicted_events += len(indices)
        covered_targets += len(np.unique(assignment)) if len(assignment) else 0
        normal_prediction = target[assignment] == 0 if len(assignment) else np.empty(0, dtype=bool)
        false_warnings += int(np.sum((event_score >= threshold) & normal_prediction))
        if len(stream.timestamps_sec) > 1:
            minutes += float(stream.timestamps_sec[-1] - stream.timestamps_sec[0]) / 60.0
        per_video.append(
            {
                "video_id": stream.video_id,
                "task_id": stream.task_id,
                "target_events": len(target),
                "error_events": int(target.sum()),
                "predicted_events": len(indices),
                "covered_target_events": len(np.unique(assignment)) if len(assignment) else 0,
            }
        )
    target = np.concatenate(targets)
    score = np.concatenate(scores)
    metric = _binary_metrics(target, score, threshold)
    native_threshold = 0.5 if method == "prego" else threshold
    result: dict[str, object] = {
        "fully_predicted": metric,
        "native_operating_point": _binary_metrics(target, score, native_threshold),
        "per_error_category": _category_metrics(target, score, categories, threshold),
        "streaming_operation": {
            "false_warnings_per_minute": false_warnings / max(minutes, np.finfo(float).tiny),
            "false_warnings": false_warnings,
            "video_minutes": minutes,
        },
        "observation": {
            "target_events": len(target),
            "predicted_events": predicted_events,
            "covered_target_events": covered_targets,
            "target_coverage": covered_targets / max(len(target), 1),
        },
        "per_video": per_video,
    }
    return result, {
        "target": target,
        "score": score,
        "video_id": np.asarray(videos),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--query-feature-root", type=Path, required=True)
    parser.add_argument("--observer-checkpoint", type=Path, required=True)
    parser.add_argument("--observer-feature-root", type=Path, default=None, help="frame-aligned feature root read by the shared candidate observer (defaults to --feature-root); lets every method read a widened stream while candidates come from the observer")
    parser.add_argument("--observer-query-feature-root", type=Path, default=None, help="observer stream root for the query split (defaults to --query-feature-root)")
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
    training_episodes = tuple(item for item in episodes if item.split == "train")
    validation_episodes = tuple(item for item in episodes if item.split == "validation")
    test_episodes = tuple(item for item in episodes if item.split == "test")
    if not training_episodes or not validation_episodes or not test_episodes:
        raise ValueError("train, validation, and test partitions are required")
    intent_space = CompositionalIntentSpace.from_segments(
        segment for episode in training_episodes for segment in episode.segments
    )
    train_streams = load_completion_streams(
        training_episodes, args.feature_root, intent_space
    )
    validation_streams = load_completion_streams(
        validation_episodes, args.feature_root, intent_space
    )

    observer_payload = torch.load(
        args.observer_checkpoint, map_location="cpu", weights_only=False
    )
    observer_metadata = observer_payload.get("metadata")
    if not isinstance(observer_metadata, Mapping):
        raise ValueError("observer checkpoint lacks metadata")
    labels = {name: intent_space.labels[name] for name in EVENT_FACTORS}
    observer_root = args.observer_feature_root or args.feature_root
    observer_query_root = args.observer_query_feature_root or args.query_feature_root
    observer_streams = {}
    for episodes_, root_ in ((training_episodes, observer_root), (validation_episodes, observer_root)):
        for stream_ in (load_completion_streams(episodes_, root_, intent_space) if root_ != args.feature_root else []):
            observer_streams[stream_.episode_id] = stream_
    observer_dim = int(next(iter(observer_streams.values())).global_features.shape[-1]) if observer_streams else int(train_streams[0].global_features.shape[-1])
    input_dim = int(train_streams[0].global_features.shape[-1])
    observer = build_marked_completion_process(observer_metadata, observer_dim, labels)
    observer.load_state_dict(observer_payload["model"], strict=True)
    observer.to(device).eval()
    priors = _priors(observer_metadata)

    with open(args.query_feature_root / "manifest.json", encoding="utf-8") as handle:
        query_manifest = json.load(handle)
    test_streams = load_completion_streams(
        test_episodes, args.query_feature_root, intent_space
    )
    if observer_query_root != args.query_feature_root:
        for stream_ in load_completion_streams(test_episodes, observer_query_root, intent_space):
            observer_streams[stream_.episode_id] = stream_
    global _OBSERVER_STREAMS
    _OBSERVER_STREAMS = observer_streams

    vocabulary = ActionVocabulary.from_normal_streams(train_streams)
    short_window, long_window, window_audit = _derive_windows(
        train_streams, vocabulary
    )
    # Inputs older than this exact receptive field cannot affect the final
    # Causal-TCN logit.  Avoid evaluating identical zero-gradient prefixes on
    # datasets whose procedure-derived long window is much larger.
    model_window = (
        min(long_window, CAUSAL_TCN_RECEPTIVE_FIELD)
        if args.method == "causal_tcn"
        else long_window
    )
    target_kind = "action" if args.method == "prego" else "binary"
    train_data = CompletionWindowDataset(
        train_streams, model_window, target_kind, vocabulary
    )
    validation_data = CompletionWindowDataset(
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
        labels_binary = np.asarray([sample[2] for sample in train_data.samples])
        positives = int(labels_binary.sum())
        negatives = len(labels_binary) - positives
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
        PrefixAnticipator(train_streams, vocabulary)
        if args.method == "prego"
        else None
    )

    validation_result, validation_scores = _evaluate(
        args.method,
        model,
        observer,
        validation_streams,
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
    # Recompute validation operating-point fields at the frozen threshold.
    validation_result, validation_scores = _evaluate(
        args.method,
        model,
        observer,
        validation_streams,
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
        test_streams,
        priors,
        device,
        model_window,
        short_window,
        args.batch_size,
        anticipator,
        threshold,
    )

    parameters = sum(value.numel() for value in model.parameters())
    metadata = {
        "schema": "streaming_neural_baseline",
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
        "trainable_parameters": parameters,
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
    torch.save(
        {"model": best_state, "metadata": metadata},
        args.output_dir / "best.pt",
    )
    _atomic_json(args.output_dir / "metadata.json", metadata)
    _atomic_json(args.output_dir / "validation.json", validation_result)
    _atomic_json(args.output_dir / "test.json", test_result)
    np.savez_compressed(args.output_dir / "validation_scores.npz", **validation_scores)
    np.savez_compressed(args.output_dir / "test_scores.npz", **test_scores)
    _atomic_json(
        args.output_dir / "complete.json",
        {
            "status": "complete",
            "method": args.method,
            "test": test_result["fully_predicted"],
        },
    )
    print(json.dumps(test_result["fully_predicted"], indent=2), flush=True)


if __name__ == "__main__":
    main()
