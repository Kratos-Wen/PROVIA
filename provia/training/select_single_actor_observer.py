"""Compose event-time and semantic components selected by validation proper loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch

from provia.data.completion_stream import load_completion_streams
from provia.data.io import read_jsonl
from provia.observer.completion_process import build_marked_completion_process
from provia.observer.factors import EVENT_FACTORS
from provia.training.train_single_actor_observer import (
    EVENT_PREFIXES,
    SEMANTIC_PREFIXES,
    _evaluate,
    _factor_counts,
)
from provia.training.common import (
    _atomic_json,
    _partitions,
    _sha256,
)
from provia.model.intents import CompositionalIntentSpace


def _read_json(path: Path) -> Mapping[str, object]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if len(args.run) < 2:
        parser.error("at least two candidate runs are required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for root in args.run:
        complete = _read_json(root / "complete.json")
        run = _read_json(root / "run.json")
        validation = _read_json(root / "validation.json")
        if complete.get("status") != "complete" or complete.get("test_split_accessed") is not False:
            raise ValueError(f"{root}: candidate is incomplete or accessed test data")
        candidates.append((root, run, validation))
    invariant = (
        "schema",
        "observer_architecture",
        "fold",
        "seed",
        "train_episodes",
        "validation_episodes",
        "base_rate_per_second",
        "training_exposure_sec",
        "event_balanced_factor_counts",
        "cache",
    )
    for key in invariant:
        if any(item[1].get(key) != candidates[0][1].get(key) for item in candidates[1:]):
            raise ValueError(f"candidate runs disagree on {key}")
    learning_rates = []
    for _, run, _ in candidates:
        raw_args = run.get("args")
        if not isinstance(raw_args, Mapping):
            raise ValueError("candidate run lacks arguments")
        event_lr = float(raw_args["event_learning_rate"])
        semantic_lr = float(raw_args["semantic_learning_rate"])
        if event_lr != semantic_lr:
            raise ValueError("search candidates must use a diagonal learning-rate design")
        learning_rates.append(event_lr)
    if len(set(learning_rates)) != len(learning_rates):
        raise ValueError("candidate learning rates are duplicated")

    event_choice = min(
        candidates, key=lambda item: float(item[2]["event_time_nll"])
    )
    semantic_choice = min(
        candidates, key=lambda item: float(item[2]["mean_semantic_nll"])
    )
    event_checkpoint = torch.load(
        event_choice[0] / "best.pt", map_location="cpu", weights_only=False
    )
    semantic_checkpoint = torch.load(
        semantic_choice[0] / "best.pt", map_location="cpu", weights_only=False
    )
    event_state = event_checkpoint["model"]
    semantic_state = semantic_checkpoint["model"]
    combined_state = dict(event_state)
    for name, value in semantic_state.items():
        if name.startswith(SEMANTIC_PREFIXES):
            combined_state[name] = value
    if not any(name.startswith(EVENT_PREFIXES) for name in combined_state):
        raise ValueError("selected checkpoint has no event-time component")

    base = candidates[0][1]
    raw_args = base.get("args")
    if not isinstance(raw_args, Mapping):
        raise ValueError("candidate run has no arguments")
    manifest = Path(str(raw_args["manifest"]))
    feature_root = Path(str(raw_args["feature_root"]))
    fold = int(base["fold"])
    episodes = tuple(read_jsonl(manifest))
    train_native, validation_native = _partitions(episodes, fold)
    intent_space = CompositionalIntentSpace.from_segments(
        segment for episode in train_native for segment in episode.segments
    )
    labels = {name: intent_space.labels[name] for name in EVENT_FACTORS}
    train = load_completion_streams(train_native, feature_root, intent_space)
    validation_streams = load_completion_streams(
        validation_native, feature_root, intent_space
    )
    factor_counts = _factor_counts(train, labels)
    model = build_marked_completion_process(
        base,
        int(train[0].global_features.shape[-1]),
        labels,
    )
    model.load_state_dict(combined_state, strict=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model.to(device).eval()
    validation = _evaluate(model, validation_streams, factor_counts, device)
    selection = {
        "schema": "completion_process_proper_loss_selection",
        "candidate_learning_rates": sorted(learning_rates),
        "candidate_runs": [
            {
                "path": str(root),
                "checkpoint_sha256": _sha256(root / "best.pt"),
                "learning_rate": learning_rate,
                "event_time_nll": float(metrics["event_time_nll"]),
                "mean_semantic_nll": float(metrics["mean_semantic_nll"]),
            }
            for (root, _, metrics), learning_rate in zip(candidates, learning_rates)
        ],
        "event_component": {
            "source": str(event_choice[0]),
            "learning_rate": float(event_choice[1]["args"]["event_learning_rate"]),  # type: ignore[index]
            
        },
        "semantic_component": {
            "source": str(semantic_choice[0]),
            "learning_rate": float(semantic_choice[1]["args"]["semantic_learning_rate"]),  # type: ignore[index]
            
        },
        "anomaly_metrics_used": False,
        "test_split_accessed": False,
    }
    metadata = {
        **base,
        "schema": "selected_completion_process",
        "selection": selection,
        "selected_event_learning_rate": selection["event_component"]["learning_rate"],  # type: ignore[index]
        "selected_semantic_learning_rate": selection["semantic_component"]["learning_rate"],  # type: ignore[index]
    }
    torch.save(
        {"model": model.state_dict(), "metadata": metadata},
        args.output_dir / "best.pt",
    )
    _atomic_json(args.output_dir / "run.json", metadata)
    _atomic_json(args.output_dir / "intent_space.json", intent_space.to_dict())
    _atomic_json(args.output_dir / "selection.json", selection)
    _atomic_json(args.output_dir / "validation.json", validation)
    _atomic_json(
        args.output_dir / "complete.json",
        {"status": "complete", "test_split_accessed": False},
    )
    print(json.dumps({"selection": selection, "validation": validation}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
