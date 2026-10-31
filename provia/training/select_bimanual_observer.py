"""Select renewal-boundary and semantic components by validation proper loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch

from provia.data.io import read_jsonl
from provia.data.role_stream import load_competitive_role_streams
from provia.observer.factors import EVENT_FACTORS
from provia.observer.marked_renewal import MarkedRenewalRoleObserver
from provia.training.common import (
    _atomic_json,
    _partitions,
    _sha256,
    _statistics,
)
from provia.training.train_bimanual_observer import (
    BOUNDARY_PREFIXES,
    SEMANTIC_PREFIXES,
    _evaluate,
)
from provia.model.intents import CompositionalIntentSpace


def _read_json(path: Path) -> Mapping[str, object]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
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
        if (
            complete.get("status") != "complete"
            or complete.get("test_split_accessed") is not False
            or run.get("anomaly_metrics_used_for_selection") is not False
            or run.get("test_split_accessed") is not False
        ):
            raise ValueError(f"{root}: candidate violates the selection boundary")
        candidates.append((root, run, validation))

    invariant = (
        "schema",
        "observer_architecture",
        "fold",
        "seed",
        "semantic_mark_statistic",
        "train_episodes",
        "validation_episodes",
        "base_rates_per_second",
        "initial_phase_probability",
        "training_mark_counts",
        "training_phase_exposure_sec",
        "event_balanced_factor_counts",
        "cache",
    )
    for key in invariant:
        if any(
            candidate[1].get(key) != candidates[0][1].get(key)
            for candidate in candidates[1:]
        ):
            raise ValueError(f"candidate runs disagree on {key}")

    learning_rates = []
    for _, run, _ in candidates:
        raw_args = run.get("args")
        if not isinstance(raw_args, Mapping):
            raise ValueError("candidate run lacks arguments")
        learning_rates.append(float(raw_args["learning_rate"]))
    if len(set(learning_rates)) != len(learning_rates):
        raise ValueError("candidate learning rates are duplicated")

    boundary_choice = min(
        candidates, key=lambda item: float(item[2]["boundary_nll"])
    )
    semantic_choice = min(
        candidates, key=lambda item: float(item[2]["mean_semantic_nll"])
    )
    boundary_checkpoint = torch.load(
        boundary_choice[0] / "best.pt", map_location="cpu", weights_only=False
    )
    semantic_checkpoint = torch.load(
        semantic_choice[0] / "best.pt", map_location="cpu", weights_only=False
    )
    combined_state = dict(boundary_checkpoint["model"])
    for name, value in semantic_checkpoint["model"].items():
        if name.startswith(SEMANTIC_PREFIXES):
            combined_state[name] = value
    if not any(name.startswith(BOUNDARY_PREFIXES) for name in combined_state):
        raise ValueError("selected checkpoint has no boundary component")
    if not any(name.startswith(SEMANTIC_PREFIXES) for name in combined_state):
        raise ValueError("selected checkpoint has no semantic component")

    base = candidates[0][1]
    raw_args = base.get("args")
    if not isinstance(raw_args, Mapping):
        raise ValueError("candidate run lacks arguments")
    manifest = Path(str(raw_args["manifest"]))
    feature_root = Path(str(raw_args["feature_root"]))
    fold = int(base["fold"])
    episodes = tuple(read_jsonl(manifest))
    train_native, validation_native = _partitions(episodes, fold)
    intent_space = CompositionalIntentSpace.from_segments(
        segment for episode in train_native for segment in episode.segments
    )
    train = load_competitive_role_streams(train_native, feature_root, intent_space)
    validation_streams = load_competitive_role_streams(
        validation_native, feature_root, intent_space
    )
    labels = {name: intent_space.labels[name] for name in EVENT_FACTORS}
    _, factor_counts = _statistics(train, labels)
    model = MarkedRenewalRoleObserver(
        int(train[0].global_features.shape[-1]),
        labels,
        torch.tensor(base["base_rates_per_second"], dtype=torch.float32),
        torch.tensor(base["initial_phase_probability"], dtype=torch.float32),
    )
    model.load_state_dict(combined_state, strict=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model.to(device).eval()
    validation = _evaluate(
        model,
        validation_streams,
        factor_counts,
        device,
    )

    selection = {
        "schema": "marked_renewal_proper_loss_selection",
        "candidate_learning_rates": sorted(learning_rates),
        "candidate_runs": [
            {
                "path": str(root),
                "checkpoint_sha256": _sha256(root / "best.pt"),
                "learning_rate": learning_rate,
                "boundary_nll": float(metrics["boundary_nll"]),
                "mean_semantic_nll": float(metrics["mean_semantic_nll"]),
            }
            for (root, _, metrics), learning_rate in zip(candidates, learning_rates)
        ],
        "boundary_component": {
            "source": str(boundary_choice[0]),
            "learning_rate": float(boundary_choice[1]["args"]["learning_rate"]),
            
        },
        "semantic_component": {
            "source": str(semantic_choice[0]),
            "learning_rate": float(semantic_choice[1]["args"]["learning_rate"]),
            
        },
        "anomaly_metrics_used": False,
        "test_split_accessed": False,
    }
    metadata = {
        **base,
        "schema": "marked_renewal_role_observer_selected",
        "selection": selection,
        "selected_boundary_learning_rate": selection["boundary_component"][
            "learning_rate"
        ],
        "selected_semantic_learning_rate": selection["semantic_component"][
            "learning_rate"
        ],
    }
    checkpoint = {
        "model": model.state_dict(),
        "metadata": metadata,
        "validation": validation,
    }
    torch.save(checkpoint, args.output_dir / "best.pt")
    _atomic_json(args.output_dir / "run.json", metadata)
    _atomic_json(args.output_dir / "intent_space.json", intent_space.to_dict())
    _atomic_json(args.output_dir / "selection.json", selection)
    _atomic_json(args.output_dir / "validation.json", validation)
    _atomic_json(
        args.output_dir / "complete.json",
        {"status": "complete", "test_split_accessed": False},
    )
    print(
        json.dumps(
            {"selection": selection, "validation": validation},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
