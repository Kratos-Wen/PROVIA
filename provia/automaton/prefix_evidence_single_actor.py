"""Compile per-prefix event and semantic evidence from the observer.

This compiler reads no test split and fits no parameter.  It runs the exact
frozen completion observer already used to produce candidate boundaries, but
retains its causal frame-wise event probability and semantic posterior instead
of discarding them after decoding.  These observations let the anomaly model
reason before completion without using endpoint marks in advance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from provia.observer.completion_process import build_marked_completion_process
from provia.observer.factors import EVENT_FACTORS
from provia.training.completion_metrics import _priors
from provia.training.common import (
    _atomic_json,
    _sha256,
)


@torch.no_grad()
def _compile_video(
    model: torch.nn.Module,
    priors: Mapping[str, torch.Tensor],
    feature_path: Path,
    output_path: Path,
    device: torch.device,
) -> dict[str, object]:
    with np.load(feature_path, allow_pickle=False) as payload:
        required = {
            "frame_ids",
            "timestamps_sec",
            "global_features",
            "role_features",
            "role_change_features",
        }
        if set(payload.files) != required:
            raise ValueError(f"{feature_path}: incompatible feature cache")
        frame_ids = np.asarray(payload["frame_ids"], dtype=np.int64).copy()
        timestamp = np.asarray(payload["timestamps_sec"], dtype=np.float64).copy()
        global_feature = np.asarray(
            payload["global_features"], dtype=np.float32
        ).copy()
        role_change = np.asarray(
            payload["role_change_features"], dtype=np.float32
        ).copy()
    frames = len(frame_ids)
    if (
        timestamp.shape != (frames,)
        or global_feature.ndim != 2
        or role_change.shape != (frames, 2, global_feature.shape[-1])
        or not frames
        or np.any(np.diff(frame_ids) <= 0)
        or np.any(np.diff(timestamp) <= 0)
        or not np.isfinite(global_feature).all()
        or not np.isfinite(role_change).all()
    ):
        raise ValueError(f"{feature_path}: malformed feature stream")
    global_tensor = torch.from_numpy(global_feature).to(device)
    change_tensor = torch.from_numpy(role_change).to(device)
    time_tensor = torch.from_numpy(timestamp).to(device=device, dtype=torch.float32)
    evidence = model(global_tensor, change_tensor, time_tensor)
    event_probability = model.event_probability(  # type: ignore[attr-defined]
        evidence.intensity_per_second, time_tensor
    )
    arrays: dict[str, np.ndarray] = {
        "frame_ids": frame_ids,
        "timestamps_sec": timestamp,
        "event_probability": event_probability.float().cpu().numpy(),
    }
    for name in EVENT_FACTORS:
        prior = priors[name].to(evidence.factor_logits[name])
        arrays[f"factor_{name}_probability"] = (
            evidence.factor_logits[name] + prior.log()
        ).softmax(dim=-1).to(torch.float16).cpu().numpy()
    np.savez(output_path, **arrays)
    return {
        "file": output_path.name,
        "sha256": _sha256(output_path),
        "frames": frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = torch.load(args.examples, map_location="cpu", weights_only=False)
    source = artifact.get("metadata")
    train = artifact.get("train")
    validation = artifact.get("validation")
    if (
        not isinstance(source, Mapping)
        or source.get("schema") != "completion_examples"
        or source.get("test_split_accessed") is not False
        or not isinstance(train, Sequence)
        or not isinstance(validation, Sequence)
    ):
        raise ValueError("completion examples are malformed")
    labels = {
        name: tuple(str(value) for value in source["factor_labels"][name])  # type: ignore[index]
        for name in EVENT_FACTORS
    }
    observer_path = Path(str(source["observer_checkpoint"]))
    if _sha256(observer_path) != source.get("observer_checkpoint_sha256"):
        raise ValueError("observer checkpoint changed after compilation")
    checkpoint = torch.load(observer_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("test_split_accessed") is not False
        or int(metadata.get("fold", -1)) != int(source["fold"])
    ):
        raise ValueError("observer checkpoint is malformed or fold-incompatible")
    cache = source.get("cache")
    if not isinstance(cache, Mapping):
        raise ValueError("examples lack feature-cache dimensions")
    model = build_marked_completion_process(
        metadata, int(cache["global_dim"]), labels
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    model.to(device).eval()
    priors = _priors(metadata)
    feature_root = Path(str(source["feature_root"]))
    video_ids = tuple(
        sorted(
            {
                str(sequence["video_id"])
                for sequence in (*train, *validation)
            }
        )
    )
    records = {}
    for index, video_id in enumerate(video_ids, start=1):
        feature_path = feature_root / f"{video_id}_stream_features.npz"
        output_path = args.output_dir / f"{video_id}_prefix_evidence.npz"
        records[video_id] = _compile_video(
            model, priors, feature_path, output_path, device
        )
        print(
            json.dumps(
                {"compiled": index, "total": len(video_ids), "video_id": video_id}
            ),
            flush=True,
        )
    registry = {
        "schema": "prefix_observer_evidence",
        "examples": str(args.examples.resolve()),
        "examples_sha256": _sha256(args.examples),
        "observer_checkpoint": str(observer_path.resolve()),
        "observer_checkpoint_sha256": _sha256(observer_path),
        "fold": int(source["fold"]),
        "causal": True,
        "parameters_fitted": False,
        "annotations_used_to_compute_evidence": False,
        "test_split_accessed": False,
        "videos": records,
    }
    _atomic_json(args.output_dir / "registry.json", registry)
    print(json.dumps({"output": str(args.output_dir), "videos": len(records)}))


if __name__ == "__main__":
    main()
