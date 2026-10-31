"""Compile frame-wise causal evidence from the bimanual step observer.

The event candidates used downstream are emitted by the same posterior-median
renewal decoder.  This compiler retains the rest of the observer filtration:
role phase posteriors and soft semantic observations at every frame.  It reads no test split, fits no parameter, and uses no anomaly
annotation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from provia.automaton.example_io import _episodes_and_intents, _load_source
from provia.data.role_stream import load_competitive_role_streams
from provia.observer.factors import EVENT_FACTORS, EVENT_PHASES
from provia.observer.build import ROLE_EVENT_OBSERVER_ARCHITECTURES, build_role_event_observer
from provia.observer.marked_renewal import MarkedRenewalRoleObserver
from provia.training.common import _atomic_json, _sha256
from provia.training.role_transport_fit import _priors


INTERACTION = EVENT_PHASES.index("interaction")


@torch.no_grad()
def _compile_stream(
    observer: MarkedRenewalRoleObserver,
    observer_metadata: Mapping[str, object],
    priors: Mapping[str, torch.Tensor],
    stream: object,
    output: Path,
    device: torch.device,
) -> dict[str, object]:
    role_change = stream.role_change_features
    if observer_metadata.get("role_change_evidence") is False:
        role_change = None
    elif observer_metadata.get("role_change_evidence") is True and role_change is None:
        raise ValueError("observer checkpoint requires role-change evidence")
    evidence, posterior = observer(
        stream.global_features.to(device),
        stream.role_features.to(device),
        stream.timestamps_sec.to(device=device, dtype=torch.float32),
        None if role_change is None else role_change.to(device),
    )
    phase = posterior.state_probability.float().cpu()
    frames = len(stream.frame_ids)
    if phase.shape != (frames, 2, len(EVENT_PHASES)):
        raise RuntimeError("role phase posterior has an incompatible shape")
    arrays: dict[str, np.ndarray] = {
        "frame_ids": stream.frame_ids.numpy().astype(np.int64, copy=True),
        "timestamps_sec": stream.timestamps_sec.numpy().astype(
            np.float64, copy=True
        ),
        "phase_probability": phase.to(torch.float16).numpy(),
        "event_probability": phase[:, :, INTERACTION].numpy(),
    }
    for name in EVENT_FACTORS:
        prior = priors[name].to(evidence.factor_logits[name])
        probability = (
            evidence.factor_logits[name] + prior.log()[None, None, :]
        ).softmax(dim=-1)
        arrays[f"factor_{name}_probability"] = (
            probability.to(torch.float16).cpu().numpy()
        )
    temporary = output.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(output)
    return {
        "file": output.name,
        "sha256": _sha256(output),
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
    artifact, source = _load_source(args.examples)
    train = artifact.get("train")
    validation = artifact.get("validation")
    if (
        source.get("schema") != "role_windowed_examples"
        or source.get("test_split_accessed") is not False
        or not isinstance(train, Sequence)
        or not isinstance(validation, Sequence)
    ):
        raise ValueError("role examples are malformed")
    train_episode, validation_episode, intent = _episodes_and_intents(source)
    feature_root = Path(str(source["feature_root"]))
    streams = (
        *load_competitive_role_streams(train_episode, feature_root, intent),
        *load_competitive_role_streams(validation_episode, feature_root, intent),
    )
    by_video = {str(stream.video_id): stream for stream in streams}
    required_video = {
        str(sequence["video_id"]) for sequence in (*train, *validation)
    }
    if set(by_video) != required_video:
        raise ValueError("role streams and examples do not agree")

    checkpoint_path = Path(str(source["observer_checkpoint"]))
    if _sha256(checkpoint_path) != source.get("observer_checkpoint_sha256"):
        raise ValueError("observer checkpoint changed after compilation")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    metadata = checkpoint.get("metadata")
    labels = {
        name: tuple(str(value) for value in source["factor_labels"][name])
        for name in EVENT_FACTORS
    }
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("observer_architecture")
        not in ROLE_EVENT_OBSERVER_ARCHITECTURES
        or metadata.get("test_split_accessed") is not False
        or int(metadata.get("fold", -1)) != int(source["fold"])
    ):
        raise ValueError("observer checkpoint is malformed or fold-incompatible")
    observer = build_role_event_observer(metadata, int(source["input_dim"]), labels)
    observer.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    observer.to(device).eval()
    priors = _priors(metadata)
    records = {}
    for index, video_id in enumerate(sorted(required_video), start=1):
        output = args.output_dir / f"video_{index:04d}.npz"
        records[video_id] = _compile_stream(
            observer,
            metadata,
            priors,
            by_video[video_id],
            output,
            device,
        )
        print(
            json.dumps(
                {"compiled": index, "total": len(required_video), "video_id": video_id}
            ),
            flush=True,
        )
    registry = {
        "schema": "role_prefix_observer_evidence",
        "examples": str(args.examples.resolve()),
        "examples_sha256": _sha256(args.examples),
        "observer_checkpoint": str(checkpoint_path.resolve()),
        "observer_checkpoint_sha256": _sha256(checkpoint_path),
        "fold": int(source["fold"]),
        "causal": True,
        "parameters_fitted": False,
        "annotations_used_to_compute_evidence": False,
        "event_probability": "posterior interaction-phase probability",
        "test_split_accessed": False,
        "videos": records,
    }
    _atomic_json(args.output_dir / "registry.json", registry)
    print(json.dumps({"output": str(args.output_dir), "videos": len(records)}))


if __name__ == "__main__":
    main()
