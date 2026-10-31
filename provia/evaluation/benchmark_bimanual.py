"""Benchmark constant-memory bimanual streaming inference.

The reasoner workload follows the frozen observer's asynchronous left/right
candidate lifecycles one video frame at a time.  No candidate prefix is
replayed.  Optional RGB timing adds the frozen backbone recorded by the stream
provenance, competitive role pooling, marked-renewal observer, and causal
posterior-median decoder used to construct the evaluation streams.
Video decoding and camera capture remain outside the timed region.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from provia.evaluation.benchmark_common import (
    _atomic_json,
    _parameter_count,
    _time_calls,
)
from provia.observer.factors import EVENT_FACTORS, EVENT_ROLES
from provia.evaluation.evaluate_bimanual import _checkpoint_relative
from provia.training.common import _intact, _sha256
from provia.training.role_inputs import (
    _load_expert_effect,
    _load_transport,
)
from provia.training.train_bimanual import (
    _build_model,
    _source_with_predictive_role_language_registry,
)
from provia.training.role_examples import (
    _load_prefix_registry,
    _prefix_evidence,
    _role_stream,
    _validate_sequence,
)
from provia.model.step_effects import ExpertEffectPosterior
from provia.model.interval_transport import ContinuationTransport
from provia.model.bimanual import (
    PROVIA_BIMANUAL_ARCHITECTURE,
    RoleIntentStateModel,
    RoleCoupledIntentStreamState,
)


@dataclass(frozen=True)
class _RoleFrameRecord:
    reset: bool
    started: tuple[bool, bool]
    completed: tuple[bool, bool]
    feature: Tensor
    event_probability: Tensor
    factor_probability: Mapping[str, Tensor]
    effect_statistic: Mapping[int, Tensor]

    def to(self, device: torch.device) -> "_RoleFrameRecord":
        return _RoleFrameRecord(
            reset=self.reset,
            started=self.started,
            completed=self.completed,
            feature=self.feature.to(device),
            event_probability=self.event_probability.to(device),
            factor_probability={
                name: value.to(device)
                for name, value in self.factor_probability.items()
            },
            effect_statistic={
                role: value.to(device)
                for role, value in self.effect_statistic.items()
            },
        )


def _load(
    examples_path: Path,
    checkpoint_path: Path,
    split: str,
    device: torch.device,
    rebuilt_prefix: Path | None,
) -> tuple[
    RoleIntentStateModel,
    Mapping[str, object],
    Sequence[Mapping[str, object]],
    ContinuationTransport,
    ExpertEffectPosterior,
    Path,
    Mapping[str, object],
    Mapping[str, object],
]:
    artifact = torch.load(examples_path, map_location="cpu", weights_only=False)
    source = artifact.get("metadata")
    sequences = artifact.get(split)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    metadata = checkpoint.get("metadata")
    if (
        not isinstance(source, Mapping)
        or source.get("schema") != "role_windowed_examples"
        or source.get("test_split_accessed") is not False
        or not isinstance(sequences, Sequence)
        or not sequences
        or not isinstance(metadata, Mapping)
        or metadata.get("architecture") != PROVIA_BIMANUAL_ARCHITECTURE
        or (rebuilt_prefix is None and metadata.get("examples_sha256") != _sha256(examples_path))
        or metadata.get("test_split_accessed") is not False
    ):
        raise ValueError("bimanual streaming benchmark artifacts are incompatible")
    registry_path = _checkpoint_relative(checkpoint_path, metadata["language_registry"])
    if not _intact(registry_path, metadata.get("language_registry_sha256")):
        raise ValueError("predictive role-language registry changed after training")
    source, _ = _source_with_predictive_role_language_registry(
        source, registry_path, examples_path, rebuilt_examples=rebuilt_prefix is not None
    )
    model = _build_model(
        source,
        device,
        )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
    transport = _load_transport(
        source, device, posterior_predictive_innovation=True
    )
    effect_path = _checkpoint_relative(checkpoint_path, metadata["expert_effect_artifact"])
    if not _intact(effect_path, metadata.get("expert_effect_artifact_sha256")):
        raise ValueError("expert-effect posterior changed after training")
    effect = _load_expert_effect(source, effect_path, transport, device, rebuilt_examples=rebuilt_prefix is not None)
    prefix_root = rebuilt_prefix or Path(str(metadata["prefix_evidence"]))
    if rebuilt_prefix is None and _sha256(prefix_root / "registry.json") != metadata.get(
        "prefix_evidence_registry_sha256"
    ):
        raise ValueError("role prefix evidence changed after training")
    prefix_registry = _load_prefix_registry(prefix_root, examples_path)
    return (
        model,
        source,
        sequences,
        transport,
        effect,
        prefix_root,
        prefix_registry,
        metadata,
    )


def _workload(
    sequences: Sequence[Mapping[str, object]],
    feature_root: Path,
    prefix_root: Path,
    prefix_registry: Mapping[str, object],
    labels: Mapping[str, Sequence[str]],
    target_frames: int,
) -> tuple[list[_RoleFrameRecord], dict[str, float | int]]:
    if target_frames <= 0:
        raise ValueError("target frame count must be positive")
    records: list[_RoleFrameRecord] = []
    quota = max(1, int(np.ceil(target_frames / len(sequences))))
    validation_frames = 0
    decoded_candidates = 0
    active_role_updates = 0
    active_video_frames = 0
    timed_candidates = 0
    timed_active_role_updates = 0
    for sequence in sequences:
        frame, timestamp, global_feature, role_feature = _role_stream(
            feature_root, str(sequence["video_id"])
        )
        candidates = _validate_sequence(sequence, len(frame), labels)
        event, factors = _prefix_evidence(
            prefix_root,
            prefix_registry,
            str(sequence["video_id"]),
            frame,
            timestamp,
            labels,
        )
        validation_frames += len(frame)
        decoded_candidates += len(candidates)
        active_mask = torch.zeros(
            len(frame), len(EVENT_ROLES), dtype=torch.bool
        )
        for candidate in candidates:
            active_mask[candidate.start : candidate.end + 1, candidate.role] = True
        active_role_updates += int(active_mask.sum())
        active_video_frames += int(active_mask.any(dim=1).sum())

        if len(records) >= target_frames:
            continue
        limit = min(len(frame), quota)
        first_completion = min(candidate.end for candidate in candidates) + 1
        limit = min(len(frame), max(limit, first_completion))
        starts: dict[int, dict[int, object]] = {}
        ends: dict[int, dict[int, object]] = {}
        for candidate in candidates:
            if candidate.role in starts.setdefault(candidate.start, {}):
                raise ValueError("a role has two candidates starting together")
            if candidate.role in ends.setdefault(candidate.end, {}):
                raise ValueError("a role has two candidates completing together")
            starts[candidate.start][candidate.role] = candidate
            ends[candidate.end][candidate.role] = candidate
        for index in range(limit):
            starting = starts.get(index, {})
            ending = ends.get(index, {})
            pair = global_feature[index].unsqueeze(0).expand(
                len(EVENT_ROLES), -1
            )
            records.append(
                _RoleFrameRecord(
                    reset=index == 0,
                    started=tuple(
                        role in starting for role in range(len(EVENT_ROLES))
                    ),
                    completed=tuple(
                        role in ending for role in range(len(EVENT_ROLES))
                    ),
                    feature=torch.cat((pair, role_feature[index]), dim=-1).float(),
                    event_probability=event[index].float(),
                    factor_probability={
                        name: value[index].float()
                        for name, value in factors.items()
                    },
                    effect_statistic={
                        role: candidate.effect_statistic.float()
                        for role, candidate in ending.items()
                    },
                )
            )
            timed_active_role_updates += int(active_mask[index].sum())
            timed_candidates += len(ending)
        if len(records) >= target_frames:
            continue
    if not records or not timed_candidates:
        raise ValueError("timed bimanual workload contains no completed candidate")
    return records, {
        "validation_videos": len(sequences),
        "validation_frames": validation_frames,
        "decoded_candidates": decoded_candidates,
        "active_role_updates": active_role_updates,
        "active_video_frames": active_video_frames,
        "active_role_updates_per_frame": active_role_updates / validation_frames,
        "active_video_frame_fraction": active_video_frames / validation_frames,
        "frames_per_predicted_candidate": validation_frames / decoded_candidates,
        "timed_workload_frames": len(records),
        "timed_workload_completions": timed_candidates,
        "timed_active_role_updates": timed_active_role_updates,
        "timed_active_role_updates_per_frame": (
            timed_active_role_updates / len(records)
        ),
    }


@torch.inference_mode()
def _benchmark_reasoner(
    model: RoleIntentStateModel,
    records: Sequence[_RoleFrameRecord],
    transport: ContinuationTransport,
    effect: ExpertEffectPosterior,
    device: torch.device,
    *,
    resident: bool,
    iterations: int,
    warmup: int,
):
    rows = [value.to(device) for value in records] if resident else list(records)
    keys = tuple(
        model.energy.build_frontier_key(transport, role)
        for role in range(len(EVENT_ROLES))
    )
    state: RoleCoupledIntentStreamState | None = None

    def update(index: int) -> float:
        nonlocal state
        row = rows[index % len(rows)]
        if not resident:
            row = row.to(device)
        if row.reset or index % len(rows) == 0:
            state = model.initial_stream_state(row.feature, transport)
        if state is None:
            raise RuntimeError("bimanual stream state was not initialized")
        step = model.forward_stream_frame(
            row.feature,
            row.factor_probability,
            row.event_probability,
            state,
            transport,
            started=row.started,
            completed=row.completed,
            effect_statistic=row.effect_statistic,
            expert_effect=(effect if row.effect_statistic else None),
            frontier_key=keys,
        )
        state = step.state
        return sum(
            0.0 if value is None else float(value.detach().cpu())
            for value in step.anomaly_probability
        )

    return _time_calls(
        update, iterations=iterations, warmup=warmup, device=device
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument(
        "--prefix-evidence",
        type=Path,
        help="prefix evidence compiled for --examples, for a released checkpoint",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workload-frames", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    return parser


def main() -> None:
    parser = _argument_parser()
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if min(args.workload_frames, args.iterations) <= 0 or args.warmup < 0:
        parser.error("benchmark counts must be positive")
    device = torch.device(args.device)
    (
        model,
        source,
        sequences,
        transport,
        effect,
        prefix_root,
        prefix_registry,
        metadata,
    ) = _load(args.examples, args.checkpoint, args.split, device, args.prefix_evidence)
    labels = {
        name: tuple(str(value) for value in source["factor_labels"][name])
        for name in EVENT_FACTORS
    }
    rows, workload = _workload(
        sequences,
        Path(str(source["feature_root"])),
        prefix_root,
        prefix_registry,
        labels,
        args.workload_frames,
    )

    resident, resident_checksum = _benchmark_reasoner(
        model,
        rows,
        transport,
        effect,
        device,
        resident=True,
        iterations=args.iterations,
        warmup=args.warmup,
    )
    cached, cached_checksum = _benchmark_reasoner(
        model,
        rows,
        transport,
        effect,
        device,
        resident=False,
        iterations=args.iterations,
        warmup=args.warmup,
    )

    sample_rate = float(source["cache"]["sample_rate_hz"])
    trainable = _parameter_count(model)
    payload = {
        "schema": "role_coupled_constant_memory_deployment_benchmark",
        "hardware": platform.processor() or platform.machine(),
        "device": str(device),
        "batch_size": 1,
        "sample_rate_hz": sample_rate,
        "scope": {
            "checkpoint_and_data_io_timed": False,
        },
        "parameters": {
            "trainable_reasoner": trainable,
        },
        "workload": workload,
        "reasoner_latency": {
            "resident_ms_per_video_frame": resident.to_dict(),
            "host_cached_ms_per_video_frame": cached.to_dict(),
            "resident_feature_fps": 1000.0 / resident.mean_ms,
            "host_cached_feature_fps": 1000.0 / cached.mean_ms,
            "resident_real_time_factor": (
                sample_rate * resident.mean_ms / 1000.0
            ),
        },
        "audit": {
            "architecture": metadata["architecture"],
            "checkpoint_sha256": _sha256(args.checkpoint),
            "examples_sha256": _sha256(args.examples),
            "checksums": {
                "reasoner_resident": resident_checksum,
                "reasoner_cached": cached_checksum,
            },
            "test_split_accessed": False,
        },
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
