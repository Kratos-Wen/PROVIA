"""Benchmark constant-memory single-actor streaming inference.

The reasoner workload advances one active candidate frame without replaying
its prefix. Optional RGB timing adds the exact checkpoint-provenance encoder, frozen
marked-completion observer, causal posterior-median decoder, and reasoner used
by the evaluation stream. Video decoding and checkpoint loading stay
outside the timed region.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor

from provia.evaluation.benchmark_common import (
    Timing,
    _atomic_json,
    _parameter_count,
    _time_calls,
)
from provia.data.completion_stream import CompletionStream, load_completion_streams
from provia.data.folds import materialize_cross_validation_fold
from provia.data.io import read_jsonl
from provia.evaluation.benchmark_frontend import (
    _benchmark_cached_updates,
    _benchmark_rgb_to_observer,
    _representative_rgb,
)
from provia.observer.completion_process import (
    MarkedCompletionProcess,
    build_marked_completion_process,
    decode_completion_posterior_medians,
)
from provia.observer.factors import EVENT_FACTORS
from provia.training.backbone import _load_backbone_from_provenance
from provia.evaluation.evaluate_single_actor import _checkpoint_relative
from provia.training.common import _intact, _sha256
from provia.training.completion_inputs import (
    _load_expert_effects,
    _load_transports,
)
from provia.training.completion_metrics import _priors
from provia.training.train_single_actor import (
    _build_model,
    _endpoint_factors,
    _global_stream,
    _load_prefix_registry,
    _prefix_evidence,
    _source_with_predictive_language_registry,
)
from provia.model.intent_state import (
    CoupledIntentPredictiveStateCompletionModel,
    CoupledIntentStreamState,
)
from provia.model.step_effects import ExpertEffectPosterior
from provia.model.intents import CompositionalIntentSpace
from provia.model.single_actor import (
    PROVIA_SHARED_SUCCESSOR_ARCHITECTURE,
    PROVIA_SINGLE_ACTOR_ARCHITECTURE,
)


@dataclass(frozen=True)
class _FrameRecord:
    task: int
    reset: bool
    completed: bool
    feature: Tensor
    event_probability: Tensor
    factor_probability: Mapping[str, Tensor]
    endpoint_factor_probability: Mapping[str, Tensor] | None
    effect_statistic: Tensor | None

    def to(self, device: torch.device) -> "_FrameRecord":
        return _FrameRecord(
            task=self.task,
            reset=self.reset,
            completed=self.completed,
            feature=self.feature.to(device),
            event_probability=self.event_probability.to(device),
            factor_probability={
                name: value.to(device)
                for name, value in self.factor_probability.items()
            },
            endpoint_factor_probability=(
                None
                if self.endpoint_factor_probability is None
                else {
                    name: value.to(device)
                    for name, value in self.endpoint_factor_probability.items()
                }
            ),
            effect_statistic=(
                None
                if self.effect_statistic is None
                else self.effect_statistic.to(device)
            ),
        )


def _load(
    examples_path: Path,
    checkpoint_path: Path,
    split: str,
    device: torch.device,
    rebuilt_prefix: Path | None,
) -> tuple[
    CoupledIntentPredictiveStateCompletionModel,
    Mapping[str, object],
    Sequence[Mapping[str, object]],
    tuple[torch.nn.Module, ...],
    tuple[ExpertEffectPosterior, ...],
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
        or source.get("schema") != "completion_examples"
        or source.get("test_split_accessed") is not False
        or not isinstance(sequences, Sequence)
        or not sequences
        or not isinstance(metadata, Mapping)
        or metadata.get("architecture")
        not in {
            PROVIA_SINGLE_ACTOR_ARCHITECTURE,
            PROVIA_SHARED_SUCCESSOR_ARCHITECTURE,
        }
        or (rebuilt_prefix is None and metadata.get("examples_sha256") != _sha256(examples_path))
        or metadata.get("test_split_accessed") is not False
    ):
        raise ValueError("streaming benchmark artifacts are incompatible")
    registry_path = _checkpoint_relative(checkpoint_path, metadata["language_registry"])
    if not _intact(registry_path, metadata.get("language_registry_sha256")):
        raise ValueError("predictive-language registry changed after training")
    source, _ = _source_with_predictive_language_registry(
        source, registry_path, examples_path, rebuilt_examples=rebuilt_prefix is not None
    )
    model = _build_model(
        source,
        device,
        independent_successors=metadata["architecture"] == PROVIA_SINGLE_ACTOR_ARCHITECTURE,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
    transports = _load_transports(
        source, device, posterior_predictive_innovation=True
    )
    effect_registry = _checkpoint_relative(checkpoint_path, metadata["expert_effect_registry"])
    if not _intact(effect_registry, metadata.get("expert_effect_registry_sha256")):
        raise ValueError("expert-effect registry changed after training")
    effects = _load_expert_effects(
        source, effect_registry, examples_path, transports, device,
        rebuilt_examples=rebuilt_prefix is not None,
    )
    prefix_root = rebuilt_prefix or Path(str(metadata["prefix_evidence"]))
    if rebuilt_prefix is None and _sha256(prefix_root / "registry.json") != metadata.get(
        "prefix_evidence_registry_sha256"
    ):
        raise ValueError("prefix evidence changed after training")
    prefix_registry = _load_prefix_registry(prefix_root, examples_path)
    return (
        model,
        source,
        sequences,
        transports,
        effects,
        prefix_root,
        prefix_registry,
        metadata,
    )


def _workload(
    sequences: Sequence[Mapping[str, object]],
    feature_root: Path,
    prefix_root: Path,
    prefix_registry: Mapping[str, object],
    target_frames: int,
    sample_rate_hz: float,
) -> tuple[list[_FrameRecord], dict[str, float | int]]:
    records: list[_FrameRecord] = []
    validation_frames = 0
    active_frames = 0
    candidates = 0
    for sequence in sequences:
        decoded = sequence["decoded_index"]
        if not isinstance(decoded, Tensor) or decoded.dtype != torch.long:
            raise ValueError("decoded candidate indices are malformed")
        duration = float(sequence["stream_end_sec"]) - float(
            sequence["stream_start_sec"]
        )
        validation_frames += int(round(duration * sample_rate_hz)) + 1
        if len(decoded):
            active_frames += int(decoded[-1])
        candidates += len(decoded)
        if len(records) >= target_frames:
            continue
        feature, timestamp = _global_stream(
            feature_root, str(sequence["video_id"])
        )
        event, factors = _prefix_evidence(
            prefix_root,
            prefix_registry,
            str(sequence["video_id"]),
            timestamp,
        )
        previous_end = 0
        first_record = True
        for candidate_index in range(len(decoded)):
            end = int(decoded[candidate_index])
            start = previous_end + 1
            endpoint = _endpoint_factors(
                sequence, candidate_index, torch.device("cpu")
            )
            statistic = sequence["statistic"][candidate_index].float()  # type: ignore[index,union-attr]
            for frame_index in range(start, end + 1):
                completed = frame_index == end
                records.append(
                    _FrameRecord(
                        task=int(sequence["task_index"]),
                        reset=first_record,
                        completed=completed,
                        feature=feature[frame_index].float(),
                        event_probability=event[frame_index].float(),
                        factor_probability={
                            name: values[frame_index].float()
                            for name, values in factors.items()
                        },
                        endpoint_factor_probability=(
                            endpoint if completed else None
                        ),
                        effect_statistic=(statistic if completed else None),
                    )
                )
                first_record = False
            previous_end = end
            if len(records) >= target_frames:
                break
    if not records or not any(value.completed for value in records):
        raise ValueError("benchmark workload contains no complete candidate")
    return records, {
        "validation_frames": validation_frames,
        "active_candidate_frames": active_frames,
        "decoded_candidates": candidates,
        "active_frame_fraction": active_frames / validation_frames,
        "active_frames_per_candidate": active_frames / candidates,
        "timed_workload_frames": len(records),
        "timed_workload_completions": sum(value.completed for value in records),
    }


@torch.inference_mode()
def _benchmark(
    model: CoupledIntentPredictiveStateCompletionModel,
    records: Sequence[_FrameRecord],
    transports: Sequence[torch.nn.Module],
    effects: Sequence[ExpertEffectPosterior],
    device: torch.device,
    *,
    resident: bool,
    iterations: int,
    warmup: int,
):
    rows = [value.to(device) for value in records] if resident else list(records)
    keys = tuple(model.energy.build_frontier_key(value) for value in transports)
    state: CoupledIntentStreamState | None = None

    def update(index: int) -> float:
        nonlocal state
        row = rows[index % len(rows)]
        if not resident:
            row = row.to(device)
        if row.reset or index % len(rows) == 0:
            state = model.initial_stream_state(
                row.feature, transports[row.task]  # type: ignore[arg-type]
            )
        if state is None:
            raise RuntimeError("stream state was not initialized")
        step = model.forward_stream_frame(
            row.feature,
            row.factor_probability,
            row.event_probability,
            state,
            transports[row.task],  # type: ignore[arg-type]
            completed=row.completed,
            endpoint_factor_probability=row.endpoint_factor_probability,
            effect_statistic=row.effect_statistic,
            expert_effect=effects[row.task] if row.completed else None,
            frontier_key=keys[row.task],
        )
        state = step.state
        return float(step.anomaly_probability.detach().cpu())

    return _time_calls(
        update, iterations=iterations, warmup=warmup, device=device
    )


def _frontend(
    source: Mapping[str, object],
    device: torch.device,
    backbone_repo: Path,
):
    cache = source.get("cache")
    if not isinstance(cache, Mapping) or cache.get("role_mode") != "single_actor":
        raise ValueError("this RGB benchmark requires a single-actor feature stream")
    episodes = materialize_cross_validation_fold(
        tuple(read_jsonl(Path(str(source["manifest"])))), int(source["fold"])
    )
    train_native = tuple(value for value in episodes if value.split == "train")
    validation_native = tuple(
        value for value in episodes if value.split == "validation"
    )
    if not train_native or not validation_native:
        raise ValueError("frontend benchmark requires train and validation streams")
    intent_space = CompositionalIntentSpace.from_segments(
        segment for episode in train_native for segment in episode.segments
    )
    validation = load_completion_streams(
        validation_native[:1], Path(str(source["feature_root"])), intent_space
    )
    labels = {
        name: tuple(str(value) for value in source["factor_labels"][name])  # type: ignore[index]
        for name in EVENT_FACTORS
    }
    observer_path = Path(str(source["observer_checkpoint"]))
    if _sha256(observer_path) != source.get("observer_checkpoint_sha256"):
        raise ValueError("frozen completion observer changed after example creation")
    checkpoint = torch.load(observer_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("test_split_accessed") is not False
        or int(metadata.get("fold", -1)) != int(source["fold"])
    ):
        raise ValueError("completion observer is malformed or fold-incompatible")
    observer = build_marked_completion_process(
        metadata, int(cache["global_dim"]), labels
    )
    observer.load_state_dict(checkpoint["model"], strict=True)
    observer.to(device).eval().requires_grad_(False)
    _, backbone = _load_backbone_from_provenance(backbone_repo, cache, device)
    return observer, backbone, validation, metadata


@torch.inference_mode()
def _completion_decoder_inputs(
    observer: MarkedCompletionProcess,
    streams: Sequence[CompletionStream],
    device: torch.device,
) -> tuple[tuple[Tensor, Mapping[str, Tensor]], ...]:
    rows = []
    for stream in streams:
        global_feature = stream.global_features.to(
            device=device, dtype=torch.float32
        )
        change_feature = stream.change_features.to(
            device=device, dtype=torch.float32
        )
        timestamp = stream.timestamps_sec.to(
            device=device, dtype=torch.float32
        )
        role_change = global_feature.new_zeros(
            len(global_feature), 2, global_feature.shape[-1]
        )
        role_change[:, 0] = change_feature
        evidence = observer(global_feature, role_change, timestamp)
        event_probability = observer.event_probability(
            evidence.intensity_per_second, timestamp
        )
        rows.append(
            (
                event_probability.float().cpu(),
                {
                    name: value.float().cpu()
                    for name, value in evidence.factor_logits.items()
                },
            )
        )
    if not rows:
        raise ValueError("completion decoder timing requires a validation stream")
    return tuple(rows)


def _benchmark_completion_decoder(
    inputs: Sequence[tuple[Tensor, Mapping[str, Tensor]]],
    priors: Mapping[str, Tensor],
    repeats: int,
) -> tuple[Timing, int, float]:
    total_frames = sum(len(event) for event, _ in inputs)
    decoded_candidates = 0

    def decode_all(_: int) -> float:
        nonlocal decoded_candidates
        checksum = 0.0
        local_count = 0
        for event, factors in inputs:
            decoded = decode_completion_posterior_medians(
                event, factors, priors
            )
            local_count += len(decoded)
            checksum += sum(value.end_index for value in decoded)
        decoded_candidates = local_count
        return checksum

    timing, checksum = _time_calls(
        decode_all, iterations=repeats, warmup=1, device=None
    )
    per_frame = Timing(
        calls=total_frames,
        total_seconds=timing.total_seconds / repeats,
        mean_ms=1000.0 * timing.total_seconds / (repeats * total_frames),
        median_ms=None,
        p95_ms=None,
    )
    return per_frame, decoded_candidates, checksum


def main() -> None:
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
    parser.add_argument(
        "--backbone-repo",
        "--dinov2-repo",
        dest="backbone_repo",
        type=Path,
    )
    parser.add_argument("--rgb-frames", type=int, default=64)
    parser.add_argument("--decoder-repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if (
        min(
            args.workload_frames,
            args.iterations,
            args.rgb_frames,
            args.decoder_repeats,
        )
        <= 0
        or args.warmup < 0
    ):
        parser.error("benchmark counts must be positive")
    device = torch.device(args.device)
    (
        model,
        source,
        sequences,
        transports,
        effects,
        prefix_root,
        prefix_registry,
        metadata,
    ) = _load(args.examples, args.checkpoint, args.split, device, args.prefix_evidence)
    sample_rate = float(source["cache"]["sample_rate_hz"])  # type: ignore[index]
    backbone_name = str(source["cache"]["backbone"])  # type: ignore[index]
    rows, workload = _workload(
        sequences,
        Path(str(source["feature_root"])),
        prefix_root,
        prefix_registry,
        args.workload_frames,
        sample_rate,
    )
    frontend = (
        None
        if args.backbone_repo is None
        else _frontend(source, device, args.backbone_repo)
    )
    resident, resident_checksum = _benchmark(
        model,
        rows,
        transports,
        effects,
        device,
        resident=True,
        iterations=args.iterations,
        warmup=args.warmup,
    )
    cached, cached_checksum = _benchmark(
        model,
        rows,
        transports,
        effects,
        device,
        resident=False,
        iterations=args.iterations,
        warmup=args.warmup,
    )
    if frontend is None:
        observer = backbone = validation_streams = None
        observer_resident = observer_cached = rgb = None
        decoder = None
        decoded_candidates = None
        observer_resident_checksum = observer_cached_checksum = rgb_checksum = None
        decoder_checksum = None
        representative_video = None
    else:
        observer, backbone, validation_streams, observer_metadata = frontend
        decoder_inputs = _completion_decoder_inputs(
            observer, validation_streams, device
        )
        decoder, decoded_candidates, decoder_checksum = (
            _benchmark_completion_decoder(
                decoder_inputs,
                _priors(observer_metadata),
                args.decoder_repeats,
            )
        )
        observer_resident, observer_resident_checksum = _benchmark_cached_updates(
            observer,
            validation_streams,
            device,
            iterations=args.iterations,
            warmup=args.warmup,
            resident=True,
        )
        observer_cached, observer_cached_checksum = _benchmark_cached_updates(
            observer,
            validation_streams,
            device,
            iterations=args.iterations,
            warmup=args.warmup,
            resident=False,
        )
        decoded_rgb, representative_video = _representative_rgb(
            Path(str(source["manifest"])),
            int(source["fold"]),
            sample_rate,
            args.rgb_frames,
        )
        rgb, rgb_checksum = _benchmark_rgb_to_observer(
            backbone,
            observer,
            decoded_rgb,
            sample_rate,
            device,
            iterations=args.iterations,
            warmup=args.warmup,
        )
    active_fraction = float(workload["active_frame_fraction"])
    active_per_candidate = float(workload["active_frames_per_candidate"])
    trainable_parameters = _parameter_count(model)
    frozen_observer_parameters = 0
    frozen_pretrained_parameters = 0
    if frontend is not None:
        frozen_observer_parameters = _parameter_count(observer)
        frozen_pretrained_parameters = _parameter_count(backbone)
    frozen_parameters = (
        frozen_observer_parameters + frozen_pretrained_parameters
    )
    events_per_frame = float(workload["decoded_candidates"]) / float(
        workload["validation_frames"]
    )
    serial_frontend_reasoner = None
    if frontend is not None:
        post_feature_ms = (
            observer_cached.mean_ms
            + decoder.mean_ms
            + resident.mean_ms * active_fraction
        )
        rgb_to_score_ms = (
            rgb.mean_ms
            + decoder.mean_ms
            + resident.mean_ms * active_fraction
        )
        serial_frontend_reasoner = {
            "post_feature_cached_ms_per_frame": post_feature_ms,
            "post_feature_cached_feature_fps": 1000.0 / post_feature_ms,
            "post_feature_amortized_compute_ms_per_predicted_completion": (
                post_feature_ms / events_per_frame
            ),
            "post_feature_real_time_factor": sample_rate
            * post_feature_ms
            / 1000.0,
            "rgb_to_score_ms_per_frame": rgb_to_score_ms,
            "rgb_to_score_fps": 1000.0 / rgb_to_score_ms,
            "rgb_to_score_real_time_factor": sample_rate
            * rgb_to_score_ms
            / 1000.0,
        }
    payload = {
        "schema": "coupled_completion_constant_memory_benchmark",
        "hardware": platform.processor() or platform.machine(),
        "device": str(device),
        "batch_size": 1,
        "sample_rate_hz": sample_rate,
        "scope": {
            "included": (
                f"decoded RGB through {source['cache']['backbone']}, the frozen marked-completion "
                "observer, causal posterior-median decoding, and one exact "
                "constant-memory anomaly-state update per active candidate frame"
                if frontend is not None
                else "one exact constant-memory anomaly-state update per active candidate frame"
            ),
            "checkpoint_and_data_io_timed": False,
        },
        "parameters": {
            "trainable": trainable_parameters,
            "frozen": {
                "event_observer": frozen_observer_parameters,
                "pretrained_backbone": frozen_pretrained_parameters,
                "total": frozen_parameters,
            },
            "ensemble_total": trainable_parameters + frozen_parameters,
            "ensemble_members": 1,
        },
        "workload": workload,
        "latency": {
            "resident_ms_per_active_frame": resident.to_dict(),
            "host_cached_ms_per_active_frame": cached.to_dict(),
            "resident_cumulative_compute_ms_per_candidate": resident.mean_ms
            * active_per_candidate,
            "host_cached_cumulative_compute_ms_per_candidate": cached.mean_ms
            * active_per_candidate,
        },
        "deployment": {
            "resident_active_feature_fps": 1000.0 / resident.mean_ms,
            "host_cached_active_feature_fps": 1000.0 / cached.mean_ms,
            "amortized_resident_ms_per_video_frame": resident.mean_ms
            * active_fraction,
            "amortized_host_cached_ms_per_video_frame": cached.mean_ms
            * active_fraction,
            "reasoner_real_time_factor": sample_rate
            * resident.mean_ms
            * active_fraction
            / 1000.0,
        },
        "frontend": (
            None
            if frontend is None
            else {
                "observer_resident_ms_per_frame": observer_resident.to_dict(),
                "observer_host_cached_ms_per_frame": observer_cached.to_dict(),
                "rgb_to_observer_ms_per_frame": rgb.to_dict(),
                "causal_decoder_ms_per_frame": decoder.to_dict(),
                "decoded_candidates_in_decoder_workload": decoded_candidates,
                "representative_video": representative_video,
                "video_decode_timed": False,
                "precision": f"{backbone_name} {'bfloat16 autocast' if device.type == 'cuda' else 'float32'}",
            }
        ),
        "serial_frontend_reasoner": serial_frontend_reasoner,
        "audit": {
            "architecture": metadata["architecture"],
            "checkpoint_sha256": _sha256(args.checkpoint),
            "examples_sha256": _sha256(args.examples),
            "checksums": {
                "resident": resident_checksum,
                "cached": cached_checksum,
                "observer_resident": observer_resident_checksum,
                "observer_cached": observer_cached_checksum,
                "rgb": rgb_checksum,
                "decoder": decoder_checksum,
            },
            "observer_checkpoint_sha256": (
                None
                if frontend is None
                else _sha256(Path(str(source["observer_checkpoint"])))
            ),
            "test_split_accessed": False,
        },
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
