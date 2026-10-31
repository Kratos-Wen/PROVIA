"""Benchmark exact batch-one streaming inference for completion PROVIA.

Video decoding, checkpoint loading, and language compilation are excluded.  RGB
timing begins with a decoded host-memory frame.  Cached-feature timing includes
host-to-device transfer.  Every input frame is CUDA-synchronized so the result
is online latency rather than batched offline throughput.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from provia.data.completion_stream import CompletionStream
from provia.data.folds import materialize_cross_validation_fold
from provia.data.io import read_jsonl
from provia.evaluation.benchmark_common import Timing, _time_calls
from provia.observer.completion_process import (
    CompletionProcessStreamState,
    MarkedCompletionProcess,
)
from provia.training.backbone import _preprocess


def _cached_rows(streams: Sequence[CompletionStream]):
    rows = []
    for stream in streams:
        for index in range(len(stream.global_features)):
            change = torch.zeros(2, stream.change_features.shape[-1])
            change[0] = stream.change_features[index]
            rows.append(
                (
                    stream.global_features[index].float(),
                    change,
                    stream.timestamps_sec[index].float(),
                    index == 0,
                )
            )
    if not rows:
        raise ValueError("empty cached-feature workload")
    return rows


@torch.inference_mode()
def _benchmark_cached_updates(
    model: MarkedCompletionProcess,
    streams: Sequence[CompletionStream],
    device: torch.device,
    *,
    iterations: int,
    warmup: int,
    resident: bool,
) -> tuple[Timing, float]:
    rows = _cached_rows(streams)
    if resident:
        rows = [
            (global_value.to(device), change.to(device), timestamp.to(device), reset)
            for global_value, change, timestamp, reset in rows
        ]
    state: CompletionProcessStreamState | None = None

    def update(index: int) -> float:
        nonlocal state
        global_value, change, timestamp, reset = rows[index % len(rows)]
        if reset or index % len(rows) == 0:
            state = None
        if not resident:
            global_value = global_value.to(device)
            change = change.to(device)
            timestamp = timestamp.to(device)
        evidence, state = model.step(
            global_value.unsqueeze(0),
            change.unsqueeze(0),
            timestamp.reshape(1),
            state,
        )
        checksum = float(evidence.intensity_per_second.float().cpu().sum())
        checksum += sum(
            float(value.float().cpu().sum())
            for value in evidence.factor_logits.values()
        )
        return checksum

    return _time_calls(
        update, iterations=iterations, warmup=warmup, device=device
    )


def _representative_rgb(
    manifest: Path, fold: int, sample_rate_hz: float, count: int
) -> tuple[np.ndarray, str]:
    try:
        import decord
    except ImportError as exc:
        raise RuntimeError("decord is required for RGB timing") from exc
    episodes = materialize_cross_validation_fold(tuple(read_jsonl(manifest)), fold)
    episode = next(value for value in episodes if value.split == "validation")
    if not episode.video_path or not Path(episode.video_path).is_file():
        raise FileNotFoundError(episode.video_path or episode.video_id)
    reader = decord.VideoReader(
        str(episode.video_path), ctx=decord.cpu(0), num_threads=2
    )
    native_fps = float(reader.get_avg_fps())
    stride = max(1, int(round(native_fps / sample_rate_hz)))
    ids = np.arange(0, min(len(reader), stride * count), stride, dtype=np.int64)
    if not len(ids):
        raise ValueError("representative video has no frames")
    return reader.get_batch(ids.tolist()).asnumpy(), str(episode.video_path)


@torch.inference_mode()
def _benchmark_rgb_to_observer(
    backbone: torch.nn.Module,
    model: MarkedCompletionProcess,
    decoded: np.ndarray,
    sample_rate_hz: float,
    device: torch.device,
    *,
    iterations: int,
    warmup: int,
) -> tuple[Timing, float]:
    state: CompletionProcessStreamState | None = None
    previous: torch.Tensor | None = None

    def update(index: int) -> float:
        nonlocal state, previous
        local = index % len(decoded)
        if local == 0:
            state = None
            previous = None
        inputs = _preprocess(decoded[local : local + 1], device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = backbone.forward_features(inputs)
            current = output["x_norm_clstoken"].float()
        change = torch.zeros(1, 2, current.shape[-1], device=device)
        if previous is not None:
            change[:, 0] = current - previous
        timestamp = torch.tensor(
            [local / sample_rate_hz], device=device, dtype=torch.float32
        )
        evidence, state = model.step(current, change, timestamp, state)
        previous = current
        checksum = float(evidence.intensity_per_second.float().cpu().sum())
        checksum += sum(
            float(value.float().cpu().sum())
            for value in evidence.factor_logits.values()
        )
        return checksum

    return _time_calls(
        update, iterations=iterations, warmup=warmup, device=device
    )
