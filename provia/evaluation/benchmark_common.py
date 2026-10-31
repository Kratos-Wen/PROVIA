"""Measure batch-one causal deployment cost for the procedure-mode filter.

The benchmark separates the RGB frontend, the per-frame marked-renewal
update, causal boundary decoding, and event-triggered procedure transport.
Every GPU latency is synchronized at each input frame.  This intentionally
measures online latency rather than batched offline throughput.

Video decoding and model/checkpoint loading are outside the timed region.  RGB
timing begins with a decoded host-memory frame and includes resize,
normalization, host-to-device transfer, the checkpoint-provenance backbone, and competitive role
pooling.  Cached-feature timing includes host-to-device transfer.  Procedure
transport is invoked only when the causal renewal decoder emits an onset.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch



@dataclass(frozen=True)
class Timing:
    calls: int
    total_seconds: float
    mean_ms: float
    median_ms: float | None
    p95_ms: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "calls": self.calls,
            "total_seconds": self.total_seconds,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
        }


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _time_calls(
    call: Callable[[int], float],
    *,
    iterations: int,
    warmup: int,
    device: torch.device | None,
) -> tuple[Timing, float]:
    if iterations <= 0 or warmup < 0:
        raise ValueError("iterations must be positive and warmup non-negative")
    checksum = 0.0
    for index in range(warmup):
        checksum += float(call(index))
    if device is not None:
        _synchronize(device)
    elapsed: list[float] = []
    for index in range(iterations):
        if device is not None:
            _synchronize(device)
        started = time.perf_counter()
        checksum += float(call(index + warmup))
        if device is not None:
            _synchronize(device)
        elapsed.append(time.perf_counter() - started)
    total = float(sum(elapsed))
    milliseconds = [1000.0 * value for value in elapsed]
    return (
        Timing(
            calls=iterations,
            total_seconds=total,
            mean_ms=1000.0 * total / iterations,
            median_ms=float(statistics.median(milliseconds)),
            p95_ms=_percentile(milliseconds, 95.0),
        ),
        checksum,
    )


def _parameter_count(module: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))
