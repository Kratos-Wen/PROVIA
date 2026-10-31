"""Frame-aligned supervision for causal latent-event monitoring."""

from __future__ import annotations

from dataclasses import dataclass





IGNORE_INDEX = -100
CORRECTNESS_INDEX = {"correct": 0, "error": 1, "correction": 2}


@dataclass(frozen=True)
class StreamingEventInterval:
    segment_id: str
    role: int
    start: int
    onset: int
    end: int
    correctness: int
    error_types: tuple[str, ...]

