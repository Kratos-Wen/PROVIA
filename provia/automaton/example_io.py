"""Read and attach observer states to training examples.

The observer state is computed from the same frozen checkpoint that produced
each candidate.  Only training and validation streams are loaded, and each
context ends at the candidate's causal checkpoint.  This preserves the full
timing and semantic state of the observer without decoding future frames or
retraining the observer for anomaly labels.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

import torch

from provia.data.io import read_jsonl
from provia.training.common import _partitions
from provia.model.intents import CompositionalIntentSpace


def _atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _load_source(path: Path) -> tuple[dict[str, object], Mapping[str, object]]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict):
        raise ValueError("completion examples are malformed")
    metadata = artifact.get("metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("test_split_accessed") is not False
        or not isinstance(artifact.get("train"), Sequence)
        or not isinstance(artifact.get("validation"), Sequence)
    ):
        raise ValueError("completion examples are malformed")
    return artifact, metadata


def _episodes_and_intents(metadata: Mapping[str, object]):
    manifest = Path(str(metadata["manifest"]))
    episodes = tuple(read_jsonl(manifest))
    train, validation = _partitions(episodes, int(metadata["fold"]))
    intent = CompositionalIntentSpace.from_segments(
        segment for episode in train for segment in episode.segments
    )
    return train, validation, intent
