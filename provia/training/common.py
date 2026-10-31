"""Helpers shared by the training entry points.

Phase and event semantics are the only optimization targets.  Anomaly and
correction labels are neither consumed by the loss nor used for checkpoint
selection; they remain reserved for the downstream procedural-state model.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from provia.data.folds import materialize_cross_validation_fold
from provia.data.role_stream import CompetitiveRoleStream
from provia.data.schema import CanonicalEpisode
from provia.data.streaming_events import IGNORE_INDEX
from provia.observer.factors import EVENT_FACTORS, EVENT_PHASES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _intact(path: Path, expected: object) -> bool:
    """Whether a file still has the hash recorded for it; released artifacts record none."""
    return expected is None or _sha256(path) == expected


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _partitions(
    episodes: Sequence[CanonicalEpisode], fold: int
) -> tuple[tuple[CanonicalEpisode, ...], tuple[CanonicalEpisode, ...]]:
    materialized = materialize_cross_validation_fold(episodes, fold)
    train = tuple(item for item in materialized if item.split == "train")
    validation = tuple(item for item in materialized if item.split == "validation")
    test = tuple(item for item in materialized if item.split == "test")
    if not train or not validation or not test:
        raise ValueError("fold requires non-empty train, validation, and test splits")
    subjects = [
        {item.subject_id for item in partition if item.subject_id is not None}
        for partition in (train, validation, test)
    ]
    if subjects[0] & subjects[1] or subjects[0] & subjects[2] or subjects[1] & subjects[2]:
        raise ValueError("participant leakage across fold partitions")
    return train, validation


def _cache_provenance(root: Path, fold: int) -> dict[str, object]:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("schema") != "role_stream_features":
        raise ValueError("unknown role-stream cache schema")
    if int(value.get("role_change_dim", -1)) != int(value.get("role_dim", -2)):
        raise ValueError("role-change and role feature dimensions must match")
    if int(value.get("fold", -1)) != fold:
        raise ValueError("feature cache belongs to another fold")
    if set(value.get("partitions", ())) != {"train", "validation"}:
        raise ValueError("feature cache must contain train and validation only")
    if value.get("contains_annotations") is not False:
        raise ValueError("feature cache must not contain annotations")
    if value.get("test_split_accessed") is not False:
        raise ValueError("feature cache reports test access")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        **value,
    }


def _statistics(
    streams: Sequence[CompetitiveRoleStream],
    labels: Mapping[str, Sequence[str]],
) -> tuple[Tensor, dict[str, Tensor]]:
    phase = torch.zeros(len(EVENT_PHASES), dtype=torch.float64)
    factors = {
        name: torch.zeros(len(labels[name]), dtype=torch.float64)
        for name in EVENT_FACTORS
    }
    for stream in streams:
        phase += torch.bincount(
            stream.phase_targets.reshape(-1), minlength=len(EVENT_PHASES)
        )
        weights = stream.factor_weights.reshape(-1).to(torch.float64)
        for name in EVENT_FACTORS:
            targets = stream.factor_targets[name].reshape(-1)
            valid = targets != IGNORE_INDEX
            factors[name].scatter_add_(0, targets[valid], weights[valid])
    if torch.any(phase <= 0):
        raise ValueError("every phase must occur in the training split")
    return phase, factors


def _jeffreys_prior(counts: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    values = counts.to(device=device, dtype=dtype) + 0.5
    return values / values.sum()


def _prior_corrected_nll(
    logits: Tensor,
    targets: Tensor,
    counts: Tensor,
    *,
    weights: Tensor | None = None,
) -> Tensor:
    prior = _jeffreys_prior(counts, device=logits.device, dtype=logits.dtype)
    losses = F.cross_entropy(
        (logits + prior.log()).reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    )
    valid = targets.reshape(-1) != IGNORE_INDEX
    if not torch.any(valid):
        raise ValueError("a supervised variable has no valid target")
    if weights is None:
        return losses[valid].mean()
    flat_weights = weights.reshape(-1).to(device=logits.device, dtype=logits.dtype)
    return (losses[valid] * flat_weights[valid]).sum() / flat_weights[
        valid
    ].sum().clamp_min(torch.finfo(logits.dtype).tiny)
