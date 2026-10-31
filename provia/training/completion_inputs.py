"""Train the frozen reliability-aware causal completion model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor

from provia.training.completion_metrics import _read_language
from provia.training.common import _intact, _sha256
from provia.model.completion_transport import FactorizedCompletionTransport
from provia.model.mode_filter import (
    CORRECTION,
    jeffreys_observed_explanation_probability,
)
from provia.model.step_effects import ExpertEffectPosterior


MODEL_DIM = 192
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
GRADIENT_CLIP = 1.0
TARGET_CANDIDATES_PER_UPDATE = 2048
MAX_EPOCHS = 80
EARLY_STOPPING_EPOCHS = 10
def _class_probability(
    sequences: Sequence[Mapping[str, object]],
    *,
    correction_supervised: bool,
) -> Tensor:
    origin = torch.cat(
        [sequence["origin_target"].bool() for sequence in sequences]  # type: ignore[union-attr]
    )
    mode = torch.cat(
        [sequence["mode_target"].long() for sequence in sequences]  # type: ignore[union-attr]
    )
    return jeffreys_observed_explanation_probability(
        origin,
        mode,
        correction_supervised=correction_supervised,
    )


def _correction_is_supervised(
    sequences: Sequence[Mapping[str, object]],
) -> bool:
    """Infer whether recovery is an explicit event label in this benchmark."""

    return any(
        bool(
            torch.any(
                sequence["origin_target"].bool()  # type: ignore[union-attr]
                & (sequence["mode_target"].long() == CORRECTION)  # type: ignore[union-attr]
            )
        )
        for sequence in sequences
    )


def _load_transports(
    source: Mapping[str, object],
    device: torch.device,
    *,
    posterior_predictive_innovation: bool = False,
) -> tuple[FactorizedCompletionTransport, ...]:
    task_ids = tuple(str(value) for value in source["task_ids"])  # type: ignore[index]
    records = source.get("languages")
    if not isinstance(records, Mapping):
        raise ValueError("examples lack a language registry")
    deletion = float(source["missed_event_probability"])
    transports = []
    for task in task_ids:
        record = records.get(task)
        if not isinstance(record, Mapping):
            raise ValueError(f"task {task} lacks a frozen language")
        path = Path(str(record["path"]))
        if not _intact(path, record.get("sha256")):
            raise ValueError(f"task {task} language changed after example freezing")
        transports.append(
            FactorizedCompletionTransport(
                _read_language(path),
                deletion,
                posterior_predictive_innovation=(
                    posterior_predictive_innovation
                ),
            )
            .to(device)
            .eval()
        )
    return tuple(transports)


def _load_expert_effects(
    source: Mapping[str, object],
    registry_path: Path,
    examples_path: Path,
    transports: Sequence[FactorizedCompletionTransport],
    device: torch.device,
    rebuilt_examples: bool = False,
) -> tuple[ExpertEffectPosterior, ...]:
    with open(registry_path, encoding="utf-8") as handle:
        registry = json.load(handle)
    if (
        not isinstance(registry, Mapping)
        or registry.get("schema")
        not in {
            "completion_expert_effect_registry",
            "completion_normative_effect_registry",
        }
        or registry.get("validation_videos_used") is not False
        or registry.get("test_videos_used") is not False
        or not rebuilt_examples and registry.get("source_examples_sha256")
        not in {source.get("source_examples_sha256", _sha256(examples_path)), _sha256(examples_path), str(source.get("gap_base_examples_sha256") or "")}
    ):
        raise ValueError("expert-effect registry is malformed")
    records = registry.get("tasks")
    if not isinstance(records, Mapping):
        raise ValueError("expert-effect registry lacks task records")
    task_ids = tuple(str(value) for value in source["task_ids"])  # type: ignore[index]
    result = []
    for task_index, task in enumerate(task_ids):
        record = records.get(task)
        if not isinstance(record, Mapping):
            raise ValueError(f"task {task} lacks frozen expert effects")
        path = registry_path.parent / str(record["effects"])
        if not _intact(path, record.get("sha256")):
            raise ValueError(f"task {task} expert effects changed after freezing")
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(artifact, Mapping):
            raise ValueError(f"task {task} expert-effect artifact is malformed")
        effect = ExpertEffectPosterior.from_artifact(artifact).to(device).eval()
        if effect.node_ids != tuple(transports[task_index].language.node_ids):
            raise ValueError(f"task {task} expert effects and language disagree")
        result.append(effect)
    return tuple(result)
