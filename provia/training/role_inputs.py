"""Load the bimanual transport and step-effect inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch

from provia.training.common import _intact, _sha256
from provia.model.step_effects import ExpertEffectPosterior
from provia.model.interval_transport import (
    ContinuationTransport,
    IntervalContinuationLanguage,
)


MODEL_DIM = 192
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
GRADIENT_CLIP = 1.0
TARGET_CHECKPOINTS_PER_UPDATE = 2048
MAX_EPOCHS = 80
EARLY_STOPPING_EPOCHS = 10
def _load_transport(
    source: Mapping[str, object],
    device: torch.device,
    *,
    posterior_predictive_innovation: bool = False,
) -> ContinuationTransport:
    path = Path(str(source["procedure_language"]))
    if not _intact(path, source.get("procedure_language_sha256")):
        raise ValueError("procedure language changed after example freezing")
    with open(path, encoding="utf-8") as handle:
        language = IntervalContinuationLanguage.from_dict(json.load(handle))
    deletion = tuple(float(value) for value in source["missed_event_probability"])  # type: ignore[index]
    return (
        ContinuationTransport(
            language,
            optional_stutter=False,
            missed_event_probability=deletion,
            posterior_predictive_innovation=posterior_predictive_innovation,
        )
        .to(device)
        .eval()
    )


def _load_expert_effect(
    source: Mapping[str, object],
    path: Path,
    transport: ContinuationTransport,
    device: torch.device,
    rebuilt_examples: bool = False,
) -> ExpertEffectPosterior:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, Mapping):
        raise ValueError("expert-effect artifact is malformed")
    metadata = artifact.get("metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("test_split_accessed") is not False
        or metadata.get("uses_expert_annotations") is not False
        or not rebuilt_examples and metadata.get("source_language_sha256")
        != source.get(
            "expert_effect_source_language_sha256",
            source.get("procedure_language_sha256"),
        )
    ):
        raise ValueError("expert-effect artifact uses another language")
    if (
        source.get("effect_statistic_interval")
        == "current predicted onset-to-end interval"
        and (
            metadata.get("effect_support_interval")
            != "current predicted onset-to-end interval"
            or metadata.get("right_censored_events_used") is not False
        )
    ):
        raise ValueError(
            "current-event examples require matching non-censored "
            "onset-to-end expert effects"
        )
    effect = ExpertEffectPosterior.from_artifact(artifact).to(device).eval()
    if effect.node_ids != tuple(transport.language.node_ids):
        raise ValueError("expert effects and continuation nodes disagree")
    return effect
