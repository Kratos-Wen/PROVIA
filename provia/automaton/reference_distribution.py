"""Compile uncertain decoded expert events into a role-asynchronous language."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from provia.model.intents import CompositionalIntentSpace


def _normalized(value: object) -> str:
    text = str(value or "<none>").strip().lower()
    return "_".join(text.replace("|", " ").split()) or "<none>"


def project_reference_distribution(
    raw: object,
    name: str,
    path: Path,
    intent_space: CompositionalIntentSpace,
) -> tuple[tuple[float, ...], dict[str, float | int]]:
    """Project a decoder posterior to the fold-local possibilistic support."""

    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: {name} distribution must be an object")
    labels = raw.get("labels")
    log_probability = raw.get("log_probability")
    if (
        not isinstance(labels, list)
        or not isinstance(log_probability, list)
        or len(labels) != len(log_probability)
        or not labels
    ):
        raise ValueError(f"{path}: malformed {name} distribution")
    log_values = np.asarray(log_probability, dtype=np.float64)
    if log_values.ndim != 1 or not np.isfinite(log_values).all():
        raise ValueError(f"{path}: non-finite {name} log probabilities")
    probability = np.exp(log_values - np.logaddexp.reduce(log_values))
    possibility = np.exp(log_values - log_values.max())
    normalized_labels = [_normalized(label) for label in labels]
    target_index = {
        label: index for index, label in enumerate(intent_space.labels[name])
    }
    unknown_index = target_index["<unk>"]
    top_label = normalized_labels[int(probability.argmax())]
    unknown_mass = float(
        sum(
            value
            for label, value in zip(normalized_labels, probability)
            if label not in target_index and label != "<unk>"
        )
    )
    target = np.zeros(len(target_index), dtype=np.float64)
    for label, value in zip(normalized_labels, possibility):
        index = target_index.get(label, unknown_index)
        target[index] = max(target[index], float(value))
    if target.max() <= 0:
        raise ValueError(f"{path}: {name} distribution has no support")
    target /= target.max()
    return tuple(float(value) for value in target), {
        "unknown_argmax": int(top_label not in target_index and top_label != "<unk>"),
        "unknown_probability_mass": unknown_mass,
        "top1_probability": float(probability.max()),
    }
