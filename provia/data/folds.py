"""Fold projection and cross-root participant identity checks."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .schema import CanonicalEpisode


def materialize_cross_validation_fold(
    episodes: Iterable[CanonicalEpisode], fold: int
) -> tuple[CanonicalEpisode, ...]:
    """Replace generic cross-validation roles by one explicit partition."""

    if fold < 0:
        raise ValueError("fold must be non-negative")
    result: list[CanonicalEpisode] = []
    for episode in episodes:
        roles = episode.attributes.get("cross_validation_roles", ())
        if not roles:
            result.append(episode)
            continue
        matches = [item for item in roles if int(item["fold"]) == fold]
        if len(matches) != 1:
            raise ValueError(
                f"{episode.episode_id} has {len(matches)} assignments for fold {fold}"
            )
        partition = str(matches[0]["partition"]).lower()
        if partition not in {"train", "val", "validation", "test"}:
            raise ValueError(
                f"{episode.episode_id} has invalid fold partition {partition!r}"
            )
        normalized = "validation" if partition == "val" else partition
        result.append(
            replace(
                episode,
                split=normalized,
                attributes={**episode.attributes, "active_fold": fold},
            )
        )
    return tuple(result)

