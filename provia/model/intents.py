"""Compositional, role-preserving intention space for procedural events."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Optional


from provia.data.schema import Segment


FACTOR_NAMES = ("hand", "verb", "noun", "tool")
UNKNOWN = "<unk>"
MISSING = "<none>"


def _value(value: Optional[str]) -> str:
    if value is None or not str(value).strip():
        return MISSING
    return "_".join(str(value).strip().lower().replace("|", " ").split())


def segment_factors(segment: Segment) -> tuple[str, str, str, str]:
    verb = _value(segment.verb or segment.action_text)
    return (_value(segment.hand), verb, _value(segment.noun), _value(segment.tool))


def factor_token(values: tuple[str, str, str, str]) -> str:
    if len(values) != len(FACTOR_NAMES) or any("|" in value for value in values):
        raise ValueError("invalid factor tuple")
    return "|".join(values)


@dataclass(frozen=True)
class CompositionalIntentSpace:
    """Cartesian action space built from training/reference factors only."""

    labels: Mapping[str, tuple[str, ...]]
    tokens: tuple[str, ...]
    token_factor_indices: tuple[tuple[int, int, int, int], ...]

    @classmethod
    def from_segments(cls, segments: Iterable[Segment]) -> "CompositionalIntentSpace":
        observed = {name: {UNKNOWN} for name in FACTOR_NAMES}
        count = 0
        for segment in segments:
            count += 1
            for name, value in zip(FACTOR_NAMES, segment_factors(segment)):
                observed[name].add(value)
        if count == 0:
            raise ValueError("cannot build an intent space without training segments")
        labels = {name: tuple(sorted(observed[name])) for name in FACTOR_NAMES}
        index = {
            name: {value: position for position, value in enumerate(labels[name])}
            for name in FACTOR_NAMES
        }
        combinations = tuple(product(*(labels[name] for name in FACTOR_NAMES)))
        tokens = tuple(factor_token(values) for values in combinations)
        factor_indices = tuple(
            tuple(index[name][value] for name, value in zip(FACTOR_NAMES, values))
            for values in combinations
        )
        result = cls(labels, tokens, factor_indices)
        result.validate()
        return result

    def validate(self) -> None:
        if tuple(self.labels) != FACTOR_NAMES:
            raise ValueError(f"factor order must be {FACTOR_NAMES}")
        if any(UNKNOWN not in self.labels[name] for name in FACTOR_NAMES):
            raise ValueError("every factor needs an unknown class")
        expected = 1
        for name in FACTOR_NAMES:
            values = self.labels[name]
            if not values or len(values) != len(set(values)):
                raise ValueError(f"invalid {name} labels")
            expected *= len(values)
        if len(self.tokens) != expected or len(self.tokens) != len(
            self.token_factor_indices
        ):
            raise ValueError("intent space is not the complete Cartesian product")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("intent tokens are not unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "provia_compositional_intent_space",
            "labels": {name: list(self.labels[name]) for name in FACTOR_NAMES},
            "tokens": list(self.tokens),
            "token_factor_indices": [list(item) for item in self.token_factor_indices],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CompositionalIntentSpace":
        if value.get("schema") != "provia_compositional_intent_space":
            raise ValueError("unknown intent-space schema")
        raw_labels = value["labels"]
        if not isinstance(raw_labels, Mapping):
            raise ValueError("intent-space labels must be a mapping")
        result = cls(
            labels={
                name: tuple(str(item) for item in raw_labels[name])  # type: ignore[index]
                for name in FACTOR_NAMES
            },
            tokens=tuple(str(item) for item in value["tokens"]),  # type: ignore[arg-type]
            token_factor_indices=tuple(
                tuple(int(part) for part in item)  # type: ignore[arg-type]
                for item in value["token_factor_indices"]  # type: ignore[union-attr]
            ),
        )
        result.validate()
        return result

