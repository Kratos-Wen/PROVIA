"""Uncertain expert events of the role-asynchronous language."""

from __future__ import annotations

from dataclasses import dataclass




@dataclass(frozen=True)
class UncertainReferenceEvent:
    verb: tuple[float, ...]
    noun: tuple[float, ...]
    tool: tuple[float, ...]

    def factor(self, name: str) -> tuple[float, ...]:
        if name not in {"verb", "noun", "tool"}:
            raise KeyError(name)
        return getattr(self, name)

