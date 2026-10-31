"""Decoded step events of the online step segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping




@dataclass(frozen=True)
class DecodedEvent:
    role: int
    start_index: int
    onset_index: int
    end_index: int
    factor_log_probability: Mapping[str, tuple[float, ...]]
    right_censored: bool = False

