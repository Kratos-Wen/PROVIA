"""Set-valued residual-effect cancellation for procedural recovery.

An execution error is not committed to procedural progress.  Its uncertain
intent is retained as a residual.  A later event is safe either because it
advances a currently viable procedure node or because it provides a
role-preserving inverse explanation of that residual.  The two explanations
partition event state into forward progress, residual cancellation, and
unresolved violation; no independent anomaly or recovery classifier is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping




@dataclass(frozen=True)
class RecoveryRelationStructure:
    factor_labels: Mapping[str, tuple[str, ...]]
    verb_possibility: tuple[tuple[float, ...], ...]
    source_pair_count: int
    entity_alignment_possibility: tuple[tuple[float, ...], ...] = (
        (1.0, 1.0),
        (1.0, 1.0),
    )
