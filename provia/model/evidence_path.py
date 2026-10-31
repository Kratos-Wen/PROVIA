"""Continuous-time evidence for conditional event marks.

An observed candidate is one marked renewal interval.  A terminal event label
does not identify an earlier decision time, so copying that label to every
prefix is invalid.  Conversely, a non-negative competing hazard makes an early
error suspicion irreversible.  This module uses neither construction.

The causal network emits a *signed* log-likelihood-rate vector.  Integrating
that vector over physical time gives the cumulative mark likelihood available
at each prefix.  Positive and negative increments allow later evidence to
revise an earlier interpretation, while the endpoint proper score trains every
causal observation in the interval without assigning it an endpoint label.
The procedure successor law enters once as the normalized likelihood of the
normal mark; it is not accumulated frame by frame or fused by a coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor






@dataclass(frozen=True)
class SignedEvidencePath:
    """Causal representation and integrated state-conditional evidence."""

    hidden: Tensor
    cumulative_log_likelihood_ratio: Tensor


__all__ = [
    "SignedEvidencePath",
]
