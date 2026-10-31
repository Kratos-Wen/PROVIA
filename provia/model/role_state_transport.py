"""Exact role-state transport for causal procedural event streams.

The observed event history and the procedure state are different random
objects.  Every event is factual and therefore enters the former.  For the
latter, the event has two mutually exclusive explanations: it advances the
accepted procedure state, or it is anomalous and leaves that state unchanged.
The filter retains both explanations and merges equal accepted states exactly;
it never commits with a score threshold.
"""

from __future__ import annotations



EventKey = tuple[str, str, str, str]