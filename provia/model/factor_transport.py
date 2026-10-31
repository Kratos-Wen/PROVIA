"""Compositional assumed-density transport for uncertain event semantics.

An event observer returns distributions over verb, noun, and tool rather than a
reliable Cartesian argmax.  This module keeps those distributions throughout
the anomaly decision.  The accepted global/left/right states are posterior
moments of the last state-changing event.  An anomalous branch retains the old
moment; a progress branch replaces it with the current event posterior.  Their
mixture is projected back to the same moment family without a hard decision.
"""

from __future__ import annotations




FACTOR_ORDER = ("verb", "noun", "tool")