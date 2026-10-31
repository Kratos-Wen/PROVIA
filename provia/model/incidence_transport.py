"""Competing incidence with effect-committed procedure transport.

An event-level label tells us which explanation became valid by the predicted
event completion, but not the first prefix at which that explanation was
causally resolved.  This module treats that time as interval censored.  At
each observed prefix, three causes compete to absorb still-unresolved mass:
observer insertion, execution error, and recovery/correction.  Correct normal
execution is the survivor at the predicted physical-effect boundary.

The construction has two useful consequences.  First, the event-completion
proper score trains exactly the same cumulative error probability that is
read online; no endpoint target is copied onto earlier prefixes and no latent
WAIT gate can postpone all learning until completion.  Second, rates are per
second and are integrated over observed time, so subdividing an interval does
not introduce an implicit frame-rate prior.

Procedure evidence is not fused as a second score.  The verified continuation
language supplies a normal-versus-open-world Bayes factor.  Its reciprocal is
the corresponding error-risk offset, with coefficient fixed by likelihood
algebra.  The accepted procedure frontier remains read-only until completion;
only surviving normal mass is conditionally transported, exactly once.
"""

from __future__ import annotations







INCIDENCE_CAUSE_NAMES = ("insertion", "error", "correction")
INSERTION_CAUSE, ERROR_CAUSE, CORRECTION_CAUSE = range(
    len(INCIDENCE_CAUSE_NAMES)
)

