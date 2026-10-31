"""One online step of the candidate filter.

The observable event decoder closes a candidate when its effect is complete.
Within that candidate, this module uses a normalized competing-risks process
to sum every possible first transition from an unresolved state to insertion,
normal, error, or correction.  No annotated phase, early-exit target, duration
threshold, or latency weight is introduced.  The accepted procedure frontier
is transported only after effect completion.

Two normalized observation laws are retained as a single latent variable for
the entire candidate: a procedure-conditioned law and an open-world visual
law.  Their equal prior is the posterior predictive law of an unobserved
Bernoulli variable under Jeffreys symmetry.  Marginalizing that variable is
part of the likelihood, not score fusion.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor




@dataclass(frozen=True)
class CandidateStep:
    """One completed candidate and all causally available warning scores."""

    explanation_probability: Tensor
    anomaly_probability: Tensor
    prefix_anomaly_probability: Tensor
    prefix_commitment_probability: Tensor
    normal_reference_probability: Tensor
    next_procedure_mode_lifecycle_probability: Tensor
    next_factual_history: Tensor
    prefix_hidden: Tensor

