"""Bayesian reference applicability.

The query observer supplies uncertain verb/noun/tool distributions.  At every
candidate prefix, the verified expert continuation language and a uniform
open-world ontology define two normalized observation laws.  Their likelihood
ratio is marginalized with the Jeffreys posterior probability implied by the
number of verified reference traces.  Thus procedure compatibility,
observation uncertainty, and learned causal visual evidence belong to one
state-space posterior rather than to independently calibrated scores.  The
semantic likelihood ratio is used one-sidedly: it can reject an impossible
normal transition but semantic agreement alone cannot certify correct
execution.  Positive correctness evidence must come from causal visual
dynamics or the observed endpoint effect.

The first causally sufficient time remains latent and class agnostic.  A
decided explanation can still change under subsequent cumulative visual
evidence.  Only a reference-grounded normal branch advances the accepted
procedure frontier; an open-world normal branch applies the observer-miss
closure and stutters.  No applicability label, learned fusion gate, semantic
temperature, tuned threshold, or dataset-specific head is introduced.
"""

from __future__ import annotations







