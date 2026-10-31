"""Evidence-quality-derived reliability of frontier-conditioned energies.

The induced predictive language records how many reference observations
support each frontier's next-outcome law.  The Jeffreys posterior-predictive
data fraction of that law shrinks the learned frontier-conditioned emission
energies toward the frontier-agnostic base energy, so mode inference falls
back to direct visual evidence exactly where the demonstrated evidence is
thin.  Languages without evidence counts must reproduce the previous
construction bit for bit.
"""

from __future__ import annotations

import dataclasses

import torch

from tests.helpers import _labels
from tests.test_intent_set import _branching_public_language

from provia.model.predictive_state_induction import (
    quotient_interval_language,
)
from provia.model.completion_transport import FactorizedCompletionTransport
from provia.model.intent_state import (
    CoupledIntentPredictiveStateCompletionModel,
)
from provia.model.interval_transport import IntervalContinuationLanguage


def _model() -> CoupledIntentPredictiveStateCompletionModel:
    return CoupledIntentPredictiveStateCompletionModel(
        5,
        _labels(),
        ((0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.2, 0.1, 0.7)),
        (0.8, 0.1, 0.1),
        0.75,
        model_dim=12,
    ).eval()


def _language_with_counts(
    evidence: tuple[int, ...], outcomes: tuple[int, ...]
) -> IntervalContinuationLanguage:
    return dataclasses.replace(
        _branching_public_language(),
        state_evidence_count=evidence,
        state_outcome_count=outcomes,
    )




def test_language_serialization_round_trips_evidence_counts() -> None:
    language, _ = quotient_interval_language(
        _branching_public_language(), ((0, 1),)
    )
    encoded = language.to_dict()
    assert encoded["schema"] == "interval_continuation_language"
    decoded = IntervalContinuationLanguage.from_dict(encoded)
    assert decoded.state_evidence_count == language.state_evidence_count
    assert decoded.state_outcome_count == language.state_outcome_count


def test_transport_reliability_is_the_jeffreys_data_fraction() -> None:
    language = _language_with_counts((4, 1, 0, 2), (2, 2, 1, 1))
    transport = FactorizedCompletionTransport(
        language, 0.1, posterior_predictive_innovation=True
    )
    expected = torch.tensor(
        [4.0 / 5.0, 1.0 / 2.0, 0.0, 2.0 / 2.5], dtype=torch.float32
    )
    torch.testing.assert_close(
        transport.state_structural_reliability, expected
    )
    base = FactorizedCompletionTransport(
        _branching_public_language(), 0.1, posterior_predictive_innovation=True
    )
    torch.testing.assert_close(
        base.state_structural_reliability,
        torch.ones(4, dtype=torch.float32),
    )




