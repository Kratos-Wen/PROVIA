from __future__ import annotations

import torch

from tests.helpers import (_labels)
from provia.model.intent_set import (exact_intent_set_log_mixture)
from provia.model.interval_transport import (IntervalContinuationLanguage)


def _branching_public_language() -> IntervalContinuationLanguage:
    return IntervalContinuationLanguage(
        factor_labels=_labels(),
        node_ids=("left|a|x", "left|b|y"),
        node_roles=("left", "left"),
        node_possibility={
            "verb": ((1.0, 0.0), (0.0, 1.0)),
            "noun": ((1.0, 0.0), (0.0, 1.0)),
            "tool": ((1.0,), (1.0,)),
        },
        state_count=4,
        start_state=0,
        accepting_states=(3,),
        transitions=((0, 0, 1), (0, 1, 2), (1, 1, 3), (2, 0, 3)),
        source_trace_count=2,
        transition_probability=(0.5, 0.5, 1.0, 1.0),
    )


def test_exact_intent_mixture_matches_enumeration_and_is_permutation_invariant() -> None:
    torch.manual_seed(901)
    node_log = torch.randn(3, 4, 4, dtype=torch.float64)
    prior = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    actual = exact_intent_set_log_mixture(node_log, prior)
    expected = torch.stack(
        [
            torch.logsumexp(
                node_log[:, :, explanation]
                + prior[0].log().unsqueeze(0),
                dim=1,
            )
            for explanation in range(4)
        ],
        dim=-1,
    )
    torch.testing.assert_close(actual[:, 0], expected)
    torch.testing.assert_close(actual[:, 1], torch.zeros_like(actual[:, 1]))
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = exact_intent_set_log_mixture(
        node_log.index_select(1, permutation),
        prior.index_select(1, permutation),
    )
    torch.testing.assert_close(actual, permuted)


def test_intent_mixture_preserves_alternatives_instead_of_mean_pooling() -> None:
    node_log = torch.tensor([[[4.0], [-4.0]]])
    prior = torch.tensor([[0.5, 0.5]])
    mixture = exact_intent_set_log_mixture(node_log, prior)
    mean_pooled_logit = (prior @ node_log[0]).reshape(())
    assert mixture.reshape(()) > mean_pooled_logit + 2.0





