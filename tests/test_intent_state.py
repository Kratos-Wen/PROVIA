from __future__ import annotations


import torch

from tests.helpers import (
    _factors,
    _labels,
    _language,
)
from tests.test_intent_set import (
    _branching_public_language,
)
from provia.model.completion_transport import FactorizedCompletionTransport
from provia.model.mode_filter import NORMAL
from provia.model.intent_state import (CoupledIntentPredictiveStateCompletionModel)
from provia.model.interval_transport import (
    ContinuationTransport,
)
from provia.model.bimanual import (RoleIntentStateModel)


def _model_and_transport() -> tuple[
    CoupledIntentPredictiveStateCompletionModel,
    FactorizedCompletionTransport,
]:
    model = CoupledIntentPredictiveStateCompletionModel(
        5,
        _labels(),
        ((0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.2, 0.1, 0.7)),
        (0.8, 0.1, 0.1),
        0.75,
        model_dim=12,
    ).eval()
    transport = FactorizedCompletionTransport(
        _branching_public_language(),
        0.1,
        posterior_predictive_innovation=True,
    )
    return model, transport
























def test_role_coupling_preserves_insertion_and_atomic_mass() -> None:
    torch.manual_seed(914)
    model = RoleIntentStateModel(
        6,
        _labels(),
        ((0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.2, 0.1, 0.7)),
        (0.8, 0.1, 0.1),
        0.75,
        model_dim=12,
    ).eval()
    transport = ContinuationTransport(
        _language(), posterior_predictive_innovation=True
    )
    observed = model.energy.window_encoder.forward_stream(torch.randn(5, 6))
    factors = _factors(5)
    path = model.energy.forward_state_path(observed, factors, torch.rand(5))
    history = model.energy.initial_history(path[:1])
    evidence = model.energy.forward_mark(
        path[-1:],
        {name: value[-1:] for name, value in factors.items()},
        torch.ones(1),
        history,
        0,
        transport,
    )
    insertion = evidence.explanation_log_likelihood[0, :, 0]
    torch.testing.assert_close(
        insertion,
        insertion[:1].expand_as(insertion),
        atol=0,
        rtol=0,
    )
    node = evidence.node_log_bayes_factor[0, :, 1 + NORMAL]
    procedure = torch.zeros(transport.language.state_count)
    procedure[transport.language.start_state] = 1.0
    belief = model.filter.initial_belief(procedure)
    marked = model.filter.mark_joint(
        belief, {0: evidence.explanation_log_likelihood[0]}
    )
    endpoint_factor = {name: value[-1] for name, value in factors.items()}
    conditioned = model.condition_for_readout(
        marked, {0: (endpoint_factor, node)}, transport
    )
    following = model.filter.finalize_joint(
        marked, {0: (endpoint_factor, node)}, transport
    )
    torch.testing.assert_close(conditioned.sum(), torch.tensor(1.0))
    torch.testing.assert_close(following.sum(), torch.tensor(1.0))


