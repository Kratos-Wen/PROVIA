
import pytest
import torch

from tests.helpers import _factors, _labels
from tests.test_intent_set import _branching_public_language
from provia.model.commitment_filter import (
    INSERTION,
    ExhaustivePartitionCommitmentFilter,
)
from provia.model.completion_transport import FactorizedCompletionTransport
from provia.model.intent_state import (causal_candidate_commitment_probability)
from provia.model.ablation import (
    NO_CAUSAL_DECISION_CLOCK,
    NO_EXECUTION_MODE_PERSISTENCE,
    NO_INSERTION_STATE,
    configure_filter_ablation,
    configure_model_ablation,
)
from provia.model.single_actor import (ProviaSingleActorSharedSuccessor, jeffreys_partition_mode_probability, partition_mode_nll, partition_mode_target, prior_preserving_partition_mode_log_score)


TRANSITION = (
    (0.8, 0.1, 0.1),
    (0.1, 0.8, 0.1),
    (0.2, 0.1, 0.7),
)
INITIAL = (0.7, 0.2, 0.1)


def _model() -> ProviaSingleActorSharedSuccessor:
    return ProviaSingleActorSharedSuccessor(
        5, _labels(), TRANSITION, INITIAL, model_dim=12
    ).eval()


def _transport() -> FactorizedCompletionTransport:
    return FactorizedCompletionTransport(
        _branching_public_language(),
        0.1,
        posterior_predictive_innovation=True,
    )


def test_partition_filter_has_no_origin_prior_or_insertion_mass() -> None:
    model = ExhaustivePartitionCommitmentFilter(TRANSITION, INITIAL)
    assert "prior_genuine_probability" not in dict(model.named_buffers())
    belief = model.initial_belief(torch.tensor((0.6, 0.4)))
    step = model.begin_from_log_likelihood(torch.zeros(2, 3), belief)
    torch.testing.assert_close(
        step.explanation_probability[0], torch.tensor(0.0), atol=0, rtol=0
    )
    torch.testing.assert_close(step.explanation_probability.sum(), torch.tensor(1.0))
    torch.testing.assert_close(
        step.procedure_mode_lifecycle_probability[:, :, INSERTION].sum(),
        torch.tensor(0.0),
        atol=0,
        rtol=0,
    )
    expected = torch.tensor(INITIAL) @ torch.tensor(TRANSITION)
    torch.testing.assert_close(step.explanation_probability[1:], expected)








def test_partition_candidate_is_normalized_and_has_zero_compatibility_mass() -> None:
    torch.manual_seed(1401)
    model = _model()
    transport = _transport()
    stream = torch.randn(7, 5)
    factors = _factors(len(stream))
    endpoint = {name: value[-1] for name, value in factors.items()}
    event = torch.rand(len(stream))
    history = model.energy.initial_history(stream[:1])
    belief = model.filter.initial_belief(transport.initial_belief)
    output = model.forward_candidate(
        stream,
        0,
        factors,
        event,
        endpoint,
        belief,
        history,
        transport,
    )
    assert output.explanation_probability.shape == (4,)
    torch.testing.assert_close(
        output.explanation_probability[0], torch.tensor(0.0), atol=0, rtol=0
    )
    torch.testing.assert_close(
        output.explanation_probability.sum(), torch.tensor(1.0)
    )
    torch.testing.assert_close(
        output.next_procedure_mode_lifecycle_probability.sum(), torch.tensor(1.0)
    )
    torch.testing.assert_close(
        output.prefix_anomaly_probability[-1], output.anomaly_probability
    )


def test_partition_decision_clock_is_causal_and_parameter_free() -> None:
    hazard = torch.tensor((0.2, 0.25, 0.1))
    actual = causal_candidate_commitment_probability(hazard)
    torch.testing.assert_close(actual, torch.tensor((0.2, 0.4, 1.0)))

    torch.manual_seed(1402)
    model = _model()
    transport = _transport()
    stream = torch.randn(6, 5)
    factors = _factors(len(stream))
    endpoint = {name: value[-1] for name, value in factors.items()}
    output = model.forward_candidate(
        stream,
        0,
        factors,
        torch.zeros(len(stream)),
        endpoint,
        model.filter.initial_belief(transport.initial_belief),
        model.energy.initial_history(stream[:1]),
        transport,
    )
    torch.testing.assert_close(
        output.prefix_commitment_probability[:-1],
        torch.zeros(len(stream) - 1),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        output.prefix_anomaly_probability[:-1],
        torch.zeros(len(stream) - 1),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        output.prefix_commitment_probability[-1], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        output.prefix_anomaly_probability[-1], output.anomaly_probability
    )


def test_partition_targets_and_balanced_score_are_proper() -> None:
    probability = torch.tensor(
        ((0.0, 0.7, 0.2, 0.1), (0.0, 0.1, 0.8, 0.1), (0.0, 0.2, 0.1, 0.7)),
        requires_grad=True,
    )
    origin = torch.tensor((False, True, True))
    mode = torch.tensor((-1, 1, 2))
    target = partition_mode_target(
        origin, mode, correction_supervised=True
    )
    torch.testing.assert_close(target, torch.tensor((0, 1, 2)))
    nll = partition_mode_nll(
        probability, origin, mode, correction_supervised=True
    )
    expected = -torch.log(torch.tensor((0.7, 0.8, 0.7))).mean()
    torch.testing.assert_close(nll.detach(), expected)
    prior = jeffreys_partition_mode_probability(
        origin, mode, correction_supervised=True
    )
    numerator, exposure = prior_preserving_partition_mode_log_score(
        probability,
        origin,
        mode,
        prior,
        correction_supervised=True,
    )
    assert torch.isfinite(numerator / exposure)
    numerator.backward()
    assert probability.grad is not None
    assert torch.isfinite(probability.grad).all()


def test_topology_specific_ablation_contract_is_explicit() -> None:
    model = ExhaustivePartitionCommitmentFilter(TRANSITION, INITIAL)
    with pytest.raises(ValueError, match="no insertion state"):
        configure_filter_ablation(model, NO_INSERTION_STATE)
    configure_filter_ablation(model, NO_EXECUTION_MODE_PERSISTENCE)
    expected = torch.tensor(INITIAL).repeat(3, 1)
    torch.testing.assert_close(model.transition_probability, expected)
    completion = _model()
    configure_model_ablation(completion, NO_CAUSAL_DECISION_CLOCK)
    assert completion.use_causal_decision_clock is False
