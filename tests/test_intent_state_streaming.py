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
from provia.model.single_actor import ProviaSingleActorSharedSuccessor
from provia.model.ablation import (
    project_to_static_frontier,
    static_frontier_probability,
)
from provia.model.interval_transport import (
    ContinuationTransport,
)
from provia.model.bimanual import RoleIntentStateModel


def _model_and_transport() -> tuple[
    ProviaSingleActorSharedSuccessor,
    FactorizedCompletionTransport,
]:
    model = ProviaSingleActorSharedSuccessor(
        5,
        _labels(),
        ((0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.2, 0.1, 0.7)),
        (0.8, 0.1, 0.1),
        model_dim=12,
    ).eval()
    transport = FactorizedCompletionTransport(
        _branching_public_language(),
        0.1,
        posterior_predictive_innovation=True,
    )
    return model, transport






def test_observer_insertion_is_exactly_frontier_independent() -> None:
    torch.manual_seed(911)
    model, transport = _model_and_transport()
    stream = torch.randn(7, 5)
    evidence = model.energy.forward_mark(
        stream,
        0,
        _factors(7),
        torch.rand(7),
        model.energy.initial_history(stream[:1]),
        transport,
    )
    insertion = evidence.explanation_log_likelihood[:, :, 0]
    torch.testing.assert_close(
        insertion,
        insertion[:, :1].expand_as(insertion),
        atol=0,
        rtol=0,
    )


def test_packed_candidate_encoding_is_elementwise_exact() -> None:
    torch.manual_seed(916)
    model, transport = _model_and_transport()
    observed = (
        model.energy.window_encoder.forward_stream(torch.randn(7, 5)),
        model.energy.window_encoder.forward_stream(torch.randn(4, 5)),
        model.energy.window_encoder.forward_stream(torch.randn(9, 5)),
    )
    factors = tuple(_factors(len(value)) for value in observed)
    events = tuple(torch.rand(len(value)) for value in observed)
    packed = model.energy.encode_candidate_state_paths(
        observed, factors, events
    )
    history = model.energy.initial_history(torch.randn(1, 5))
    for encoded, factor, event, precomputed in zip(
        observed, factors, events, packed, strict=True
    ):
        direct = model.energy.forward_mark(
            encoded,
            0,
            factor,
            event,
            history,
            transport,
            encoded_prefix_hidden=encoded,
        )
        batched = model.energy.forward_mark(
            encoded,
            0,
            factor,
            event,
            history,
            transport,
            encoded_prefix_hidden=encoded,
            candidate_state_hidden=precomputed,
        )
        torch.testing.assert_close(batched.hidden, direct.hidden)
        torch.testing.assert_close(
            batched.explanation_log_likelihood,
            direct.explanation_log_likelihood,
        )
        torch.testing.assert_close(
            batched.node_log_bayes_factor,
            direct.node_log_bayes_factor,
        )


def test_coupled_candidate_is_future_invariant_and_normalized() -> None:
    torch.manual_seed(912)
    model, transport = _model_and_transport()
    stream = torch.randn(7, 5)
    changed = stream.clone()
    changed[5:] = 20.0 * torch.randn_like(changed[5:])
    factors = _factors(7)
    event = torch.rand(7)
    history = model.energy.initial_history(stream[:1])
    with torch.no_grad():
        first = model.energy.forward_mark(
            stream, 0, factors, event, history, transport
        )
        second = model.energy.forward_mark(
            changed, 0, factors, event, history, transport
        )
    torch.testing.assert_close(first.hidden[:5], second.hidden[:5])
    torch.testing.assert_close(
        first.explanation_log_likelihood[:5],
        second.explanation_log_likelihood[:5],
    )
    torch.testing.assert_close(
        first.node_log_bayes_factor[:5],
        second.node_log_bayes_factor[:5],
    )

    endpoint = {name: value[-1] for name, value in factors.items()}
    previous = model.filter.initial_belief(transport.initial_belief)
    output = model.forward_candidate(
        stream,
        0,
        factors,
        event,
        endpoint,
        previous,
        history,
        transport,
    )
    torch.testing.assert_close(
        output.explanation_probability.sum(), torch.tensor(1.0)
    )
    torch.testing.assert_close(
        output.next_procedure_mode_lifecycle_probability.sum(),
        torch.tensor(1.0),
    )


def test_constant_memory_streaming_is_exactly_prefix_equivalent() -> None:
    torch.manual_seed(917)
    model, transport = _model_and_transport()
    stream = torch.randn(8, 5)
    factors = _factors(len(stream))
    endpoint = {name: value[-1] for name, value in factors.items()}
    event = torch.rand(len(stream))
    history = model.energy.initial_history(stream[:1])
    belief = model.filter.initial_belief(transport.initial_belief)
    batch = model.forward_candidate(
        stream,
        0,
        factors,
        event,
        endpoint,
        belief,
        history,
        transport,
    )

    state = model.initial_stream_state(stream[0], transport)
    frontier_key = model.energy.build_frontier_key(transport)
    streamed_scores = []
    for index in range(len(stream)):
        completed = index == len(stream) - 1
        step = model.forward_stream_frame(
            stream[index],
            {name: value[index] for name, value in factors.items()},
            event[index],
            state,
            transport,
            completed=completed,
            endpoint_factor_probability=endpoint if completed else None,
            frontier_key=frontier_key,
        )
        streamed_scores.append(step.anomaly_probability)
        state = step.state

    torch.testing.assert_close(
        torch.stack(streamed_scores), batch.prefix_anomaly_probability
    )
    torch.testing.assert_close(
        step.explanation_probability, batch.explanation_probability
    )
    torch.testing.assert_close(
        state.procedure_mode_lifecycle_probability,
        batch.next_procedure_mode_lifecycle_probability,
    )
    torch.testing.assert_close(
        state.factual_history, batch.next_factual_history
    )
    assert state.candidate_recurrent_state is None


def test_frontier_free_candidate_and_constant_memory_recursions_are_exact() -> None:
    torch.manual_seed(922)
    model, transport = _model_and_transport()
    stream = torch.randn(8, 5)
    factors = _factors(len(stream))
    endpoint = {name: value[-1] for name, value in factors.items()}
    event = torch.rand(len(stream))
    history = model.energy.initial_history(stream[:1])
    belief = model.filter.initial_belief(transport.initial_belief)
    frontier = static_frontier_probability(transport, belief)
    belief = project_to_static_frontier(belief, frontier)
    batch = model.forward_candidate(
        stream,
        0,
        factors,
        event,
        endpoint,
        belief,
        history,
        transport,
        marginalize_procedure_frontier=True,
    )

    initial = model.initial_stream_state(stream[0], transport)
    state = type(initial)(
        procedure_mode_lifecycle_probability=belief,
        factual_history=initial.factual_history,
        candidate_recurrent_state=initial.candidate_recurrent_state,
    )
    frontier_key = model.energy.build_frontier_key(transport)
    streamed_scores = []
    for index in range(len(stream)):
        completed = index == len(stream) - 1
        step = model.forward_stream_frame(
            stream[index],
            {name: value[index] for name, value in factors.items()},
            event[index],
            state,
            transport,
            completed=completed,
            endpoint_factor_probability=endpoint if completed else None,
            frontier_key=frontier_key,
            marginalize_procedure_frontier=True,
        )
        streamed_scores.append(step.anomaly_probability)
        state = step.state

    torch.testing.assert_close(
        torch.stack(streamed_scores), batch.prefix_anomaly_probability
    )
    torch.testing.assert_close(
        state.procedure_mode_lifecycle_probability,
        batch.next_procedure_mode_lifecycle_probability,
    )
    torch.testing.assert_close(state.factual_history, batch.next_factual_history)










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






