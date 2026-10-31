"""Role-preserving joint emission and transport inference.

Both hands share one candidate-state encoder and one successor relation.  For
each active hand, the normal relation to a latent legal successor is consumed
by the same exact operator that transports procedure state at completion.
Insertions remain independent of the shared latent frontier, and concurrent
completions retain the filter's exact order marginalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping, Sequence

import torch
from torch import Tensor

from provia.observer.factors import EVENT_FACTORS, EVENT_ROLES
from provia.observer.temporal import TemporalStreamState

from .mode_filter import ERROR, EXPLANATION_NAMES, NORMAL
from .step_effects import ExpertEffectPosterior
from .interval_transport import ContinuationTransport
from .role_intent_mark import RoleIntentConditionedMarkFilter
from .role_intent_set import (
    RoleIntentSetPredictiveStateMarkEnergy,
    RoleIntentSetModel,
)
from .intent_set import exact_intent_set_log_mixture
from .state_mark import validate_candidate_stream_state


PROVIA_BIMANUAL_ARCHITECTURE = (
    "provia_bimanual"
)


@dataclass(frozen=True)
class RoleCoupledIntentMarkEvidence:
    """One role's mark components at one or more causal prefixes."""

    explanation_log_likelihood: Tensor
    node_log_bayes_factor: Tensor


@dataclass(frozen=True)
class RoleCoupledIntentStreamState:
    """Constant-memory state of two asynchronous candidate streams."""

    procedure_mode_lifecycle_probability: Tensor
    factual_history: Tensor
    candidate_recurrent_state: tuple[
        Tensor | TemporalStreamState | None,
        Tensor | TemporalStreamState | None,
    ]


@dataclass(frozen=True)
class RoleCoupledIntentStreamStep:
    """One joint timestamp read; inactive roles have no emitted posterior."""

    explanation_probability: tuple[Tensor | None, Tensor | None]
    anomaly_probability: tuple[Tensor | None, Tensor | None]
    state: RoleCoupledIntentStreamState


class RoleCoupledIntentPredictiveStateMarkEnergy(
    RoleIntentSetPredictiveStateMarkEnergy
):
    """Shared bimanual state with one uncollapsed legal-successor variable."""

    def __init__(
        self,
        input_dim: int,
        factor_labels: Mapping[str, Sequence[str]],
        *,
        model_dim: int = 192,
        couple_successor: bool = True,
    ) -> None:
        super().__init__(input_dim, factor_labels, model_dim=model_dim)
        self.couple_successor = bool(couple_successor)

    def forward_mark(
        self,
        encoded_hidden: Tensor,
        factor_probability: Mapping[str, Tensor],
        event_probability: Tensor,
        factual_history: Tensor,
        role: int,
        transport: ContinuationTransport,
        *,
        frontier_key: Tensor | None = None,
        readout_log_bayes: Tensor | None = None,
    ) -> RoleCoupledIntentMarkEvidence:
        del factor_probability, event_probability
        if (
            encoded_hidden.ndim != 2
            or encoded_hidden.shape[-1] != self.model_dim
            or not len(encoded_hidden)
            or not torch.isfinite(encoded_hidden).all()
        ):
            raise ValueError("candidate predictive states are malformed")
        if factual_history.shape != (len(EVENT_ROLES), self.model_dim):
            raise ValueError("role factual history has the wrong shape")
        if not 0 <= int(role) < len(EVENT_ROLES):
            raise ValueError("role must select a fixed left/right slot")
        own = factual_history[int(role)]
        other = factual_history[1 - int(role)]
        role_index = torch.tensor(
            int(role), dtype=torch.long, device=encoded_hidden.device
        )
        query = self.state_norm(
            encoded_hidden
            + own.unsqueeze(0)
            + self.other_role_map(other).unsqueeze(0)
            + self.role_embedding(role_index).unsqueeze(0)
        )
        keys = (
            self.build_frontier_key(transport, role)
            if frontier_key is None
            else frontier_key
        ).to(query)
        states = transport.language.state_count
        nodes = len(transport.language.node_ids)
        if keys.shape != (states + nodes, self.model_dim):
            raise ValueError("role coupled-intent keys do not align with language")
        cause_query = self.mark_query(query).reshape(
            len(query), len(EXPLANATION_NAMES), self.model_dim
        )
        scale = sqrt(float(self.model_dim))
        state_log = torch.einsum(
            "sd,tkd->tsk", keys[:states], cause_query
        ) / scale
        node_log = torch.einsum(
            "nd,tkd->tnk", keys[states:], cause_query
        ) / scale
        intent_log = exact_intent_set_log_mixture(
            node_log,
            getattr(
                transport, f"frontier_node_probability_{int(role)}"
            ).to(node_log),
        )
        # Frontier-conditioned energies are shrunk by the language's Jeffreys
        # posterior-predictive data fraction, so sparsely demonstrated
        # frontiers defer to the frontier-agnostic visual base energy.
        reliability = getattr(transport, "state_structural_reliability", None)
        if reliability is not None:
            reliability = reliability.to(state_log).view(1, states, 1)
            state_log = state_log * reliability
            intent_log = intent_log * reliability
        base = self.mark_base(query)
        likelihood = base.unsqueeze(1) + state_log + intent_log
        likelihood[:, :, 0] = base[:, 0].unsqueeze(1)
        if self.couple_successor:
            likelihood[:, :, 1 + NORMAL] = (
                base[:, 1 + NORMAL].unsqueeze(1)
                + state_log[:, :, 1 + NORMAL]
            )
        if readout_log_bayes is not None:
            # Frozen cross-fitted convex readout of the candidate's running
            # mean (computed by the caller): calibrated log Bayes factor for
            # error against normal, added to the error coordinate for every
            # frontier.  No trained parameter; transport untouched.
            if readout_log_bayes.shape != (len(encoded_hidden),):
                raise ValueError("readout log Bayes factor is malformed")
            likelihood = likelihood.clone()
            likelihood[:, :, 1 + ERROR] = likelihood[:, :, 1 + ERROR] + readout_log_bayes.to(
                likelihood
            ).unsqueeze(1)
        return RoleCoupledIntentMarkEvidence(
            explanation_log_likelihood=likelihood,
            node_log_bayes_factor=node_log,
        )


class RoleCoupledIntentMarkFilter(RoleIntentConditionedMarkFilter):
    """Exact role filter without latent-frontier recentering."""

    @staticmethod
    def _validate_mark(log_likelihood: Tensor, states: int) -> Tensor:
        if (
            log_likelihood.shape != (states, len(EXPLANATION_NAMES))
            or not torch.isfinite(log_likelihood).all()
        ):
            raise ValueError("role mark likelihood must have shape [states,4]")
        return log_likelihood


class RoleIntentStateModel(
    RoleIntentSetModel
):
    """Bimanual observed-candidate inference with coupled successor transport."""

    posterior_normal_history = False

    def __init__(
        self,
        input_dim: int,
        factor_labels: Mapping[str, Sequence[str]],
        transition_probability: Tensor | Sequence[Sequence[float]],
        initial_probability: Tensor | Sequence[float],
        genuine_probability: Tensor | float,
        *,
        model_dim: int = 192,
    ) -> None:
        super().__init__(
            input_dim,
            factor_labels,
            transition_probability,
            initial_probability,
            genuine_probability,
            model_dim=model_dim,
        )
        self.energy = RoleCoupledIntentPredictiveStateMarkEnergy(
            input_dim, factor_labels, model_dim=model_dim
        )
        self.filter = RoleCoupledIntentMarkFilter(
            transition_probability,
            initial_probability,
            genuine_probability,
        )

    def completion_history(
        self,
        previous: Tensor,
        endpoint_hidden: Tensor,
        endpoint_factor_probability: Mapping[str, Tensor],
        role: int,
        explanation_probability: Tensor,
    ) -> Tensor:
        """Update one role history under the completion law."""

        return self.energy.update_history(
            previous,
            endpoint_hidden,
            endpoint_factor_probability,
            role,
        )

    def initial_stream_state(
        self,
        reference: Tensor,
        transport: ContinuationTransport,
    ) -> RoleCoupledIntentStreamState:
        """Initialize the exact online recursion before either role starts."""

        if reference.ndim < 1 or reference.shape[-1] != self.energy.input_dim:
            raise ValueError("role stream reference has the wrong feature dimension")
        initial_procedure = reference.new_zeros(transport.language.state_count)
        initial_procedure[transport.language.start_state] = 1.0
        belief = self.filter.initial_belief(initial_procedure)
        history = self.energy.initial_history(reference)
        return RoleCoupledIntentStreamState(
            procedure_mode_lifecycle_probability=belief,
            factual_history=history,
            candidate_recurrent_state=(None, None),
        )

    def forward_stream_frame(
        self,
        role_feature: Tensor,
        factor_probability: Mapping[str, Tensor],
        event_probability: Tensor,
        state: RoleCoupledIntentStreamState,
        transport: ContinuationTransport,
        *,
        started: Sequence[bool],
        completed: Sequence[bool],
        effect_statistic: Mapping[int, Tensor] | None = None,
        expert_effect: ExpertEffectPosterior | None = None,
        frontier_key: Sequence[Tensor] | None = None,
    ) -> RoleCoupledIntentStreamStep:
        """Advance both hand streams once without replaying either prefix.

        Candidate GRU states persist independently between their predicted
        starts and completions.  The shared procedure posterior and factual
        histories change only when one or both candidates complete.  If both
        complete at the same timestamp, the existing filter
        marginalizes both within-frame orders exactly.
        """

        roles = len(EVENT_ROLES)
        if role_feature.shape != (roles, self.energy.input_dim):
            raise ValueError("one role-stream update needs one feature per role")
        if event_probability.shape != (roles,) or (
            not torch.isfinite(event_probability).all()
            or torch.any((event_probability < 0) | (event_probability > 1))
        ):
            raise ValueError("role event probabilities must lie in [0,1]")
        if tuple(factor_probability) != EVENT_FACTORS:
            raise ValueError(f"factor probabilities must follow {EVENT_FACTORS}")
        for name in EVENT_FACTORS:
            expected = (roles, len(self.energy.factor_labels[name]))
            value = factor_probability[name]
            if value.shape != expected or not torch.isfinite(value).all():
                raise ValueError(f"{name} role probabilities have wrong shape")
        begin = tuple(bool(value) for value in started)
        end = tuple(bool(value) for value in completed)
        if len(begin) != roles or len(end) != roles:
            raise ValueError("start and completion flags must follow fixed roles")
        if len(state.candidate_recurrent_state) != roles:
            raise ValueError("stream state must retain one recurrent slot per role")
        belief = self.filter._validate(
            state.procedure_mode_lifecycle_probability
        )
        if state.factual_history.shape != (roles, self.energy.model_dim):
            raise ValueError("stream factual history has the wrong shape")
        if frontier_key is not None and len(frontier_key) != roles:
            raise ValueError("frontier keys must follow fixed roles")

        ending_roles = {role for role in range(roles) if end[role]}
        supplied_effects = set(() if effect_statistic is None else effect_statistic)
        if expert_effect is None:
            if supplied_effects:
                raise ValueError("effect statistics require an expert posterior")
        elif supplied_effects != ending_roles:
            raise ValueError(
                "expert endpoint statistics are required exactly at completion"
            )

        recurrent = list(state.candidate_recurrent_state)
        mark_updates: dict[int, Tensor] = {}
        observations: dict[int, tuple[Mapping[str, Tensor], Tensor | None]] = {}
        endpoint_hidden: dict[int, Tensor] = {}
        endpoint_factors: dict[int, Mapping[str, Tensor]] = {}
        active_roles = []
        for role in range(roles):
            was_active = recurrent[role] is not None
            if begin[role] and was_active:
                raise ValueError("a role cannot start a second active candidate")
            active = was_active or begin[role]
            if end[role] and not active:
                raise ValueError("an inactive role cannot complete a candidate")
            if not active:
                continue
            active_roles.append(role)
            previous_recurrent = recurrent[role]
            validate_candidate_stream_state(
                self.energy.candidate_state,
                previous_recurrent,
                model_dim=self.energy.model_dim,
            )
            encoded = self.energy.window_encoder.forward_stream(
                role_feature[role : role + 1]
            )
            factors = {
                name: factor_probability[name][role].to(encoded)
                for name in EVENT_FACTORS
            }
            factor_rows = {
                name: value.unsqueeze(0) for name, value in factors.items()
            }
            semantic = self.energy.semantic_expectation(factor_rows).to(encoded)
            event = self.energy.event_probability_map(
                event_probability[role].to(encoded).reshape(1, 1)
            )
            recurrent_input = self.energy.query_norm(encoded + semantic + event)
            hidden, following_recurrent = self.energy.candidate_state(
                recurrent_input.unsqueeze(0), previous_recurrent
            )
            hidden = hidden.squeeze(0)
            evidence = self.energy.forward_mark(
                hidden,
                factor_rows,
                event_probability[role : role + 1].to(encoded),
                state.factual_history,
                role,
                transport,
                frontier_key=(
                    None if frontier_key is None else frontier_key[role]
                ),
            )
            mark_updates[role] = evidence.explanation_log_likelihood[0]
            normal_node = evidence.node_log_bayes_factor[0, :, 1 + NORMAL]
            transport_node = self.transport_node_log_bayes_factor(normal_node)
            if end[role] and expert_effect is not None:
                statistic = effect_statistic[role]  # type: ignore[index]
                effect_node = self.node_effect_log_bayes_factor(
                    statistic.to(normal_node), expert_effect, transport
                )
                transport_node = (
                    effect_node
                    if transport_node is None
                    else transport_node + effect_node
                )
            observations[role] = (factors, transport_node)
            endpoint_hidden[role] = hidden[0]
            endpoint_factors[role] = factors
            recurrent[role] = following_recurrent

        explanations: list[Tensor | None] = [None, None]
        anomalies: list[Tensor | None] = [None, None]
        if active_roles:
            score_belief = self.filter.mark_joint(belief, mark_updates)
            score_belief = self.condition_for_readout(
                score_belief, observations, transport
            )
            for role in active_roles:
                explanation = self.filter.summarize(
                    score_belief, role
                ).explanation_probability
                explanations[role] = explanation
                anomalies[role] = explanation[1 + ERROR]

        history = state.factual_history
        if ending_roles:
            ending_marks = {
                role: mark_updates[role] for role in sorted(ending_roles)
            }
            ending_observations = {
                role: observations[role] for role in sorted(ending_roles)
            }
            marked = self.filter.mark_joint(belief, ending_marks)
            following_belief = self.filter.finalize_joint(
                marked, ending_observations, transport
            )
            for role in sorted(ending_roles):
                explanation = explanations[role]
                if explanation is None:
                    raise RuntimeError("completed role lacks its mark posterior")
                history = self.completion_history(
                    history,
                    endpoint_hidden[role],
                    endpoint_factors[role],
                    role,
                    explanation,
                )
                recurrent[role] = None
        else:
            following_belief = belief

        return RoleCoupledIntentStreamStep(
            explanation_probability=(explanations[0], explanations[1]),
            anomaly_probability=(anomalies[0], anomalies[1]),
            state=RoleCoupledIntentStreamState(
                procedure_mode_lifecycle_probability=following_belief,
                factual_history=history,
                candidate_recurrent_state=(recurrent[0], recurrent[1]),
            ),
        )




class ProviaBimanual(
    RoleIntentStateModel
):
    """PROVIA on bimanual streams, with two successor draws."""

    def __init__(
        self,
        input_dim: int,
        factor_labels: Mapping[str, Sequence[str]],
        transition_probability: Tensor | Sequence[Sequence[float]],
        initial_probability: Tensor | Sequence[float],
        genuine_probability: Tensor | float,
        *,
        model_dim: int = 192,
    ) -> None:
        super().__init__(
            input_dim,
            factor_labels,
            transition_probability,
            initial_probability,
            genuine_probability,
            model_dim=model_dim,
        )
        self.energy.couple_successor = False

    @staticmethod
    def transport_node_log_bayes_factor(normal_node: Tensor) -> None:
        del normal_node
        return None




__all__ = [
    "PROVIA_BIMANUAL_ARCHITECTURE",
    "RoleCoupledIntentMarkEvidence",
    "RoleCoupledIntentStreamState",
    "RoleCoupledIntentStreamStep",
    "RoleCoupledIntentMarkFilter",
    "RoleCoupledIntentPredictiveStateMarkEnergy",
    "RoleIntentStateModel",
    "ProviaBimanual",
]
