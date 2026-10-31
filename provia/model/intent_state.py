"""Joint emission--transport inference for streaming procedural marks.

The legal successor of a procedure state is latent.  It must not be
marginalized once to score a candidate and then independently marginalized a
second time to advance procedure state: those two operations would describe
two unrelated successor draws.  This module uses one successor variable for
both purposes.  Its learned normal-mark relation is multiplied into the
expert semantic/effect law before normalization, and the resulting posterior
over successors is reused by the completion transport.

The observation law also preserves a necessary conditional independence:
an observer insertion is independent of the latent procedure frontier.  We
therefore do not apply frontier-wise class centering.  Such centering is
posterior-preserving only when the frontier is observed; with a latent
frontier it changes its posterior and makes insertion evidence spuriously
state dependent.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping, Sequence

import torch
from torch import Tensor
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

from provia.observer.factors import EVENT_FACTORS
from provia.observer.temporal import TemporalStreamState

from .commitment_filter import LIFECYCLE_NAMES
from .intent_mark import IntentConditionedMarkedRenewalCompletionModel
from .completion_transport import FactorizedCompletionTransport
from .mode_filter import ERROR, MODE_NAMES, NORMAL
from .candidate_step import CandidateStep
from .step_effects import ExpertEffectPosterior
from .intent_set import (
    IntentSetPredictiveStateMarkEnergy,
    exact_intent_set_log_mixture,
)
from .ablation import marginalize_frontier_log_likelihood
from .state_mark import validate_candidate_stream_state


CAUSAL_CANDIDATE_DECISION_CLOCK = (
    "posterior_completion_commitment_times_error_probability"
)
MARGINAL_PREFIX_ERROR_READOUT = "marginal_prefix_error_probability_control"


@dataclass(frozen=True)
class CoupledIntentMarkEvidence:
    """Candidate likelihood components before the normal successor update."""

    hidden: Tensor
    explanation_log_likelihood: Tensor
    node_log_bayes_factor: Tensor


@dataclass(frozen=True)
class CoupledIntentStreamState:
    """Constant-memory state for exact frame-by-frame inference."""

    procedure_mode_lifecycle_probability: Tensor
    factual_history: Tensor
    candidate_recurrent_state: Tensor | TemporalStreamState | None
    candidate_survival_probability: Tensor | None = None


@dataclass(frozen=True)
class CoupledIntentStreamStep:
    """One causal mark read and the state carried to the next frame."""

    explanation_probability: Tensor
    anomaly_probability: Tensor
    normal_reference_probability: Tensor
    state: CoupledIntentStreamState


def causal_candidate_commitment_probability(
    event_probability: Tensor,
) -> Tensor:
    """Return the parameter-free posterior decision clock for one candidate.

    The frozen renewal observer supplies the conditional completion hazard at
    every frame. Its cumulative event probability is the probability that the
    current candidate has become committed by that prefix. The observer emits
    the candidate at the final prefix, so that boundary is a causal hard
    commitment and receives probability one.
    """

    if (
        event_probability.ndim != 1
        or not len(event_probability)
        or not torch.isfinite(event_probability).all()
        or torch.any((event_probability < 0) | (event_probability > 1))
    ):
        raise ValueError(
            "candidate event hazards must be a non-empty vector in [0,1]"
        )
    survival = torch.cumprod(1.0 - event_probability, dim=0)
    commitment = 1.0 - survival
    commitment = commitment.clone()
    commitment[-1] = 1.0
    return commitment


class CoupledIntentPredictiveStateMarkEnergy(
    IntentSetPredictiveStateMarkEnergy
):
    """One bounded candidate state with identity-preserving successor energy."""

    mode_likelihood_offset = 1
    has_insertion_likelihood = True

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

    def mode_likelihood_index(self, mode: int) -> int:
        """Map an execution mode to its learned emission coordinate."""

        if not 0 <= int(mode) < len(MODE_NAMES):
            raise ValueError("unknown execution mode")
        return self.mode_likelihood_offset + int(mode)

    def encode_candidate_state_paths(
        self,
        encoded_prefix_paths: Sequence[Tensor],
        prefix_factor_paths: Sequence[Mapping[str, Tensor]],
        prefix_event_paths: Sequence[Tensor],
    ) -> tuple[Tensor, ...]:
        """Encode independent candidates in one exact packed-GRU operation."""

        if (
            not encoded_prefix_paths
            or len(prefix_factor_paths) != len(encoded_prefix_paths)
            or len(prefix_event_paths) != len(encoded_prefix_paths)
        ):
            raise ValueError("batched candidates must be non-empty and aligned")
        recurrent_inputs = []
        lengths = []
        for observed, factors, event_probability in zip(
            encoded_prefix_paths,
            prefix_factor_paths,
            prefix_event_paths,
            strict=True,
        ):
            if (
                observed.ndim != 2
                or observed.shape[-1] != self.model_dim
                or not len(observed)
                or not torch.isfinite(observed).all()
                or event_probability.shape != (len(observed),)
                or not torch.isfinite(event_probability).all()
                or torch.any(
                    (event_probability < 0) | (event_probability > 1)
                )
            ):
                raise ValueError("batched candidate observations are malformed")
            semantic = self.semantic_expectation(factors).to(observed)
            event = self.event_probability_map(
                event_probability.to(observed).unsqueeze(-1)
            )
            recurrent_inputs.append(
                self.query_norm(observed + semantic + event)
            )
            lengths.append(len(observed))
        packed = pack_sequence(recurrent_inputs, enforce_sorted=False)
        encoded, _ = self.candidate_state(packed)
        padded, recovered_lengths = pad_packed_sequence(
            encoded, batch_first=True
        )
        if recovered_lengths.tolist() != lengths:
            raise RuntimeError("packed candidate order changed during encoding")
        return tuple(
            padded[index, :length].contiguous()
            for index, length in enumerate(lengths)
        )

    def forward_mark(
        self,
        causal_context: Tensor,
        prefix_offset: int,
        prefix_factor_probability: Mapping[str, Tensor],
        prefix_event_probability: Tensor,
        factual_history: Tensor,
        transport: FactorizedCompletionTransport,
        *,
        encoded_prefix_hidden: Tensor | None = None,
        candidate_state_hidden: Tensor | None = None,
        frontier_key: Tensor | None = None,
        readout_fold: int | None = None,
    ) -> CoupledIntentMarkEvidence:
        if factual_history.shape != (self.model_dim,):
            raise ValueError("factual history has the wrong shape")
        if encoded_prefix_hidden is None:
            if not 0 <= int(prefix_offset) < len(causal_context):
                raise ValueError("prefix offset is outside the causal context")
            observed = self.window_encoder.forward_stream(causal_context)[
                int(prefix_offset) :
            ]
        else:
            observed = encoded_prefix_hidden
        if (
            observed.ndim != 2
            or observed.shape[-1] != self.model_dim
            or not len(observed)
            or not torch.isfinite(observed).all()
        ):
            raise ValueError("encoded candidate observations are malformed")
        if (
            prefix_event_probability.shape != (len(observed),)
            or not torch.isfinite(prefix_event_probability).all()
            or torch.any(
                (prefix_event_probability < 0)
                | (prefix_event_probability > 1)
            )
        ):
            raise ValueError("prefix event probability must lie in [0,1]")

        if candidate_state_hidden is None:
            semantic = self.semantic_expectation(prefix_factor_probability).to(
                observed
            )
            event = self.event_probability_map(
                prefix_event_probability.to(observed).unsqueeze(-1)
            )
            recurrent_input = self.query_norm(observed + semantic + event)
            state, _ = self.candidate_state(recurrent_input.unsqueeze(0))
            state = state.squeeze(0)
        else:
            if (
                candidate_state_hidden.shape
                != (len(observed), self.model_dim)
                or not torch.isfinite(candidate_state_hidden).all()
            ):
                raise ValueError("pre-encoded candidate state is malformed")
            state = candidate_state_hidden
        query = self.state_norm(state + factual_history.unsqueeze(0))

        keys = (
            self.build_frontier_key(transport)
            if frontier_key is None
            else frontier_key
        ).to(query)
        states = transport.language.state_count
        nodes = len(transport.language.node_ids)
        if keys.shape != (states + nodes, self.model_dim):
            raise ValueError("coupled intent keys do not align with the language")
        mark_count = int(self.mark_base.out_features)
        expected_count = len(MODE_NAMES) + int(self.has_insertion_likelihood)
        if mark_count != expected_count:
            raise RuntimeError("mark head and candidate topology disagree")
        cause_query = self.mark_query(query).reshape(
            len(query), mark_count, self.model_dim
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
            transport.frontier_node_probability.to(node_log),
        )
        # Frontier-conditioned energies inherit the frontier's reference
        # support: the Jeffreys posterior-predictive data fraction from the
        # induced language shrinks them toward the frontier-agnostic base
        # exactly where the demonstrated evidence is thin, so mode inference
        # falls back to direct visual evidence instead of trusting an
        # under-demonstrated procedural context.  Languages without evidence
        # counts carry a reliability of one and reproduce the prior behavior.
        reliability = getattr(transport, "state_structural_reliability", None)
        if reliability is not None:
            reliability = reliability.to(state_log).view(1, states, 1)
            state_log = state_log * reliability
            intent_log = intent_log * reliability
        base = self.mark_base(query)
        likelihood = base.unsqueeze(1) + state_log + intent_log

        if self.has_insertion_likelihood:
            # Insertions are generated by the observer, not by a procedure state.
            likelihood[:, :, 0] = base[:, 0].unsqueeze(1)
        if self.couple_successor:
            # The normal successor relation is not marginalized here.  It is
            # consumed jointly by the normal emission and completion transport.
            normal_index = self.mode_likelihood_index(NORMAL)
            likelihood[:, :, normal_index] = (
                base[:, normal_index].unsqueeze(1)
                + state_log[:, :, normal_index]
            )
        return CoupledIntentMarkEvidence(
            hidden=state,
            explanation_log_likelihood=likelihood,
            node_log_bayes_factor=node_log,
        )


class CoupledIntentPredictiveStateCompletionModel(
    IntentConditionedMarkedRenewalCompletionModel
):
    """Observed-candidate inference with coupled latent-intent transport."""

    use_causal_decision_clock = True

    def completion_history(
        self,
        previous: Tensor,
        endpoint_hidden: Tensor,
        endpoint_factor_probability: Mapping[str, Tensor],
        explanation_probability: Tensor,
    ) -> Tensor:
        """Update candidate history under the completion law."""

        return self.energy.update_history(
            previous, endpoint_hidden, endpoint_factor_probability
        )

    @staticmethod
    def transport_node_log_bayes_factor(normal_node: Tensor) -> Tensor | None:
        """Select learned successor evidence shared with state transport."""

        return normal_node

    def initial_stream_state(
        self,
        reference: Tensor,
        transport: FactorizedCompletionTransport,
    ) -> CoupledIntentStreamState:
        """Initialize the exact online recursion before the first candidate."""

        if reference.ndim not in {1, 2} or reference.shape[-1] != self.energy.input_dim:
            raise ValueError("stream reference has the wrong feature dimension")
        belief = self.filter.initial_belief(
            transport.initial_belief.to(reference)
        )
        history = self.energy.initial_history(reference)
        return CoupledIntentStreamState(
            procedure_mode_lifecycle_probability=belief,
            factual_history=history,
            candidate_recurrent_state=None,
            candidate_survival_probability=None,
        )

    def forward_stream_frame(
        self,
        frame_feature: Tensor,
        factor_probability: Mapping[str, Tensor],
        event_probability: Tensor,
        state: CoupledIntentStreamState,
        transport: FactorizedCompletionTransport,
        *,
        completed: bool,
        endpoint_factor_probability: Mapping[str, Tensor] | None = None,
        effect_statistic: Tensor | None = None,
        expert_effect: ExpertEffectPosterior | None = None,
        frontier_key: Tensor | None = None,
        marginalize_procedure_frontier: bool = False,
    ) -> CoupledIntentStreamStep:
        """Read one causal frame without replaying the candidate prefix.

        The candidate GRU state is retained while a predicted event is active.
        Procedure belief and factual history remain fixed during that event and
        are changed exactly once, at its predicted completion.  This recursion
        is algebraically identical to :meth:`forward_candidate`.
        """

        if frame_feature.ndim != 1 or frame_feature.shape[0] != self.energy.input_dim:
            raise ValueError("one stream update requires one feature vector")
        if event_probability.numel() != 1 or not torch.isfinite(event_probability).all():
            raise ValueError("event probability must be one finite scalar")
        event_probability = event_probability.reshape(())
        if event_probability < 0 or event_probability > 1:
            raise ValueError("event probability must lie in [0,1]")
        if (effect_statistic is None) != (expert_effect is None):
            raise ValueError("effect statistic and expert posterior are paired")
        if expert_effect is not None and not completed:
            raise ValueError("endpoint effects are unavailable before completion")
        if completed != (endpoint_factor_probability is not None):
            raise ValueError("endpoint semantics are required exactly at completion")
        if state.procedure_mode_lifecycle_probability.ndim != 3 or (
            state.procedure_mode_lifecycle_probability.shape[1:]
            != (len(MODE_NAMES), len(LIFECYCLE_NAMES))
        ):
            raise ValueError("stream procedure belief has incompatible latent axes")
        if state.procedure_mode_lifecycle_probability[:, :, 1:].sum() > 2e-6:
            raise ValueError("a streaming candidate requires an inactive lifecycle")
        if state.factual_history.shape != (self.energy.model_dim,):
            raise ValueError("stream factual history has the wrong shape")
        recurrent = state.candidate_recurrent_state
        validate_candidate_stream_state(
            self.energy.candidate_state,
            recurrent,
            model_dim=self.energy.model_dim,
        )
        if self.use_causal_decision_clock:
            survival = state.candidate_survival_probability
            if survival is None:
                if recurrent is not None:
                    raise ValueError(
                        "an active candidate lacks its decision-clock state"
                    )
                survival = event_probability.new_tensor(1.0)
            elif (
                survival.numel() != 1
                or not torch.isfinite(survival).all()
                or survival < 0
                or survival > 1
            ):
                raise ValueError(
                    "candidate survival probability must be one scalar in [0,1]"
                )
            survival = survival.reshape(()) * (1.0 - event_probability)
            commitment = (
                event_probability.new_tensor(1.0)
                if completed
                else 1.0 - survival
            )
        else:
            survival = None
            commitment = event_probability.new_tensor(1.0)

        observed = self.energy.window_encoder.forward_stream(
            frame_feature.unsqueeze(0)
        )
        current_factor = (
            endpoint_factor_probability if completed else factor_probability
        )
        if current_factor is None or tuple(current_factor) != EVENT_FACTORS:
            raise ValueError(f"factor probabilities must follow {EVENT_FACTORS}")
        causal_factor: dict[str, Tensor] = {}
        for name in EVENT_FACTORS:
            probability = current_factor[name]
            expected = len(self.energy.factor_labels[name])
            if probability.shape != (expected,):
                raise ValueError(f"{name} stream probability has wrong shape")
            causal_factor[name] = probability.to(observed).unsqueeze(0)
        semantic = self.energy.semantic_expectation(causal_factor).to(observed)
        event = self.energy.event_probability_map(
            event_probability.to(observed).reshape(1, 1)
        )
        recurrent_input = self.energy.query_norm(observed + semantic + event)
        candidate_hidden, next_recurrent = self.energy.candidate_state(
            recurrent_input.unsqueeze(0), recurrent
        )
        candidate_hidden = candidate_hidden.squeeze(0)
        evidence = self.energy.forward_mark(
            frame_feature.unsqueeze(0),
            0,
            causal_factor,
            event_probability.to(observed).reshape(1),
            state.factual_history,
            transport,
            encoded_prefix_hidden=observed,
            candidate_state_hidden=candidate_hidden,
            frontier_key=frontier_key,
        )
        normal_node = evidence.node_log_bayes_factor[
            :, :, self.energy.mode_likelihood_index(NORMAL)
        ]
        transport_node = self.transport_node_log_bayes_factor(normal_node)
        endpoint_node = None if transport_node is None else transport_node[0]
        mixed, reference, _, _ = transport.predictive_reference_statistics(
            causal_factor,
            node_effect_log_bayes_factor=transport_node,
        )
        if expert_effect is not None:
            if expert_effect.node_ids != tuple(transport.language.node_ids):
                raise ValueError("expert effects and continuation nodes disagree")
            if effect_statistic is None or effect_statistic.ndim != 1:
                raise ValueError("endpoint effect statistic must be one vector")
            effect_node = expert_effect.node_log_bayes_factor(
                effect_statistic.to(normal_node)
            )[0]
            endpoint_node = (
                effect_node
                if endpoint_node is None
                else endpoint_node + effect_node
            )
            mixed, reference, _, _ = transport.predictive_reference_statistics(
                causal_factor,
                node_effect_log_bayes_factor=endpoint_node.unsqueeze(0),
            )
        likelihood = evidence.explanation_log_likelihood[0].clone()
        likelihood[:, self.energy.mode_likelihood_index(NORMAL)] += mixed[0].log()
        if marginalize_procedure_frontier:
            likelihood = marginalize_frontier_log_likelihood(
                likelihood,
                state.procedure_mode_lifecycle_probability.sum(dim=(1, 2)),
            )
        proposal = self.filter.begin_from_log_likelihood(
            likelihood,
            state.procedure_mode_lifecycle_probability
            / state.procedure_mode_lifecycle_probability.sum(),
        )
        explanation = proposal.explanation_probability

        if completed:
            def advance(source: Tensor) -> Tensor:
                destination = source if marginalize_procedure_frontier else (
                    transport.conditional_predictive_advance_probability_batch(
                        source.unsqueeze(0),
                        causal_factor,
                        node_effect_log_bayes_factor=endpoint_node,
                    )[0]
                )
                if not torch.allclose(
                    destination.sum(), source.sum(), atol=4e-5, rtol=4e-5
                ):
                    raise RuntimeError(
                        "coupled normal transport changed probability mass"
                    )
                return destination

            following = self.filter.finalize(
                proposal.procedure_mode_lifecycle_probability, advance
            )
            history = self.completion_history(
                state.factual_history,
                evidence.hidden[0],
                endpoint_factor_probability,
                explanation,
            )
            next_recurrent_state = None
            next_survival = None
        else:
            following = state.procedure_mode_lifecycle_probability
            history = state.factual_history
            next_recurrent_state = next_recurrent
            next_survival = survival

        reference_probability = 1.0 - transport.innovation_probability.to(mixed)
        posterior_reference = (
            reference_probability * reference[0] / mixed[0]
        ).clamp(0.0, 1.0)
        return CoupledIntentStreamStep(
            explanation_probability=explanation,
            anomaly_probability=commitment * explanation[1 + ERROR],
            normal_reference_probability=posterior_reference,
            state=CoupledIntentStreamState(
                procedure_mode_lifecycle_probability=following,
                factual_history=history,
                candidate_recurrent_state=next_recurrent_state,
                candidate_survival_probability=next_survival,
            ),
        )

    def forward_candidate(
        self,
        causal_context: Tensor,
        prefix_offset: int,
        prefix_factor_probability: Mapping[str, Tensor],
        prefix_event_probability: Tensor,
        endpoint_factor_probability: Mapping[str, Tensor],
        previous_belief: Tensor,
        factual_history: Tensor,
        transport: FactorizedCompletionTransport,
        *,
        effect_statistic: Tensor | None = None,
        expert_effect: ExpertEffectPosterior | None = None,
        prefix_displacement: Tensor | None = None,
        prefix_elapsed_seconds: Tensor | None = None,
        encoded_prefix_hidden: Tensor | None = None,
        candidate_state_hidden: Tensor | None = None,
        frontier_key: Tensor | None = None,
        endpoint_only: bool = False,
        marginalize_procedure_frontier: bool = False,
        readout_fold: int | None = None,
    ) -> CandidateStep:
        if (effect_statistic is None) != (expert_effect is None):
            raise ValueError("effect statistic and expert posterior are paired")
        if (prefix_displacement is None) != (prefix_elapsed_seconds is None) or (
            (prefix_displacement is None) != True
        ):
            raise ValueError("prefix path statistic and expert posterior are paired")
        if previous_belief.ndim != 3 or previous_belief.shape[1:] != (
            len(MODE_NAMES),
            len(LIFECYCLE_NAMES),
        ):
            raise ValueError("candidate prior has incompatible latent axes")
        if previous_belief[:, :, 1:].sum() > 2e-6:
            raise ValueError("a new candidate requires an inactive lifecycle")
        previous_belief = previous_belief / previous_belief.sum()
        prefix_commitment = (
            causal_candidate_commitment_probability(prefix_event_probability)
            if self.use_causal_decision_clock
            else torch.ones_like(prefix_event_probability)
        )

        causal_factor = self._endpoint_causal_factors(
            prefix_factor_probability, endpoint_factor_probability
        )
        evidence = self.energy.forward_mark(
            causal_context,
            prefix_offset,
            causal_factor,
            prefix_event_probability,
            factual_history,
            transport,
            encoded_prefix_hidden=encoded_prefix_hidden,
            candidate_state_hidden=candidate_state_hidden,
            frontier_key=frontier_key,
            readout_fold=readout_fold,
        )
        if endpoint_only:
            evidence = CoupledIntentMarkEvidence(
                hidden=evidence.hidden[-1:].contiguous(),
                explanation_log_likelihood=(
                    evidence.explanation_log_likelihood[-1:].contiguous()
                ),
                node_log_bayes_factor=(
                    evidence.node_log_bayes_factor[-1:].contiguous()
                ),
            )
            causal_factor = {
                name: value[-1:].contiguous()
                for name, value in causal_factor.items()
            }
            prefix_commitment = prefix_commitment[-1:].contiguous()

        normal_node = evidence.node_log_bayes_factor[
            :, :, self.energy.mode_likelihood_index(NORMAL)
        ]
        transport_node = self.transport_node_log_bayes_factor(normal_node)
        mixed, reference, _, _ = transport.predictive_reference_statistics(
            causal_factor,
            node_effect_log_bayes_factor=transport_node,
        )
        endpoint_factor = {
            name: causal_factor[name][-1:].contiguous()
            for name in EVENT_FACTORS
        }
        endpoint_node = None if transport_node is None else transport_node[-1]
        if expert_effect is not None:
            if expert_effect.node_ids != tuple(transport.language.node_ids):
                raise ValueError("expert effects and continuation nodes disagree")
            if effect_statistic is None or effect_statistic.ndim != 1:
                raise ValueError("endpoint effect statistic must be one vector")
            effect_node = expert_effect.node_log_bayes_factor(
                effect_statistic.to(normal_node)
            )[0]
            endpoint_node = (
                effect_node
                if endpoint_node is None
                else endpoint_node + effect_node
            )
            endpoint_mixed, endpoint_reference, _, _ = (
                transport.predictive_reference_statistics(
                    endpoint_factor,
                    node_effect_log_bayes_factor=endpoint_node,
                )
            )
            endpoint_mixed = endpoint_mixed[0]
            endpoint_reference = endpoint_reference[0]
            mixed = mixed.clone()
            reference = reference.clone()
            mixed[-1] = endpoint_mixed
            reference[-1] = endpoint_reference
        else:
            endpoint_mixed = mixed[-1]
            endpoint_reference = reference[-1]

        likelihood = evidence.explanation_log_likelihood.clone()
        likelihood[:, :, self.energy.mode_likelihood_index(NORMAL)] += mixed.log()
        if marginalize_procedure_frontier:
            likelihood = marginalize_frontier_log_likelihood(
                likelihood, previous_belief.sum(dim=(1, 2))
            )
        proposal_belief, prefix_explanation = (
            self.filter.begin_many_from_log_likelihood(
                likelihood, previous_belief
            )
        )
        prefix_anomaly = (
            prefix_commitment.to(prefix_explanation)
            * prefix_explanation[:, 1 + ERROR]
        )
        prefix_committed = prefix_commitment.to(prefix_explanation)

        def advance(source: Tensor) -> Tensor:
            destination = source if marginalize_procedure_frontier else (
                transport.conditional_predictive_advance_probability_batch(
                    source.unsqueeze(0),
                    endpoint_factor,
                    node_effect_log_bayes_factor=endpoint_node,
                )[0]
            )
            if not torch.allclose(
                destination.sum(), source.sum(), atol=4e-5, rtol=4e-5
            ):
                raise RuntimeError("coupled normal transport changed probability mass")
            return destination

        following = self.filter.finalize(
            proposal_belief[-1], advance
        )
        if following[:, :, 1:].sum() > 2e-6:
            raise RuntimeError("completed mark retained an active lifecycle")
        next_history = self.completion_history(
            factual_history,
            evidence.hidden[-1],
            endpoint_factor_probability,
            prefix_explanation[-1],
        )
        reference_probability = 1.0 - transport.innovation_probability.to(
            endpoint_mixed
        )
        posterior_reference = (
            reference_probability * endpoint_reference / endpoint_mixed
        ).clamp(0.0, 1.0)
        explanation = prefix_explanation[-1]
        return CandidateStep(
            explanation_probability=explanation,
            anomaly_probability=explanation[1 + ERROR],
            prefix_anomaly_probability=prefix_anomaly,
            prefix_commitment_probability=prefix_committed,
            normal_reference_probability=posterior_reference,
            next_procedure_mode_lifecycle_probability=following,
            next_factual_history=next_history,
            prefix_hidden=evidence.hidden,
        )


__all__ = [
    "CAUSAL_CANDIDATE_DECISION_CLOCK",
    "MARGINAL_PREFIX_ERROR_READOUT",
    "CoupledIntentMarkEvidence",
    "CoupledIntentStreamState",
    "CoupledIntentStreamStep",
    "CoupledIntentPredictiveStateCompletionModel",
    "CoupledIntentPredictiveStateMarkEnergy",
    "causal_candidate_commitment_probability",
]
