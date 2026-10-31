"""Exact factorized continuation transport for single-actor completions.

Completion-only benchmarks project every observable action onto one structural
actor.  The generic bimanual transport remains the reference implementation,
but materializing its full verb--noun--tool Cartesian support for every
automaton state is unnecessary in this reduction.  This module evaluates the
same probability partition while factorizing single-successor states and
materializing joint support only at genuine procedure branches.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Mapping

import numpy as np
import torch
from torch import Tensor, nn

from provia.observer.factors import EVENT_FACTORS

from .interval_transport import (
    IntervalContinuationLanguage,
    exact_missed_event_closure,
)


class FactorizedCompletionTransport(nn.Module):
    """Numerically exact single-actor reduction of continuation transport."""

    def __init__(
        self,
        language: IntervalContinuationLanguage,
        missed_event_probability: float,
        *,
        posterior_predictive_innovation: bool = False,
    ) -> None:
        super().__init__()
        language.validate()
        deletion = float(missed_event_probability)
        if not np.isfinite(deletion) or not 0.0 <= deletion <= 1.0:
            raise ValueError("missed-event probability must lie in [0,1]")
        roles = set(language.node_roles)
        if len(roles) != 1:
            raise ValueError("completion transport requires exactly one active actor")
        self.active_role = next(iter(roles))
        self.language = language
        self.missed_event_probability = deletion
        innovation = (
            0.5 / (float(language.source_trace_count) + 1.0)
            if posterior_predictive_innovation
            else 0.0
        )
        self.register_buffer(
            "innovation_probability",
            torch.tensor(innovation, dtype=torch.float32),
            persistent=False,
        )
        self.factor_labels = {
            name: tuple(language.factor_labels[name]) for name in EVENT_FACTORS
        }

        node_factor_possibility = {}
        for name in EVENT_FACTORS:
            values = torch.tensor(
                language.node_possibility[name], dtype=torch.float32
            )
            values = values / values.amax(dim=-1, keepdim=True).clamp_min(
                torch.finfo(values.dtype).tiny
            )
            self.register_buffer(
                f"node_{name}_possibility", values, persistent=False
            )
            node_factor_possibility[name] = values

        edges = torch.tensor(language.transitions, dtype=torch.long)
        self.register_buffer("edge_source", edges[:, 0], persistent=False)
        self.register_buffer("edge_node", edges[:, 1], persistent=False)
        self.register_buffer("edge_target", edges[:, 2], persistent=False)

        outgoing_edges: list[list[int]] = [
            [] for _ in range(language.state_count)
        ]
        outgoing_nodes: dict[int, set[int]] = defaultdict(set)
        for edge_index, (source, node, target) in enumerate(language.transitions):
            outgoing_edges[source].append(edge_index)
            outgoing_nodes[source].add(node)
        self.register_buffer(
            "direct_successor_node_count",
            torch.tensor(
                [len(outgoing_nodes.get(state, ())) for state in range(language.state_count)],
                dtype=torch.long,
            ),
            persistent=False,
        )
        # Frontier-conditioned learned evidence is trustworthy exactly to the
        # extent that the frontier's predictive law is supported by reference
        # observations.  The Jeffreys posterior-predictive data fraction
        # N_s / (N_s + alpha * C_s) is the linear shrinkage of that law toward
        # its symmetric prior and is reused here as the structural-evidence
        # reliability; languages compiled without evidence counts keep the
        # previous behavior through a reliability of one.
        if language.state_evidence_count is not None:
            evidence = torch.tensor(
                language.state_evidence_count, dtype=torch.float32
            )
            outcomes = torch.tensor(
                language.state_outcome_count, dtype=torch.float32
            )
            reliability = evidence / (evidence + 0.5 * outcomes)
        else:
            reliability = torch.ones(language.state_count, dtype=torch.float32)
        self.register_buffer(
            "state_structural_reliability", reliability, persistent=False
        )
        outdegree = torch.tensor([len(values) for values in outgoing_edges], dtype=torch.float32)
        edge_prior = (
            torch.tensor(language.transition_probability, dtype=torch.float32)
            if language.transition_probability is not None
            else outdegree.index_select(0, self.edge_source).reciprocal()
        )
        self.register_buffer(
            "edge_prior_probability",
            edge_prior,
            persistent=False,
        )

        single_states = []
        single_nodes = []
        branch_states = []
        branch_nodes = []
        for state in range(language.state_count):
            nodes = tuple(sorted(outgoing_nodes.get(state, ())))
            if len(nodes) == 1:
                single_states.append(state)
                single_nodes.append(nodes[0])
            elif len(nodes) > 1:
                branch_states.append(state)
                branch_nodes.append(nodes)
        self.register_buffer(
            "single_states", torch.tensor(single_states, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "single_nodes", torch.tensor(single_nodes, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "branch_states", torch.tensor(branch_states, dtype=torch.long), persistent=False
        )

        factor_indices = tuple(
            product(
                *(
                    range(len(self.factor_labels[name]))
                    for name in EVENT_FACTORS
                )
            )
        )
        index_tensor = torch.tensor(factor_indices, dtype=torch.long)
        self.register_buffer("factor_indices", index_tensor, persistent=False)
        branch_coverage = torch.zeros(
            len(branch_states), len(factor_indices), dtype=torch.float32
        )
        for branch_index, nodes in enumerate(branch_nodes):
            for node in nodes:
                possibility = torch.ones(len(factor_indices), dtype=torch.float32)
                for column, name in enumerate(EVENT_FACTORS):
                    possibility *= node_factor_possibility[name][
                        node
                    ].index_select(0, index_tensor[:, column])
                branch_coverage[branch_index] = torch.maximum(
                    branch_coverage[branch_index], possibility
                )
        self.register_buffer(
            "branch_token_coverage", branch_coverage, persistent=False
        )

        # The exact absorbing-Markov resolvent handles both ordinary DAGs and
        # evidence-induced repeat/revisit cycles without a rollout horizon.
        closure = exact_missed_event_closure(
            language.state_count,
            language.transitions,
            deletion * edge_prior,
        )
        self.register_buffer("closure", closure, persistent=False)
        # Retain a sparse transpose for memory diagnostics.  Query-time vector
        # products use the dense closure, which evaluates the identical finite
        # DAG sum and is faster on the largest automaton (2,080 states, 1.97%
        # non-zero closure entries).
        self.register_buffer(
            "closure_transpose_sparse",
            closure.transpose(0, 1).to_sparse().coalesce(),
            persistent=False,
        )
        self.register_buffer(
            "initial_belief",
            torch.nn.functional.one_hot(
                torch.tensor(language.start_state),
                num_classes=language.state_count,
            ).to(torch.float32),
            persistent=False,
        )

        # Compile the normative observation attached to each latent source
        # frontier.  Keeping these rows separate is essential: averaging them
        # before evaluating the observation likelihood destroys information
        # about mutually exclusive legal continuations.  These tensors are
        # fixed by the expert language and contain no learned or tuned weight.
        state_node = torch.zeros(
            language.state_count, len(language.node_ids), dtype=torch.float32
        )
        if self.edge_source.numel():
            state_node.index_put_(
                (self.edge_source, self.edge_node),
                edge_prior,
                accumulate=True,
            )
        successor_mass = closure @ state_node
        successor_support = successor_mass.sum(dim=-1)
        conditional_node = successor_mass / successor_support.unsqueeze(-1).clamp_min(
            torch.finfo(successor_mass.dtype).tiny
        )
        conditional_node = torch.where(
            successor_support[:, None] > 0,
            conditional_node,
            torch.zeros_like(conditional_node),
        )
        self.register_buffer(
            "frontier_node_probability",
            conditional_node,
            persistent=False,
        )
        self.register_buffer(
            "frontier_successor_support",
            successor_support.clamp(0.0, 1.0),
            persistent=False,
        )
        for name in EVENT_FACTORS:
            possibility = node_factor_possibility[name]
            possibility = possibility / possibility.sum(
                dim=-1, keepdim=True
            ).clamp_min(torch.finfo(possibility.dtype).tiny)
            self.register_buffer(
                f"frontier_{name}_probability",
                conditional_node @ possibility,
                persistent=False,
            )
        if language.state_terminal_probability is not None:
            terminal = closure @ torch.tensor(
                language.state_terminal_probability, dtype=closure.dtype
            )
        else:
            accepting = torch.tensor(
                language.accepting_states, dtype=torch.long
            )
            terminal = (
                closure.index_select(1, accepting).sum(dim=-1)
                if accepting.numel()
                else torch.zeros(language.state_count, dtype=closure.dtype)
            )
        entropy = torch.zeros(language.state_count, dtype=closure.dtype)
        if language.state_count > 1:
            positive = closure.clamp_min(torch.finfo(closure.dtype).tiny)
            entropy = -(closure * positive.log()).sum(dim=-1) / torch.log(
                closure.new_tensor(float(language.state_count))
            )
        self.register_buffer(
            "frontier_terminal_probability", terminal, persistent=False
        )
        self.register_buffer(
            "frontier_closure_entropy", entropy, persistent=False
        )

    def predictive_node_bayes_factor(
        self, factor_probability: Mapping[str, Tensor]
    ) -> Tensor:
        """Compare a soft event mark with every expert successor node.

        Each node defines a normalized categorical law for every semantic
        factor.  The null law is uniform on the same factor ontology.  The
        returned product is therefore a proper likelihood ratio under the
        factorized observation model.  In particular, a uniform (maximally
        uncertain) observer posterior yields a Bayes factor of exactly one,
        so uncertainty cannot be mistaken for structural contradiction.
        """

        if tuple(factor_probability) != EVENT_FACTORS:
            raise ValueError(f"factor probabilities must follow {EVENT_FACTORS}")
        first = factor_probability[EVENT_FACTORS[0]]
        if first.ndim not in {1, 2}:
            raise ValueError("factor probabilities must describe one or many events")
        batched = first.ndim == 2
        events = len(first) if batched else 1
        result = self.closure.new_ones(events, len(self.language.node_ids))
        tiny = torch.finfo(result.dtype).tiny
        for name in EVENT_FACTORS:
            value = factor_probability[name].to(result)
            classes = len(self.factor_labels[name])
            expected = (events, classes) if batched else (classes,)
            if (
                value.shape != expected
                or not torch.isfinite(value).all()
                or torch.any(value < 0)
                or torch.any(value.sum(dim=-1) <= 0)
            ):
                raise ValueError(f"{name} probabilities must have shape {expected}")
            if not batched:
                value = value.unsqueeze(0)
            value = value / value.sum(dim=-1, keepdim=True)
            node = getattr(self, f"node_{name}_possibility").to(result)
            node = node / node.sum(dim=-1, keepdim=True).clamp_min(tiny)
            result = result * (float(classes) * (value @ node.transpose(0, 1)))
        if not torch.isfinite(result).all() or torch.any(result < 0):
            raise RuntimeError("predictive semantic Bayes factors are malformed")
        return result if batched else result[0]

    def predictive_reference_statistics(
        self,
        factor_probability: Mapping[str, Tensor],
        *,
        node_effect_log_bayes_factor: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return mixed/reference evidence for each frontier.

        The latent applicability variable has two branches.  A reference
        branch emits the event from the expert successor language and moves
        the frontier; an open-world branch emits from the uniform ontology
        null and only applies the observer-miss closure.  Its Jeffreys
        posterior probability is already stored as ``innovation_probability``.
        Semantic agreement identifies which transition is being observed but
        is not, by itself, proof of correct execution.  We therefore retain
        only one-sided evidence against reference feasibility: the normalized
        semantic Bayes factor is capped by the probability that a successor
        exists.  Positive endpoint-effect evidence remains uncapped because it
        directly measures the action consequence.  This is a structural
        property of the observation model, not a tunable clipping threshold.
        """

        semantic_node_bayes = self.predictive_node_bayes_factor(factor_probability)
        batched = semantic_node_bayes.ndim == 2
        if not batched:
            semantic_node_bayes = semantic_node_bayes.unsqueeze(0)
        events = len(semantic_node_bayes)
        node_bayes = semantic_node_bayes
        if node_effect_log_bayes_factor is not None:
            effect = node_effect_log_bayes_factor.to(node_bayes)
            if effect.ndim == 1:
                effect = effect.unsqueeze(0).expand(events, -1)
            if effect.shape != node_bayes.shape or not torch.isfinite(effect).all():
                raise ValueError("node effect evidence must align with events and nodes")
            node_bayes = node_bayes * effect.exp()
        semantic_immediate = node_bayes.new_zeros(
            events, self.language.state_count
        )
        combined_immediate = torch.zeros_like(semantic_immediate)
        if self.edge_source.numel():
            source = self.edge_source.to(node_bayes.device)
            node = self.edge_node.to(node_bayes.device)
            prior = self.edge_prior_probability.to(node_bayes).unsqueeze(0)
            semantic_edge = semantic_node_bayes.index_select(1, node) * prior
            combined_edge = node_bayes.index_select(1, node) * prior
            index = source.unsqueeze(0).expand(events, -1)
            semantic_immediate.scatter_add_(1, index, semantic_edge)
            combined_immediate.scatter_add_(1, index, combined_edge)
        closure = self.closure.to(node_bayes)
        semantic_raw = semantic_immediate @ closure.transpose(0, 1)
        combined_raw = combined_immediate @ closure.transpose(0, 1)
        support = self.frontier_successor_support.to(node_bayes).unsqueeze(0).expand(
            events, -1
        )
        scale = torch.where(
            semantic_raw > 0,
            torch.minimum(
                torch.ones_like(semantic_raw),
                support
                / semantic_raw.clamp_min(torch.finfo(semantic_raw.dtype).tiny),
            ),
            torch.ones_like(semantic_raw),
        )
        reference = scale * combined_raw
        open_probability = self.innovation_probability.to(reference)
        reference_probability = 1.0 - open_probability
        mixed = open_probability + reference_probability * reference
        if (
            not torch.isfinite(reference).all()
            or torch.any(reference < 0)
            or not torch.isfinite(mixed).all()
            or torch.any(mixed <= 0)
        ):
            raise RuntimeError("predictive reference law has no finite support")
        if batched:
            return mixed, reference, node_bayes, scale
        return mixed[0], reference[0], node_bayes[0], scale[0]

    def conditional_predictive_advance_probability_batch(
        self,
        source_probability: Tensor,
        factor_probability: Mapping[str, Tensor],
        *,
        node_effect_log_bayes_factor: Tensor | None = None,
    ) -> Tensor:
        """Apply the latent reference/open-world kernel after observing it.

        ``source_probability`` already contains the mixed normal likelihood
        ratio at each latent decision time.  Dividing row-wise by that ratio
        and applying the corresponding mixed operator is the exact
        conditional state transition; it preserves every row's mass and does
        not select an applicability branch.
        """

        if (
            source_probability.ndim != 2
            or source_probability.shape[1] != self.language.state_count
            or not torch.isfinite(source_probability).all()
            or torch.any(source_probability < 0)
        ):
            raise ValueError("predictive source paths must have shape [paths,states]")
        mixed, _, node_bayes, reference_scale = self.predictive_reference_statistics(
            factor_probability,
            node_effect_log_bayes_factor=node_effect_log_bayes_factor,
        )
        if mixed.ndim == 1:
            mixed = mixed.unsqueeze(0)
            node_bayes = node_bayes.unsqueeze(0)
            reference_scale = reference_scale.unsqueeze(0)
        if mixed.shape != source_probability.shape:
            raise ValueError("predictive source paths and observations disagree")
        recovered = source_probability / mixed.clamp_min(
            torch.finfo(source_probability.dtype).tiny
        )
        closure = self.closure.to(recovered)
        open_closed = recovered @ closure
        reference_closed = (recovered * reference_scale) @ closure
        open_probability = self.innovation_probability.to(open_closed)
        reference_probability = 1.0 - open_probability
        destination = open_probability * open_closed
        if self.edge_source.numel():
            source = self.edge_source.to(reference_closed.device)
            node = self.edge_node.to(reference_closed.device)
            target = self.edge_target.to(reference_closed.device)
            edge_mass = (
                reference_probability
                * reference_closed.index_select(1, source)
                * self.edge_prior_probability.to(reference_closed).unsqueeze(0)
                * node_bayes.index_select(1, node)
            )
            destination.scatter_add_(
                1, target.unsqueeze(0).expand(len(destination), -1), edge_mass
            )
        if not torch.allclose(
            destination.sum(dim=-1),
            source_probability.sum(dim=-1),
            atol=4e-5,
            rtol=4e-5,
        ):
            raise RuntimeError("conditional predictive transport changed path mass")
        return destination
