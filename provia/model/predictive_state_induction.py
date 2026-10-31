"""Bayesian predictive-state induction from successful event traces.

An exact prefix-tree language treats two observed histories as different unless
their complete future languages are identical.  That is appropriate with one
short procedure but memorizes demonstrations when a task has many legitimate
executions.  This module instead asks a predictive question: do two histories
need separate next-event laws?

Each prefix state supplies categorical observations over the next event and a
terminal symbol.  States are pooled only when the Jeffreys-integrated evidence
of one shared categorical law exceeds the evidence of two separate laws.  The
equal-model-prior decision is a Bayes factor greater than one; there is no
similarity radius, support threshold, edit cost, or selected state count.

Pooling never joins an ancestor with its descendant.  Equivalent predictive
states from executions with different local order can nevertheless induce a
cycle; this is retained as evidence for a repeatable or revisitable procedure
state rather than removed by an order heuristic.  If one pooled state can
reach different future states through the same event, all destinations remain
latent and receive posterior-predictive probability.  The result is therefore
a probabilistic predictive-state process rather than an over-generalized
deterministic graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import combinations
from math import lgamma
from typing import Mapping, Sequence

from .interval_transport import IntervalContinuationLanguage


JEFFREYS_CONCENTRATION = 0.5


@dataclass(frozen=True)
class BayesianPredictiveQuotient:
    """Finite posterior-predictive continuation process."""

    state_count: int
    start_state: int
    accepting_states: tuple[int, ...]
    transitions: tuple[tuple[int, int, int], ...]
    transition_probability: tuple[float, ...]
    state_terminal_probability: tuple[float, ...]
    source_trace_count: int
    source_state_count: int
    merged_state_count: int
    log_bayes_evidence_gain: float

    def validate(self, node_count: int) -> None:
        if (
            self.state_count <= 1
            or self.start_state != 0
            or self.source_trace_count <= 0
            or self.source_state_count < self.state_count
            or self.merged_state_count != self.source_state_count - self.state_count
        ):
            raise ValueError("predictive quotient metadata is malformed")
        if not self.accepting_states or any(
            state < 0 or state >= self.state_count
            for state in self.accepting_states
        ):
            raise ValueError("predictive quotient has invalid accepting states")
        if len(self.transitions) != len(self.transition_probability):
            raise ValueError("transition probabilities must be edge aligned")
        if len(self.state_terminal_probability) != self.state_count:
            raise ValueError("terminal probabilities must be state aligned")
        if any(
            not 0.0 < value < 1.0
            for value in self.state_terminal_probability
        ):
            raise ValueError("terminal probabilities must lie strictly in (0,1)")
        seen: set[tuple[int, int, int]] = set()
        outgoing = [0.0] * self.state_count
        adjacency: list[list[int]] = [[] for _ in range(self.state_count)]
        reverse: list[list[int]] = [[] for _ in range(self.state_count)]
        for edge, probability in zip(
            self.transitions, self.transition_probability
        ):
            source, node, target = edge
            if (
                source < 0
                or source >= self.state_count
                or target < 0
                or target >= self.state_count
                or node < 0
                or node >= node_count
                or source == target
                or edge in seen
                or probability <= 0.0
            ):
                raise ValueError("predictive quotient has an invalid edge")
            seen.add(edge)
            outgoing[source] += probability
            adjacency[source].append(target)
            reverse[target].append(source)
        for value in outgoing:
            if value and abs(value - 1.0) > 1e-8:
                raise ValueError("outgoing transition probabilities must normalize")
        reachable = {self.start_state}
        queue = [self.start_state]
        while queue:
            state = queue.pop()
            for target in adjacency[state]:
                if target not in reachable:
                    reachable.add(target)
                    queue.append(target)
        if len(reachable) != self.state_count:
            raise ValueError("predictive quotient contains an unreachable state")
        successful = set(self.accepting_states)
        queue = list(self.accepting_states)
        while queue:
            state = queue.pop()
            for source in reverse[state]:
                if source not in successful:
                    successful.add(source)
                    queue.append(source)
        if len(successful) != self.state_count:
            raise ValueError("every predictive state must retain a successful continuation")


@dataclass
class _Cluster:
    members: tuple[int, ...]
    member_mask: int
    descendant_mask: int
    counts: dict[int, int]
    active: bool = True


def _integrated_categorical_log_evidence(
    counts: Mapping[int, int], classes: int
) -> float:
    """Ordered-sample evidence under a symmetric Jeffreys categorical law."""

    if classes <= 1 or any(value <= 0 for value in counts.values()):
        raise ValueError("categorical evidence requires positive sparse counts")
    alpha = JEFFREYS_CONCENTRATION
    total = sum(counts.values())
    value = lgamma(classes * alpha) - lgamma(total + classes * alpha)
    value += sum(
        lgamma(count + alpha) - lgamma(alpha)
        for count in counts.values()
    )
    return value


def _integrated_bernoulli_log_evidence(successes: int, failures: int) -> float:
    """Ordered-sample evidence under the Jeffreys Beta(1/2, 1/2) law."""

    if successes < 0 or failures < 0:
        raise ValueError("Bernoulli counts must be non-negative")
    alpha = JEFFREYS_CONCENTRATION
    return (
        lgamma(2.0 * alpha)
        - lgamma(successes + failures + 2.0 * alpha)
        + lgamma(successes + alpha)
        - lgamma(alpha)
        + lgamma(failures + alpha)
        - lgamma(alpha)
    )


def _predictive_log_evidence(
    counts: Mapping[int, int], node_count: int
) -> float:
    """Evidence for termination and the next mark as two distinct clocks."""

    if node_count <= 0:
        raise ValueError("predictive evidence requires at least one event mark")
    terminal = int(counts.get(node_count, 0))
    marks = {
        outcome: count
        for outcome, count in counts.items()
        if outcome != node_count
    }
    continuation = sum(marks.values())
    value = _integrated_bernoulli_log_evidence(terminal, continuation)
    if continuation:
        value += _integrated_categorical_log_evidence(marks, node_count)
    return value


def _combined_counts(
    first: Mapping[int, int], second: Mapping[int, int]
) -> dict[int, int]:
    result = dict(first)
    for outcome, count in second.items():
        result[outcome] = result.get(outcome, 0) + count
    return result


def _log_merge_bayes_factor(
    first: Mapping[int, int], second: Mapping[int, int], node_count: int
) -> float:
    combined = _combined_counts(first, second)
    return (
        _predictive_log_evidence(combined, node_count)
        - _predictive_log_evidence(first, node_count)
        - _predictive_log_evidence(second, node_count)
    )


def _has_ancestor_conflict(
    first: _Cluster,
    second: _Cluster,
) -> bool:
    """Whether either cluster contains an ancestor of the other cluster."""

    return bool(
        (first.descendant_mask & second.member_mask)
        or (second.descendant_mask & first.member_mask)
    )


def compile_bayesian_predictive_quotient(
    traces: Sequence[Sequence[int]],
    *,
    node_count: int,
    source_language: IntervalContinuationLanguage | None = None,
) -> BayesianPredictiveQuotient:
    """Pool predictively exchangeable states by integrated model evidence.

    With no source language, the empirical prefix tree is used.  Supplying an
    exact chain or role-partial-order language preserves every legal source
    branch; only source states visited by the decoded expert streams carry
    observations and can be pooled.
    """

    materialized = tuple(tuple(int(node) for node in trace) for trace in traces)
    if (
        node_count <= 0
        or not materialized
        or any(not trace for trace in materialized)
        or any(node < 0 or node >= node_count for trace in materialized for node in trace)
    ):
        raise ValueError("predictive-state compilation needs valid non-empty traces")

    if source_language is None:
        children: list[dict[int, int]] = [{}]
        edge_count: list[dict[int, int]] = [{}]
        terminal_count = [0]
        for trace in materialized:
            state = 0
            for node in trace:
                edge_count[state][node] = edge_count[state].get(node, 0) + 1
                following = children[state].get(node)
                if following is None:
                    following = len(children)
                    children[state][node] = following
                    children.append({})
                    edge_count.append({})
                    terminal_count.append(0)
                state = following
            terminal_count[state] += 1
        structural_accepting = {
            state for state, count in enumerate(terminal_count) if count
        }
    else:
        source_language.validate()
        if (
            len(source_language.node_ids) != node_count
            or source_language.start_state != 0
        ):
            raise ValueError("source language and expert event ontology disagree")
        children = [{} for _ in range(source_language.state_count)]
        edge_count = [{} for _ in range(source_language.state_count)]
        terminal_count = [0 for _ in range(source_language.state_count)]
        for source, node, target in source_language.transitions:
            if node in children[source]:
                raise ValueError(
                    "predictive induction requires a deterministic source language"
                )
            children[source][node] = target
            edge_count[source][node] = 0
        structural_accepting = set(source_language.accepting_states)
        for trace in materialized:
            state = source_language.start_state
            for node in trace:
                following = children[state].get(node)
                if following is None:
                    raise ValueError(
                        "a decoded expert trace is absent from the source language"
                    )
                edge_count[state][node] += 1
                state = following
            if state not in structural_accepting:
                raise ValueError("a decoded expert trace does not reach a source goal")
            terminal_count[state] += 1

    indegree = [0] * len(children)
    for outgoing in children:
        for target in outgoing.values():
            indegree[target] += 1
    topological = [state for state, degree in enumerate(indegree) if degree == 0]
    position = 0
    while position < len(topological):
        source = topological[position]
        position += 1
        for target in children[source].values():
            indegree[target] -= 1
            if indegree[target] == 0:
                topological.append(target)
    if len(topological) != len(children):
        raise ValueError("source expert language must be acyclic before pooling")
    reachable = {0}
    queue_states = [0]
    while queue_states:
        source = queue_states.pop()
        for target in children[source].values():
            if target not in reachable:
                reachable.add(target)
                queue_states.append(target)
    if len(reachable) != len(children):
        raise ValueError("source expert language contains an unreachable state")
    descendant_mask = [0] * len(children)
    for source in reversed(topological):
        mask = 0
        for target in children[source].values():
            mask |= (1 << target) | descendant_mask[target]
        descendant_mask[source] = mask
    terminal_outcome = node_count
    singleton_counts: list[dict[int, int]] = []
    for state in range(len(children)):
        counts = {
            node: count
            for node, count in edge_count[state].items()
            if count > 0
        }
        if terminal_count[state]:
            counts[terminal_outcome] = terminal_count[state]
        singleton_counts.append(counts)

    clusters: dict[int, _Cluster] = {}
    evidence_gain = 0.0
    for state, counts in enumerate(singleton_counts):
        members = (state,)
        cluster_id = state
        clusters[cluster_id] = _Cluster(
            members=members,
            member_mask=1 << state,
            descendant_mask=descendant_mask[state],
            counts=dict(counts),
        )

    support_index: dict[int, set[int]] = {}
    for state, cluster in clusters.items():
        if state == 0:
            continue
        for outcome in cluster.counts:
            support_index.setdefault(outcome, set()).add(state)

    queue: list[tuple[float, int, int]] = []
    scheduled: set[tuple[int, int]] = set()

    def schedule(first_id: int, second_id: int) -> None:
        if first_id == second_id or 0 in {first_id, second_id}:
            return
        left, right = sorted((first_id, second_id))
        pair = (left, right)
        if pair in scheduled:
            return
        first = clusters.get(left)
        second = clusters.get(right)
        if (
            first is None
            or second is None
            or not first.active
            or not second.active
            or not (first.counts.keys() & second.counts.keys())
            or _has_ancestor_conflict(first, second)
        ):
            return
        gain = _log_merge_bayes_factor(
            first.counts, second.counts, node_count
        )
        if gain > 0.0:
            scheduled.add(pair)
            heappush(queue, (-gain, left, right))

    initial_pairs: set[tuple[int, int]] = set()
    for states in support_index.values():
        initial_pairs.update(combinations(sorted(states), 2))
    for first, second in sorted(initial_pairs):
        schedule(first, second)

    next_cluster = len(clusters)
    while queue:
        negative_gain, first_id, second_id = heappop(queue)
        scheduled.discard((first_id, second_id))
        first = clusters.get(first_id)
        second = clusters.get(second_id)
        if (
            first is None
            or second is None
            or not first.active
            or not second.active
            or _has_ancestor_conflict(first, second)
        ):
            continue
        gain = _log_merge_bayes_factor(
            first.counts, second.counts, node_count
        )
        if gain <= 0.0:
            continue
        # A stale priority can only be present after one participant was
        # replaced, in which case it is inactive above.  Recomputing keeps the
        # evidence audit explicit and guards numerical drift.
        if abs(gain + negative_gain) > 1e-10:
            heappush(queue, (-gain, first_id, second_id))
            scheduled.add((first_id, second_id))
            continue
        first.active = False
        second.active = False
        members = tuple(sorted(first.members + second.members))
        combined = _Cluster(
            members=members,
            member_mask=first.member_mask | second.member_mask,
            descendant_mask=(
                first.descendant_mask | second.descendant_mask
            ),
            counts=_combined_counts(first.counts, second.counts),
        )
        cluster_id = next_cluster
        next_cluster += 1
        clusters[cluster_id] = combined
        evidence_gain += gain
        neighbors: set[int] = set()
        for outcome in combined.counts:
            neighbors.update(support_index.get(outcome, ()))
            support_index.setdefault(outcome, set()).add(cluster_id)
        for other in sorted(neighbors):
            schedule(cluster_id, other)

    active = [
        cluster_id
        for cluster_id, cluster in clusters.items()
        if cluster.active
    ]
    state_cluster = [-1] * len(children)
    for cluster_id in active:
        for state in clusters[cluster_id].members:
            state_cluster[state] = cluster_id
    if any(cluster < 0 for cluster in state_cluster):
        raise RuntimeError("predictive quotient lost a prefix state")

    edge_traversal: dict[tuple[int, int, int], int] = {}
    for source, outgoing in enumerate(children):
        source_cluster = state_cluster[source]
        for node, target in outgoing.items():
            target_cluster = state_cluster[target]
            if source_cluster == target_cluster:
                raise RuntimeError("an ancestor merge introduced a self-loop")
            edge = (source_cluster, node, target_cluster)
            edge_traversal[edge] = edge_traversal.get(edge, 0) + edge_count[
                source
            ][node]

    outgoing_by_cluster: dict[int, list[tuple[int, int, int]]] = {}
    for (source, node, target), count in edge_traversal.items():
        outgoing_by_cluster.setdefault(source, []).append((node, target, count))

    # Deterministic breadth-first indexing makes independently compiled
    # artifacts byte stable.  Same-node latent destinations are ordered by
    # their source-prefix membership rather than by hash iteration order.
    start_cluster = state_cluster[0]
    dense = {start_cluster: 0}
    order = [start_cluster]
    position = 0
    while position < len(order):
        source = order[position]
        position += 1
        for _, target, _ in sorted(
            outgoing_by_cluster.get(source, ()),
            key=lambda item: (item[0], clusters[item[1]].members),
        ):
            if target not in dense:
                dense[target] = len(dense)
                order.append(target)
    if len(dense) != len(active):
        raise RuntimeError("predictive quotient contains an unreachable state")

    transitions: list[tuple[int, int, int]] = []
    probability: list[float] = []
    terminal_probability: list[float] = []
    accepting: list[int] = []
    for source_cluster in order:
        source = dense[source_cluster]
        members = clusters[source_cluster].members
        terminal = sum(terminal_count[state] for state in members)
        outgoing = outgoing_by_cluster.get(source_cluster, [])
        outgoing_total = sum(count for _, _, count in outgoing)
        terminal_probability.append(
            (terminal + JEFFREYS_CONCENTRATION)
            / (terminal + outgoing_total + 2.0 * JEFFREYS_CONCENTRATION)
        )
        if any(state in structural_accepting for state in members):
            accepting.append(source)
        if not outgoing:
            continue
        by_node: dict[int, list[tuple[int, int]]] = {}
        for node, target, count in outgoing:
            by_node.setdefault(node, []).append((target, count))
        node_total = {
            node: sum(count for _, count in values)
            for node, values in by_node.items()
        }
        node_denominator = outgoing_total + JEFFREYS_CONCENTRATION * len(
            by_node
        )
        for node in sorted(by_node):
            values = sorted(
                by_node[node], key=lambda item: clusters[item[0]].members
            )
            node_probability = (
                node_total[node] + JEFFREYS_CONCENTRATION
            ) / node_denominator
            target_denominator = node_total[node] + JEFFREYS_CONCENTRATION * len(
                values
            )
            for target, count in values:
                transitions.append((source, node, dense[target]))
                probability.append(
                    node_probability
                    * (count + JEFFREYS_CONCENTRATION)
                    / target_denominator
                )

    result = BayesianPredictiveQuotient(
        state_count=len(order),
        start_state=0,
        accepting_states=tuple(accepting),
        transitions=tuple(transitions),
        transition_probability=tuple(probability),
        state_terminal_probability=tuple(terminal_probability),
        source_trace_count=len(materialized),
        source_state_count=len(children),
        merged_state_count=len(children) - len(order),
        log_bayes_evidence_gain=evidence_gain,
    )
    result.validate(node_count)
    return result


def quotient_interval_language(
    source: IntervalContinuationLanguage,
    traces: Sequence[Sequence[int]],
) -> tuple[IntervalContinuationLanguage, BayesianPredictiveQuotient]:
    """Replace exact history identity with the Bayesian predictive quotient."""

    source.validate()
    quotient = compile_bayesian_predictive_quotient(
        traces,
        node_count=len(source.node_ids),
        source_language=source,
    )
    language = IntervalContinuationLanguage(
        factor_labels=source.factor_labels,
        node_ids=source.node_ids,
        node_roles=source.node_roles,
        node_possibility=source.node_possibility,
        state_count=quotient.state_count,
        start_state=quotient.start_state,
        accepting_states=quotient.accepting_states,
        transitions=quotient.transitions,
        source_trace_count=quotient.source_trace_count,
        transition_probability=quotient.transition_probability,
        state_terminal_probability=quotient.state_terminal_probability,
    )
    language.validate()
    return language, quotient


__all__ = [
    "BayesianPredictiveQuotient",
    "JEFFREYS_CONCENTRATION",
    "compile_bayesian_predictive_quotient",
    "quotient_interval_language",
]
