"""Transport over role-asynchronous expert continuations.

Verified successful demonstrations are represented as commitment intervals.
Events of the same role remain ordered, while a cross-role precedence is kept
only when one effect ends before the other event commits.  The reachable
downsets of these orders form a finite continuation language.

At query time, semantic probability mass is transported through this language.
Mass covered by a goal-reaching successor advances the accepted prefix; all
other mass is an innovation and leaves that prefix unchanged.  Alternative
successors share accepted mass according to their semantic evidence.  The
operator is normalized by construction and introduces no score threshold,
beam, edit cost, duration window, or anomaly classifier.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from provia.observer.factors import EVENT_FACTORS, EVENT_ROLES

from .recovery import RecoveryRelationStructure
from .reference_event import UncertainReferenceEvent


def exact_missed_event_closure(
    state_count: int,
    transitions: Sequence[tuple[int, int, int]],
    hidden_edge_probability: Tensor,
) -> Tensor:
    """Marginalize any finite run of unobserved transitions exactly.

    Let ``Q[s,t]`` be the probability that the next procedural transition is
    hidden from the observer.  The probability of stopping the hidden run at
    ``s`` is ``1 - Q[s,:].sum()``.  The closure therefore satisfies

    ``C = diag(stop) + Q C``.

    Solving this linear system supports both acyclic procedures and
    evidence-induced repeat/revisit cycles.  Existence is a property of the
    observation process (the hidden chain must be transient), not a tuned
    truncation depth.
    """

    if state_count <= 0 or hidden_edge_probability.shape != (len(transitions),):
        raise ValueError("hidden-edge probabilities must be edge aligned")
    probability = hidden_edge_probability.detach().to(
        device="cpu", dtype=torch.float64
    )
    if not torch.isfinite(probability).all() or torch.any(probability < 0):
        raise ValueError("hidden-edge probabilities must be finite and non-negative")
    hidden = torch.zeros(state_count, state_count, dtype=torch.float64)
    for edge_index, (source, _, target) in enumerate(transitions):
        hidden[source, target] += probability[edge_index]
    hidden_row_sum = hidden.sum(dim=-1)
    tolerance = 64.0 * torch.finfo(hidden.dtype).eps
    if torch.any(hidden_row_sum > 1.0 + tolerance):
        raise ValueError("hidden transition mass cannot exceed one")
    stop = (1.0 - hidden_row_sum).clamp_min(0.0)
    system = torch.eye(state_count, dtype=torch.float64) - hidden
    try:
        closure = torch.linalg.solve(system, torch.diag(stop))
    except RuntimeError as error:
        raise ValueError(
            "unobserved-transition process is not transient"
        ) from error
    residual = system @ closure - torch.diag(stop)
    if (
        not torch.isfinite(closure).all()
        or residual.abs().amax() > 1e-9
        or torch.any(closure < -1e-9)
    ):
        raise ValueError("failed to obtain a valid missed-event closure")
    closure = closure.clamp_min(0.0)
    row_sum = closure.sum(dim=-1, keepdim=True)
    if torch.any(row_sum <= 0):
        raise ValueError("missed-event closure has an empty row")
    return (closure / row_sum).to(torch.float32)


@dataclass(frozen=True)
class CommittedReferenceAction:
    """One uncertain event decoded from a verified successful video."""

    role: str
    start: float
    commitment: float
    end: float
    event: UncertainReferenceEvent


@dataclass(frozen=True)
class IntervalContinuationLanguage:
    """Union of the exact downset languages induced by expert videos."""

    factor_labels: Mapping[str, tuple[str, ...]]
    node_ids: tuple[str, ...]
    node_roles: tuple[str, ...]
    node_possibility: Mapping[str, tuple[tuple[float, ...], ...]]
    state_count: int
    start_state: int
    accepting_states: tuple[int, ...]
    transitions: tuple[tuple[int, int, int], ...]
    source_trace_count: int
    transition_probability: tuple[float, ...] | None = None
    state_terminal_probability: tuple[float, ...] | None = None
    state_evidence_count: tuple[int, ...] | None = None
    state_outcome_count: tuple[int, ...] | None = None

    def validate(self) -> None:
        if tuple(self.factor_labels) != EVENT_FACTORS:
            raise ValueError(f"factor labels must follow {EVENT_FACTORS}")
        if any(
            not self.factor_labels[name]
            or len(self.factor_labels[name]) != len(set(self.factor_labels[name]))
            for name in EVENT_FACTORS
        ):
            raise ValueError("factor labels must be non-empty and unique")
        node_count = len(self.node_ids)
        if not node_count or len(set(self.node_ids)) != node_count:
            raise ValueError("reference nodes must be non-empty and unique")
        if len(self.node_roles) != node_count or any(
            role not in EVENT_ROLES for role in self.node_roles
        ):
            raise ValueError("every reference node needs one fixed role")
        if tuple(self.node_possibility) != EVENT_FACTORS:
            raise ValueError(f"node possibilities must follow {EVENT_FACTORS}")
        for name in EVENT_FACTORS:
            values = torch.tensor(self.node_possibility[name], dtype=torch.float64)
            if values.shape != (node_count, len(self.factor_labels[name])):
                raise ValueError(f"{name} node possibilities have the wrong shape")
            if (
                not torch.isfinite(values).all()
                or torch.any(values < 0)
                or torch.any(values > 1)
                or not torch.allclose(
                    values.amax(dim=-1),
                    torch.ones(node_count, dtype=values.dtype),
                )
            ):
                raise ValueError(
                    f"{name} possibilities must lie in [0,1] with maximum one"
                )
        if self.state_count <= 1 or not 0 <= self.start_state < self.state_count:
            raise ValueError("continuation state space is invalid")
        if not self.accepting_states or any(
            not 0 <= state < self.state_count for state in self.accepting_states
        ):
            raise ValueError("accepting states are invalid")
        seen: set[tuple[int, int, int]] = set()
        reverse: list[list[int]] = [[] for _ in range(self.state_count)]
        for source, node, target in self.transitions:
            if (
                not 0 <= source < self.state_count
                or not 0 <= target < self.state_count
                or not 0 <= node < node_count
                or source == target
            ):
                raise ValueError("continuation transition is invalid")
            edge = (source, node, target)
            if edge in seen:
                raise ValueError("duplicate continuation transition")
            seen.add(edge)
            reverse[target].append(source)
        if self.transition_probability is not None:
            probability = torch.tensor(
                self.transition_probability, dtype=torch.float64
            )
            if (
                probability.shape != (len(self.transitions),)
                or not torch.isfinite(probability).all()
                or torch.any(probability <= 0)
            ):
                raise ValueError(
                    "transition probabilities must be finite, positive, and edge aligned"
                )
            source_index = torch.tensor(
                [edge[0] for edge in self.transitions], dtype=torch.long
            )
            row_sum = torch.zeros(self.state_count, dtype=torch.float64)
            row_sum.scatter_add_(0, source_index, probability)
            outgoing = torch.zeros(self.state_count, dtype=torch.bool)
            outgoing[source_index] = True
            if not torch.allclose(
                row_sum[outgoing],
                torch.ones_like(row_sum[outgoing]),
                atol=1e-8,
                rtol=1e-8,
            ):
                raise ValueError(
                    "transition probabilities must normalize at every nonterminal state"
                )
        if self.state_terminal_probability is not None:
            terminal = torch.tensor(
                self.state_terminal_probability, dtype=torch.float64
            )
            if (
                terminal.shape != (self.state_count,)
                or not torch.isfinite(terminal).all()
                or torch.any(terminal <= 0)
                or torch.any(terminal >= 1)
            ):
                raise ValueError(
                    "state terminal probabilities must lie strictly in (0,1)"
                )
        reachable = [False] * self.state_count
        queue = deque(self.accepting_states)
        for state in self.accepting_states:
            reachable[state] = True
        while queue:
            target = queue.popleft()
            for source in reverse[target]:
                if not reachable[source]:
                    reachable[source] = True
                    queue.append(source)
        if not all(reachable):
            raise ValueError("every continuation state must reach a successful goal")
        if self.source_trace_count <= 0:
            raise ValueError("source trace count must be positive")
        if (self.state_evidence_count is None) != (
            self.state_outcome_count is None
        ):
            raise ValueError(
                "state evidence and outcome counts must be provided together"
            )
        if self.state_evidence_count is not None:
            if (
                len(self.state_evidence_count) != self.state_count
                or len(self.state_outcome_count) != self.state_count
                or any(value < 0 for value in self.state_evidence_count)
                or any(value <= 0 for value in self.state_outcome_count)
            ):
                raise ValueError(
                    "state evidence counts must be non-negative, outcome "
                    "counts positive, and both state aligned"
                )

    def node_is_globally_necessary(self, node: int) -> bool:
        """Return whether every successful continuation uses this event type."""

        if not 0 <= node < len(self.node_ids):
            raise IndexError(node)
        reverse: list[list[int]] = [[] for _ in range(self.state_count)]
        for source, edge_node, target in self.transitions:
            if edge_node != node:
                reverse[target].append(source)
        reachable = [False] * self.state_count
        queue = deque(self.accepting_states)
        for state in self.accepting_states:
            reachable[state] = True
        while queue:
            target = queue.popleft()
            for source in reverse[target]:
                if not reachable[source]:
                    reachable[source] = True
                    queue.append(source)
        return not reachable[self.start_state]

    def optional_nodes(self) -> tuple[str, ...]:
        return tuple(
            node_id
            for node, node_id in enumerate(self.node_ids)
            if not self.node_is_globally_necessary(node)
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "interval_continuation_language",
            "factor_labels": {
                name: list(self.factor_labels[name]) for name in EVENT_FACTORS
            },
            "node_ids": list(self.node_ids),
            "node_roles": list(self.node_roles),
            "node_possibility": {
                name: [list(row) for row in self.node_possibility[name]]
                for name in EVENT_FACTORS
            },
            "state_count": self.state_count,
            "start_state": self.start_state,
            "accepting_states": list(self.accepting_states),
            "transitions": [list(edge) for edge in self.transitions],
            "source_trace_count": self.source_trace_count,
            
        }
        if self.transition_probability is not None:
            result["transition_probability"] = list(self.transition_probability)
            result["transition_probability_estimator"] = (
                "per-state Jeffreys posterior predictive over expert traversals"
            )
        if self.state_terminal_probability is not None:
            result["state_terminal_probability"] = list(
                self.state_terminal_probability
            )
            result["state_terminal_probability_estimator"] = (
                "state-wise Jeffreys posterior predictive termination law"
            )
        if self.state_evidence_count is not None:
            result["state_evidence_count"] = list(self.state_evidence_count)
            result["state_outcome_count"] = list(self.state_outcome_count)
            result["state_evidence_estimator"] = (
                "reference next-outcome observations and distinct outcomes "
                "per predictive state, from the same Jeffreys evidence "
                "integrals that decide state pooling"
            )
        return result

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> "IntervalContinuationLanguage":
        if value.get("schema") != "interval_continuation_language":
            raise ValueError("unknown interval-continuation schema")
        raw_labels = value.get("factor_labels")
        raw_possibility = value.get("node_possibility")
        if not isinstance(raw_labels, Mapping) or not isinstance(
            raw_possibility, Mapping
        ):
            raise ValueError("factor labels and possibilities must be mappings")
        result = cls(
            factor_labels={
                name: tuple(str(item) for item in raw_labels[name])  # type: ignore[index]
                for name in EVENT_FACTORS
            },
            node_ids=tuple(str(item) for item in value["node_ids"]),  # type: ignore[arg-type]
            node_roles=tuple(str(item) for item in value["node_roles"]),  # type: ignore[arg-type]
            node_possibility={
                name: tuple(
                    tuple(float(part) for part in row)
                    for row in raw_possibility[name]  # type: ignore[index]
                )
                for name in EVENT_FACTORS
            },
            state_count=int(value["state_count"]),
            start_state=int(value["start_state"]),
            accepting_states=tuple(
                int(item) for item in value["accepting_states"]  # type: ignore[arg-type]
            ),
            transitions=tuple(
                tuple(int(part) for part in edge)  # type: ignore[arg-type]
                for edge in value["transitions"]  # type: ignore[union-attr]
            ),
            source_trace_count=int(value["source_trace_count"]),
            transition_probability=(
                tuple(float(item) for item in value["transition_probability"])  # type: ignore[arg-type]
                if "transition_probability" in value
                else None
            ),
            state_terminal_probability=(
                tuple(
                    float(item)
                    for item in value["state_terminal_probability"]  # type: ignore[arg-type]
                )
                if "state_terminal_probability" in value
                else None
            ),
            state_evidence_count=(
                tuple(
                    int(item) for item in value["state_evidence_count"]  # type: ignore[arg-type]
                )
                if "state_evidence_count" in value
                else None
            ),
            state_outcome_count=(
                tuple(
                    int(item) for item in value["state_outcome_count"]  # type: ignore[arg-type]
                )
                if "state_evidence_count" in value
                else None
            ),
        )
        result.validate()
        return result


def compile_interval_continuations(
    traces: Sequence[Sequence[CommittedReferenceAction]],
    *,
    factor_labels: Mapping[str, Sequence[str]],
) -> IntervalContinuationLanguage:
    """Compile exact role-asynchronous downsets without alignment thresholds."""

    materialized = tuple(tuple(trace) for trace in traces)
    if not materialized or any(not trace for trace in materialized):
        raise ValueError("at least one non-empty successful trace is required")
    labels = {
        name: tuple(str(item) for item in factor_labels[name])
        for name in EVENT_FACTORS
    }
    if tuple(labels) != EVENT_FACTORS:
        raise ValueError(f"factor labels must follow {EVENT_FACTORS}")

    ordered_traces: list[tuple[CommittedReferenceAction, ...]] = []
    identities: set[str] = set()
    for raw_trace in materialized:
        indexed = list(enumerate(raw_trace))
        for _, action in indexed:
            if action.role not in EVENT_ROLES:
                raise ValueError("reference actions require left/right roles")
            if not 0 <= action.start <= action.commitment <= action.end:
                raise ValueError("reference action timing is invalid")
        indexed.sort(
            key=lambda item: (
                item[1].commitment,
                item[1].start,
                item[1].end,
                EVENT_ROLES.index(item[1].role),
                item[0],
            )
        )
        actions = tuple(action for _, action in indexed)
        ordered_traces.append(actions)
        for action in actions:
            verb = int(torch.tensor(action.event.verb).argmax())
            noun = int(torch.tensor(action.event.noun).argmax())
            identities.add(
                f"{action.role}|{labels['verb'][verb]}|{labels['noun'][noun]}"
            )

    node_ids = tuple(sorted(identities))
    node_index = {value: index for index, value in enumerate(node_ids)}
    node_roles = tuple(value.split("|", 1)[0] for value in node_ids)
    observations: dict[int, dict[str, list[Tensor]]] = {
        node: {name: [] for name in EVENT_FACTORS}
        for node in range(len(node_ids))
    }

    def identity(action: CommittedReferenceAction) -> int:
        verb = int(torch.tensor(action.event.verb).argmax())
        noun = int(torch.tensor(action.event.noun).argmax())
        return node_index[
            f"{action.role}|{labels['verb'][verb]}|{labels['noun'][noun]}"
        ]

    for actions in ordered_traces:
        for action in actions:
            node = identity(action)
            for name in EVENT_FACTORS:
                values = tuple(float(item) for item in action.event.factor(name))
                if len(values) != len(labels[name]):
                    raise ValueError(f"{name} reference possibility has wrong size")
                tensor = torch.tensor(values, dtype=torch.float64)
                if (
                    not torch.isfinite(tensor).all()
                    or torch.any(tensor < 0)
                    or tensor.max() <= 0
                ):
                    raise ValueError(f"{name} reference possibility is invalid")
                tensor = tensor / tensor.max()
                observations[node][name].append(tensor)

    node_possibility: dict[str, tuple[tuple[float, ...], ...]] = {}
    for name in EVENT_FACTORS:
        rows = []
        for node in range(len(node_ids)):
            merged = torch.stack(observations[node][name]).amax(dim=0)
            merged = merged / merged.max()
            rows.append(tuple(float(item) for item in merged))
        node_possibility[name] = tuple(rows)

    # Build one NFA of all downsets. Each reference contributes its exact
    # commitment-order language; the NFA union keeps whole demonstrated paths
    # without requiring event alignment or a similarity threshold.
    nfa_outgoing: list[dict[int, set[int]]] = []
    nfa_start: list[int] = []
    nfa_accepting: set[int] = set()

    def new_nfa_state() -> int:
        nfa_outgoing.append({})
        return len(nfa_outgoing) - 1

    for actions in ordered_traces:
        prerequisites = [0] * len(actions)
        by_role: dict[str, list[int]] = {role: [] for role in EVENT_ROLES}
        for index, action in enumerate(actions):
            for previous in by_role[action.role]:
                prerequisites[index] |= 1 << previous
            by_role[action.role].append(index)
            for previous, earlier in enumerate(actions[:index]):
                if earlier.role != action.role and earlier.end < action.commitment:
                    prerequisites[index] |= 1 << previous

        full = (1 << len(actions)) - 1
        state_for_mask = {0: new_nfa_state()}
        nfa_start.append(state_for_mask[0])
        queue = deque([0])
        while queue:
            mask = queue.popleft()
            source = state_for_mask[mask]
            for event_index in range(len(actions)):
                bit = 1 << event_index
                if mask & bit or prerequisites[event_index] & ~mask:
                    continue
                following = mask | bit
                target = state_for_mask.get(following)
                if target is None:
                    target = new_nfa_state()
                    state_for_mask[following] = target
                    queue.append(following)
                node = identity(actions[event_index])
                nfa_outgoing[source].setdefault(node, set()).add(target)
        nfa_accepting.add(state_for_mask[full])

    # Subset construction turns the union NFA into an exact deterministic
    # automaton. An accepted state is successful if any source demonstration
    # has reached its goal.
    dfa_subsets = [frozenset(nfa_start)]
    dfa_index = {dfa_subsets[0]: 0}
    dfa_outgoing: list[dict[int, int]] = []
    dfa_accepting: set[int] = set()
    queue = deque([0])
    while queue:
        state = queue.popleft()
        subset = dfa_subsets[state]
        if subset & nfa_accepting:
            dfa_accepting.add(state)
        labels_out = sorted(
            {node for nfa_state in subset for node in nfa_outgoing[nfa_state]}
        )
        outgoing = {}
        for node in labels_out:
            target_subset = frozenset(
                target
                for nfa_state in subset
                for target in nfa_outgoing[nfa_state].get(node, ())
            )
            target = dfa_index.get(target_subset)
            if target is None:
                target = len(dfa_subsets)
                dfa_index[target_subset] = target
                dfa_subsets.append(target_subset)
                queue.append(target)
            outgoing[node] = target
        dfa_outgoing.append(outgoing)

    # Exact right-language minimization. The determinized graph is acyclic
    # because every NFA transition completes one additional reference event.
    visiting: set[int] = set()
    visited: set[int] = set()
    postorder: list[int] = []

    def visit(state: int) -> None:
        if state in visited:
            return
        if state in visiting:
            raise RuntimeError("compiled continuation graph contains a cycle")
        visiting.add(state)
        for target in dfa_outgoing[state].values():
            visit(target)
        visiting.remove(state)
        visited.add(state)
        postorder.append(state)

    visit(0)
    signature_class: dict[tuple[bool, tuple[tuple[int, int], ...]], int] = {}
    state_class: dict[int, int] = {}
    class_signature: list[tuple[bool, tuple[tuple[int, int], ...]]] = []
    for state in postorder:
        signature = (
            state in dfa_accepting,
            tuple(
                sorted(
                    (node, state_class[target])
                    for node, target in dfa_outgoing[state].items()
                )
            ),
        )
        group = signature_class.get(signature)
        if group is None:
            group = len(class_signature)
            signature_class[signature] = group
            class_signature.append(signature)
        state_class[state] = group

    raw_start = state_class[0]
    dense_index = {raw_start: 0}
    dense_accepting: list[int] = []
    transitions: list[tuple[int, int, int]] = []
    queue = deque([raw_start])
    while queue:
        raw_source = queue.popleft()
        source = dense_index[raw_source]
        terminal, outgoing = class_signature[raw_source]
        if terminal:
            dense_accepting.append(source)
        for node, raw_target in outgoing:
            if raw_target not in dense_index:
                dense_index[raw_target] = len(dense_index)
                queue.append(raw_target)
            transitions.append((source, node, dense_index[raw_target]))

    ordered_transitions = tuple(sorted(transitions))
    edge_lookup = {
        (source, node): (edge_index, target)
        for edge_index, (source, node, target) in enumerate(ordered_transitions)
    }
    traversal_count = torch.zeros(len(ordered_transitions), dtype=torch.float64)
    for actions in ordered_traces:
        state = 0
        for action in actions:
            node = identity(action)
            edge = edge_lookup.get((state, node))
            if edge is None:
                raise RuntimeError(
                    "a source demonstration is absent from its compiled language"
                )
            edge_index, state = edge
            traversal_count[edge_index] += 1.0
        if state not in dense_accepting:
            raise RuntimeError("a source demonstration does not reach a goal state")
    transition_probability = torch.zeros_like(traversal_count)
    for source in range(len(dense_index)):
        indices = [
            index
            for index, edge in enumerate(ordered_transitions)
            if edge[0] == source
        ]
        if not indices:
            continue
        index = torch.tensor(indices, dtype=torch.long)
        posterior = traversal_count.index_select(0, index) + 0.5
        transition_probability[index] = posterior / posterior.sum()

    result = IntervalContinuationLanguage(
        factor_labels=labels,
        node_ids=node_ids,
        node_roles=node_roles,
        node_possibility=node_possibility,
        state_count=len(dense_index),
        start_state=0,
        accepting_states=tuple(sorted(dense_accepting)),
        transitions=ordered_transitions,
        source_trace_count=len(materialized),
        transition_probability=tuple(
            float(value) for value in transition_probability
        ),
    )
    result.validate()
    return result


class ContinuationTransport(nn.Module):
    """Causal probability transport over progress and unresolved deviations.

    Legal semantic mass advances the expert continuation language. Rejected
    mass leaves the accepted prefix unchanged and enters a per-role random-set
    residual represented by complementary log-survival intensities. If a later
    event has a fold-observed inverse verb relation and refers to the same
    entity, it thins that residual without advancing procedural progress.
    Progress, contradiction, and cancellation partition one probability mass;
    there is no separate anomaly/recovery classifier or score gate.
    """

    def __init__(
        self,
        language: IntervalContinuationLanguage,
        recovery: RecoveryRelationStructure | None = None,
        *,
        optional_stutter: bool = False,
        missed_event_probability: Sequence[float] | Tensor | None = None,
        posterior_predictive_innovation: bool = False,
    ) -> None:
        super().__init__()
        language.validate()
        self.language = language
        self.optional_stutter = bool(optional_stutter)
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
        # Frontier-conditioned learned evidence inherits its frontier's
        # reference support through the Jeffreys posterior-predictive data
        # fraction N_s / (N_s + alpha * C_s); languages without evidence
        # counts keep a reliability of one and the previous behavior.
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
        if missed_event_probability is None:
            missed_event_probability = (0.0,) * len(EVENT_ROLES)
        deletion = torch.as_tensor(
            missed_event_probability, dtype=torch.float32
        )
        if deletion.shape != (len(EVENT_ROLES),):
            raise ValueError("missed-event probability must have shape [2]")
        if (
            not torch.isfinite(deletion).all()
            or torch.any(deletion < 0)
            or torch.any(deletion > 1)
        ):
            raise ValueError("missed-event probabilities must lie in [0,1]")
        self.register_buffer(
            "missed_event_probability", deletion, persistent=False
        )
        factor_indices = tuple(
            product(
                *(range(len(language.factor_labels[name])) for name in EVENT_FACTORS)
            )
        )
        self.register_buffer(
            "factor_indices",
            torch.tensor(factor_indices, dtype=torch.long),
            persistent=False,
        )
        verb_count = len(language.factor_labels["verb"])
        verb_relation = torch.zeros(verb_count, verb_count, dtype=torch.float32)
        token_entities: tuple[tuple[str, str], ...] = tuple()
        entity_alignment = torch.zeros(2, 2, dtype=torch.float32)
        if recovery is not None:
            recovery.validate()
            if any(
                tuple(recovery.factor_labels[name])
                != tuple(language.factor_labels[name])
                for name in EVENT_FACTORS
            ):
                raise ValueError(
                    "recovery relation and continuation factors disagree"
                )
            hand_labels = tuple(recovery.factor_labels["hand"])
            if any(role not in hand_labels for role in EVENT_ROLES):
                raise ValueError("recovery relation has no left/right roles")
            verb_relation = torch.tensor(
                recovery.verb_possibility, dtype=torch.float32
            )
            entity_alignment = torch.tensor(
                recovery.entity_alignment_possibility, dtype=torch.float32
            )
            labels = language.factor_labels

            def entities(token: tuple[int, ...]) -> tuple[str, str]:
                values = (
                    labels["noun"][token[EVENT_FACTORS.index("noun")]],
                    labels["tool"][token[EVENT_FACTORS.index("tool")]],
                )
                return tuple(
                    ""
                    if value in {"<none>", "<unk>", "none", "unknown"}
                    else value
                    for value in values
                )  # type: ignore[return-value]

            token_entities = tuple(entities(token) for token in factor_indices)
        self.recovery = recovery
        self.register_buffer(
            "verb_cancellation_possibility", verb_relation, persistent=False
        )
        if recovery is None:
            entity_signatures = (("", ""),)
            token_entity_group = torch.zeros(len(factor_indices), dtype=torch.long)
        else:
            entity_signatures = tuple(sorted(set(token_entities)))
            entity_index = {
                values: index for index, values in enumerate(entity_signatures)
            }
            token_entity_group = torch.tensor(
                [entity_index[values] for values in token_entities],
                dtype=torch.long,
            )

        def entity_compatibility(
            before: tuple[str, str], after: tuple[str, str]
        ) -> float:
            survival = 1.0
            for before_slot, before_value in enumerate(before):
                if not before_value:
                    continue
                for after_slot, after_value in enumerate(after):
                    if before_value == after_value:
                        survival *= 1.0 - float(
                            entity_alignment[before_slot, after_slot]
                        )
            return 1.0 - survival

        entity_overlap = torch.tensor(
            [
                [
                    entity_compatibility(before, after)
                    for after in entity_signatures
                ]
                for before in entity_signatures
            ],
            dtype=torch.float32,
        )
        verb_column = EVENT_FACTORS.index("verb")
        token_recovery_state = (
            self.factor_indices[:, verb_column] * len(entity_signatures)
            + token_entity_group
        )
        self.recovery_entity_count = len(entity_signatures)
        self.register_buffer(
            "token_recovery_state", token_recovery_state, persistent=False
        )
        self.register_buffer(
            "recovery_entity_overlap", entity_overlap, persistent=False
        )
        node_token = torch.ones(
            len(language.node_ids), len(factor_indices), dtype=torch.float32
        )
        for column, name in enumerate(EVENT_FACTORS):
            possibility = torch.tensor(
                language.node_possibility[name], dtype=torch.float32
            )
            predictive = possibility / possibility.sum(
                dim=-1, keepdim=True
            ).clamp_min(torch.finfo(possibility.dtype).tiny)
            self.register_buffer(
                f"predictive_node_{name}_probability",
                predictive,
                persistent=False,
            )
            node_token = node_token * possibility.index_select(
                1, self.factor_indices[:, column]
            )
        node_token = node_token / node_token.amax(dim=-1, keepdim=True).clamp_min(
            torch.finfo(node_token.dtype).tiny
        )
        self.register_buffer("node_token_possibility", node_token, persistent=False)

        edges = torch.tensor(language.transitions, dtype=torch.long)
        self.register_buffer("edge_source", edges[:, 0], persistent=False)
        self.register_buffer("edge_node", edges[:, 1], persistent=False)
        self.register_buffer("edge_target", edges[:, 2], persistent=False)
        edge_role = torch.tensor(
            [EVENT_ROLES.index(language.node_roles[node]) for node in edges[:, 1]],
            dtype=torch.long,
        )
        self.register_buffer("edge_role", edge_role, persistent=False)
        outgoing: list[list[int]] = [[] for _ in range(language.state_count)]
        for edge_index, (source, _, target) in enumerate(language.transitions):
            outgoing[source].append(edge_index)
        outdegree = torch.tensor([len(values) for values in outgoing], dtype=torch.float32)
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
        self._outgoing_edges = tuple(tuple(edges) for edges in outgoing)
        # Exact absorbing-Markov closure remains valid when predictive-state
        # induction discovers repeatable or revisitable procedure states.
        missed_closure = exact_missed_event_closure(
            language.state_count,
            language.transitions,
            deletion.index_select(0, edge_role) * edge_prior,
        )
        self.register_buffer(
            "missed_event_closure_matrix",
            missed_closure,
            persistent=False,
        )
        coverage = torch.zeros(
            len(EVENT_ROLES),
            language.state_count,
            len(factor_indices),
            dtype=torch.float32,
        )
        for edge_index, (source, node, _) in enumerate(language.transitions):
            role = int(edge_role[edge_index])
            coverage[role, source] = torch.maximum(
                coverage[role, source], node_token[node]
            )
        self.register_buffer("state_token_coverage", coverage, persistent=False)
        optional_coverage = torch.zeros(
            len(EVENT_ROLES), len(factor_indices), dtype=torch.float32
        )
        if self.optional_stutter:
            for node, node_id in enumerate(language.node_ids):
                if node_id not in language.optional_nodes():
                    continue
                role = EVENT_ROLES.index(language.node_roles[node])
                optional_coverage[role] = torch.maximum(
                    optional_coverage[role], node_token[node]
                )
        # A token is neutral only to the extent that its optional-event support
        # is not already claimed by an outgoing progress transition.  Thus
        # progress, neutral stutter, and rejection remain a pointwise partition.
        neutral = (
            optional_coverage[:, None, :] - coverage
        ).clamp_min(0.0)
        self.register_buffer("state_token_neutral", neutral, persistent=False)
        accepting = torch.zeros(language.state_count, dtype=torch.bool)
        accepting[list(language.accepting_states)] = True
        self.register_buffer("accepting", accepting, persistent=False)

        # Preserve the normative successor law for every latent source
        # frontier and role.  This is the state-conditioned counterpart of
        # successor_factor_probability; it prevents distinct legal branches
        # from being collapsed into a belief-averaged pseudo action before the
        # observation likelihood is evaluated.
        normalized_node_token = node_token / node_token.sum(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(node_token.dtype).tiny)
        for role in range(len(EVENT_ROLES)):
            selected = edge_role == role
            state_node = torch.zeros(
                language.state_count,
                len(language.node_ids),
                dtype=torch.float32,
            )
            if torch.any(selected):
                state_node.index_put_(
                    (self.edge_source[selected], self.edge_node[selected]),
                    edge_prior[selected],
                    accumulate=True,
                )
            successor_mass = missed_closure @ state_node
            support = successor_mass.sum(dim=-1)
            conditional_node = successor_mass / support.unsqueeze(-1).clamp_min(
                torch.finfo(successor_mass.dtype).tiny
            )
            conditional_node = torch.where(
                support[:, None] > 0,
                conditional_node,
                torch.zeros_like(conditional_node),
            )
            self.register_buffer(
                f"frontier_node_probability_{role}",
                conditional_node,
                persistent=False,
            )
            token = conditional_node @ normalized_node_token
            self.register_buffer(
                f"frontier_successor_support_{role}",
                support.clamp(0.0, 1.0),
                persistent=False,
            )
            for column, name in enumerate(EVENT_FACTORS):
                marginal = torch.zeros(
                    language.state_count,
                    len(language.factor_labels[name]),
                    dtype=token.dtype,
                )
                marginal.scatter_add_(
                    1,
                    self.factor_indices[:, column]
                    .unsqueeze(0)
                    .expand(language.state_count, -1),
                    token,
                )
                self.register_buffer(
                    f"frontier_{name}_probability_{role}",
                    marginal,
                    persistent=False,
                )
        terminal = missed_closure[:, accepting].sum(dim=-1)
        entropy = torch.zeros(language.state_count, dtype=missed_closure.dtype)
        if language.state_count > 1:
            positive = missed_closure.clamp_min(
                torch.finfo(missed_closure.dtype).tiny
            )
            entropy = -(missed_closure * positive.log()).sum(dim=-1) / torch.log(
                missed_closure.new_tensor(float(language.state_count))
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
        """Return normalized expert-node versus uniform-null evidence."""

        if tuple(factor_probability) != EVENT_FACTORS:
            raise ValueError(f"factor probabilities must follow {EVENT_FACTORS}")
        result = self.node_token_possibility.new_ones(len(self.language.node_ids))
        for name in EVENT_FACTORS:
            value = factor_probability[name].to(result)
            classes = len(self.language.factor_labels[name])
            if (
                value.shape != (classes,)
                or not torch.isfinite(value).all()
                or torch.any(value < 0)
                or value.sum() <= 0
            ):
                raise ValueError(f"{name} event probability must have shape {(classes,)}")
            value = value / value.sum()
            node = getattr(self, f"predictive_node_{name}_probability").to(result)
            result = result * (float(classes) * (node @ value))
        if not torch.isfinite(result).all() or torch.any(result < 0):
            raise RuntimeError("predictive semantic Bayes factors are malformed")
        return result

    def predictive_reference_statistics(
        self,
        factor_probability: Mapping[str, Tensor],
        role: int,
        *,
        node_effect_log_bayes_factor: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return one-sided semantic feasibility and latent applicability."""

        if not 0 <= int(role) < len(EVENT_ROLES):
            raise ValueError("event role must select a fixed left/right slot")
        semantic_node_bayes = self.predictive_node_bayes_factor(factor_probability)
        node_bayes = semantic_node_bayes
        if node_effect_log_bayes_factor is not None:
            effect = node_effect_log_bayes_factor.to(node_bayes)
            if effect.shape != node_bayes.shape or not torch.isfinite(effect).all():
                raise ValueError("node effect evidence must align with reference nodes")
            node_bayes = node_bayes * effect.exp()
        semantic_immediate = node_bayes.new_zeros(self.language.state_count)
        combined_immediate = torch.zeros_like(semantic_immediate)
        selected = self.edge_role == int(role)
        if torch.any(selected):
            source = self.edge_source[selected].to(node_bayes.device)
            node = self.edge_node[selected].to(node_bayes.device)
            prior = self.edge_prior_probability[selected].to(node_bayes)
            semantic_immediate.scatter_add_(
                0, source, prior * semantic_node_bayes.index_select(0, node)
            )
            combined_immediate.scatter_add_(
                0, source, prior * node_bayes.index_select(0, node)
            )
        closure = self.missed_event_closure_matrix.to(node_bayes)
        semantic_raw = closure @ semantic_immediate
        combined_raw = closure @ combined_immediate
        support = getattr(
            self, f"frontier_successor_support_{int(role)}"
        ).to(node_bayes)
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
        return mixed, reference, node_bayes, scale

    def predictive_advance_components(
        self,
        source_probability: Tensor,
        role: int,
        factor_probability: Mapping[str, Tensor],
        *,
        node_effect_log_bayes_factor: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply open and reference branches without collapsing applicability.

        The two returned state tensors retain their latent branch identity.
        Their sum has row mass equal to the mixed likelihood ratio.  This lets
        the bimanual filter apply endpoint effects only to the
        reference-grounded normal branch while preserving exact normalization.
        """

        states = self.language.state_count
        if source_probability.ndim < 1 or source_probability.shape[0] != states:
            raise ValueError("source frontier probability must lead with states")
        if (
            not torch.isfinite(source_probability).all()
            or torch.any(source_probability < 0)
        ):
            raise ValueError("source frontier probability is malformed")
        mixed, _, node_bayes, reference_scale = self.predictive_reference_statistics(
            factor_probability,
            role,
            node_effect_log_bayes_factor=node_effect_log_bayes_factor,
        )
        trailing = source_probability.shape[1:]
        source = source_probability.reshape(states, -1).transpose(0, 1)
        closure = self.missed_event_closure_matrix.to(source)
        open_closed = source @ closure
        reference_closed = (source * reference_scale.unsqueeze(0)) @ closure
        open_probability = self.innovation_probability.to(open_closed)
        reference_probability = 1.0 - open_probability
        open_destination = open_probability * open_closed
        reference_destination = torch.zeros_like(reference_closed)
        selected = self.edge_role == int(role)
        if torch.any(selected):
            edge_source = self.edge_source[selected].to(reference_closed.device)
            edge_node = self.edge_node[selected].to(reference_closed.device)
            edge_target = self.edge_target[selected].to(reference_closed.device)
            edge_mass = (
                reference_probability
                * reference_closed.index_select(1, edge_source)
                * self.edge_prior_probability[selected].to(reference_closed).unsqueeze(0)
                * node_bayes.index_select(0, edge_node).unsqueeze(0)
            )
            reference_destination.scatter_add_(
                1,
                edge_target.unsqueeze(0).expand(len(reference_destination), -1),
                edge_mass,
            )
        expected = (
            source_probability
            * mixed.reshape(states, *((1,) * len(trailing))).to(source_probability)
        ).sum()
        if not torch.allclose(
            open_destination.sum() + reference_destination.sum(),
            expected,
            atol=4e-5,
            rtol=4e-5,
        ):
            raise RuntimeError("predictive role transport changed likelihood mass")

        def restore(value: Tensor) -> Tensor:
            return value.transpose(0, 1).reshape(states, *trailing)

        return restore(open_destination), restore(reference_destination), mixed

    def conditional_predictive_advance_probability(
        self,
        source_probability: Tensor,
        role: int,
        factor_probability: Mapping[str, Tensor],
        *,
        node_effect_log_bayes_factor: Tensor | None = None,
    ) -> Tensor:
        """Apply one observed endpoint law without changing posterior mass.

        ``source_probability`` is already conditioned on the mixed
        reference/open-world endpoint likelihood.  This method removes that
        likelihood once, applies the corresponding latent transition, and
        restores exactly the same mass.  It is the role-aware counterpart of
        :meth:`FactorizedCompletionTransport.conditional_predictive_advance_probability_batch`
        and is used only when an event effect is complete.
        """

        states = self.language.state_count
        if source_probability.ndim < 1 or source_probability.shape[0] != states:
            raise ValueError(
                "conditional source probability must lead with continuation states"
            )
        if (
            not torch.isfinite(source_probability).all()
            or torch.any(source_probability < 0)
        ):
            raise ValueError("conditional source probability is malformed")
        mixed, _, _, _ = self.predictive_reference_statistics(
            factor_probability,
            role,
            node_effect_log_bayes_factor=node_effect_log_bayes_factor,
        )
        view = mixed.reshape(states, *((1,) * (source_probability.ndim - 1))).to(
            source_probability
        )
        recovered = source_probability / view.clamp_min(
            torch.finfo(source_probability.dtype).tiny
        )
        open_destination, reference_destination, _ = (
            self.predictive_advance_components(
                recovered,
                role,
                factor_probability,
                node_effect_log_bayes_factor=node_effect_log_bayes_factor,
            )
        )
        destination = open_destination + reference_destination
        if not torch.allclose(
            destination.sum(dim=0),
            source_probability.sum(dim=0),
            atol=4e-5,
            rtol=4e-5,
        ):
            raise RuntimeError("conditional role transport changed posterior mass")
        return destination
