"""Inference ablations of the procedure-state models.

The interventions in this module do not retrain or replace the learned visual
emission model.  They alter exactly one probabilistic part of the frozen
filter, which makes component attribution independent of optimization noise.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn

from .mode_filter import CORRECTION, MODE_NAMES


FULL_INFERENCE: Final = "full"
NO_PROCEDURE_FRONTIER: Final = "no_procedure_frontier"
NO_PERSISTENT_PROCEDURE_FRONTIER: Final = (
    "no_persistent_procedure_frontier"
)
NO_INSERTION_STATE: Final = "no_insertion_state"
NO_EXECUTION_MODE_PERSISTENCE: Final = "no_execution_mode_persistence"
NO_CAUSAL_DECISION_CLOCK: Final = "no_causal_decision_clock"
NO_OPEN_WORLD_INNOVATION: Final = "no_open_world_innovation"
POSTERIOR_NORMAL_HISTORY: Final = "posterior_normal_history"
IDENTIFIABLE_MODE_SUPPORT: Final = "identifiable_mode_support"
NO_STATE_SEPARATION: Final = "no_state_separation"
NO_FACTUAL_HISTORY: Final = "no_factual_history"
NO_CROSS_ACTOR_HISTORY: Final = "no_cross_actor_history"

INFERENCE_ABLATIONS: Final = (
    FULL_INFERENCE,
    NO_PROCEDURE_FRONTIER,
    NO_PERSISTENT_PROCEDURE_FRONTIER,
    NO_INSERTION_STATE,
    NO_EXECUTION_MODE_PERSISTENCE,
    NO_CAUSAL_DECISION_CLOCK,
    NO_OPEN_WORLD_INNOVATION,
    POSTERIOR_NORMAL_HISTORY,
    IDENTIFIABLE_MODE_SUPPORT,
    NO_STATE_SEPARATION,
    NO_FACTUAL_HISTORY,
    NO_CROSS_ACTOR_HISTORY,
)

INFERENCE_ABLATION_DEFINITIONS: Final = {
    FULL_INFERENCE: "Unmodified frozen model and exact causal filter.",
    NO_FACTUAL_HISTORY: (
        "Reset the factual history to its initial value before every "
        "candidate, so the observation model sees no record of the "
        "preceding segments; belief, transport, and emissions are unchanged."
    ),
    NO_CROSS_ACTOR_HISTORY: (
        "Zero the linear map that adds the other actor's factual history to "
        "a segment's query, so every hand is read against its own history "
        "alone; the joint belief, transport, and emissions are unchanged."
    ),
    NO_PROCEDURE_FRONTIER: (
        "Analytically marginalize every frontier-conditioned mark over a "
        "uniform bag of expert-language frontiers with legal successors, "
        "then prohibit procedure-state transport; retain execution mode, "
        "lifecycle, and factual history."
    ),
    NO_PERSISTENT_PROCEDURE_FRONTIER: (
        "Replace the dynamic frontier before every candidate observation by "
        "a uniform mixture over all expert-language frontiers with at least "
        "one legal successor; preserve the complete mode/lifecycle marginal."
    ),
    NO_INSERTION_STATE: (
        "Set the observer-origin prior to genuine with probability one, "
        "thereby removing the insertion explanation without changing learned "
        "mark likelihoods."
    ),
    NO_EXECUTION_MODE_PERSISTENCE: (
        "Replace every execution-mode transition row by the fold-training "
        "initial mode distribution, making consecutive genuine event modes "
        "conditionally independent."
    ),
    NO_CAUSAL_DECISION_CLOCK: (
        "Replace the joint committed-error anytime posterior by the marginal "
        "prefix error-mode posterior; preserve learned emissions, endpoint "
        "scores, procedure transport, and execution-mode dynamics."
    ),
    NO_OPEN_WORLD_INNOVATION: (
        "Set the posterior-predictive open-world innovation probability to "
        "zero while retaining the learned emissions, legal-successor law, "
        "and observer missed-event closure."
    ),
    POSTERIOR_NORMAL_HISTORY: (
        "Replace unconditional candidate-history writes by their exact "
        "posterior expectation under the completed normal-mark probability; "
        "retain all learned emissions and procedure-state transport."
    ),
    IDENTIFIABLE_MODE_SUPPORT: (
        "Restrict an exhaustive partition to the normal/error modes observed "
        "by its training ontology, removing an unidentifiable correction "
        "state while retaining learned emissions and procedure transport."
    ),
    NO_STATE_SEPARATION: (
        "Write every completed genuine observation into procedural progress "
        "using the same event-conditioned successor law, including events "
        "inferred as errors or corrections; retain factual history, learned "
        "emissions, execution modes, and null-proposal handling."
    ),
}


def parse_inference_ablations(name: str) -> tuple[str, ...]:
    """Split a composed intervention name ``a+b`` into validated parts.

    A single registered name is returned unchanged as a one-element tuple, so
    every existing artifact keeps its meaning.  Composed names apply each part
    in turn; the parts act on disjoint elements of the frozen filter.
    """

    parts = tuple(part.strip() for part in str(name).split("+") if part.strip())
    if not parts:
        raise ValueError("empty inference ablation")
    for part in parts:
        validate_inference_ablation(part)
    if len(parts) > 1 and FULL_INFERENCE in parts:
        raise ValueError("full inference cannot be composed with an ablation")
    if len(set(parts)) != len(parts):
        raise ValueError("repeated inference ablation")
    return parts


def validate_inference_ablation(name: str) -> str:
    """Return a validated registered intervention name."""

    if name not in INFERENCE_ABLATIONS:
        raise ValueError(f"unknown inference ablation: {name}")
    return name


@torch.no_grad()
def configure_filter_ablation(filter_module: nn.Module, name: str) -> None:
    """Apply a parameter-free intervention to a loaded filter in place."""

    name = validate_inference_ablation(name)
    if name in {
        FULL_INFERENCE,
        NO_FACTUAL_HISTORY,
        NO_CROSS_ACTOR_HISTORY,
        NO_PROCEDURE_FRONTIER,
        NO_PERSISTENT_PROCEDURE_FRONTIER,
        NO_OPEN_WORLD_INNOVATION,
        POSTERIOR_NORMAL_HISTORY,
    }:
        return
    if name == NO_CAUSAL_DECISION_CLOCK:
        raise ValueError("the causal decision clock is a model-level readout")
    if name == NO_STATE_SEPARATION:
        setattr(filter_module, "advance_all_genuine_modes", True)
        return
    transition = getattr(filter_module, "transition_probability", None)
    initial = getattr(filter_module, "initial_probability", None)
    if name == NO_INSERTION_STATE:
        genuine = getattr(filter_module, "prior_genuine_probability", None)
        if not isinstance(genuine, Tensor):
            raise ValueError(
                "the selected topology has no insertion state to remove"
            )
        genuine.fill_(1.0)
        return
    if name == IDENTIFIABLE_MODE_SUPPORT:
        if not all(isinstance(value, Tensor) for value in (transition, initial)):
            raise TypeError("the selected model does not expose execution modes")
        if transition.shape != (len(MODE_NAMES), len(MODE_NAMES)) or (
            initial.shape != (len(MODE_NAMES),)
        ):
            raise ValueError("execution-mode filter buffers are malformed")
        transition[:, CORRECTION].zero_()
        row_sum = transition.sum(dim=-1, keepdim=True)
        if torch.any(row_sum <= 0):
            raise ValueError("observable execution modes have no transition support")
        transition.div_(row_sum)
        initial[CORRECTION] = 0.0
        if initial.sum() <= 0:
            raise ValueError("observable execution modes have no initial support")
        initial.div_(initial.sum())
        return
    if not all(isinstance(value, Tensor) for value in (transition, initial)):
        raise TypeError("the selected model does not expose execution-mode dynamics")
    if transition.ndim != 2 or initial.shape != (transition.shape[1],):
        raise ValueError("execution-mode filter buffers are malformed")
    transition.copy_(initial.unsqueeze(0).expand_as(transition))


@torch.no_grad()
def configure_model_ablation(model: nn.Module, name: str) -> None:
    """Apply one registered intervention at the correct model level."""

    name = validate_inference_ablation(name)
    if name == NO_CROSS_ACTOR_HISTORY:
        energy = getattr(model, "energy", None)
        cross = getattr(energy, "other_role_map", None)
        if not isinstance(cross, nn.Linear) or cross.bias is not None:
            raise TypeError("the selected model has no cross-actor history map")
        with torch.no_grad():
            cross.weight.zero_()
        return
    if name == NO_CAUSAL_DECISION_CLOCK:
        enabled = getattr(model, "use_causal_decision_clock", None)
        if enabled is not True:
            raise ValueError(
                "the selected topology has no causal renewal decision clock"
            )
        setattr(model, "use_causal_decision_clock", False)
        return
    if name == POSTERIOR_NORMAL_HISTORY:
        enabled = getattr(model, "posterior_normal_history", None)
        if enabled is not False:
            raise ValueError(
                "the selected model has no unconditional candidate history"
            )
        setattr(model, "posterior_normal_history", True)
        return
    filter_module = getattr(model, "filter", None)
    if not isinstance(filter_module, nn.Module):
        raise TypeError("the selected model does not expose a causal filter")
    configure_filter_ablation(filter_module, name)


def configure_transport_ablation(transport: nn.Module, name: str) -> None:
    """Apply an intervention whose target is the frozen procedure transport."""

    if name != NO_OPEN_WORLD_INNOVATION:
        return
    innovation = getattr(transport, "innovation_probability", None)
    if (
        not isinstance(innovation, Tensor)
        or innovation.numel() != 1
        or not torch.isfinite(innovation).all()
        or torch.any((innovation < 0) | (innovation >= 1))
    ):
        raise TypeError("the selected transport has no valid innovation law")
    # Some query observations have exactly zero support under the finite
    # reference language.  Retain only the dtype's numerical support floor so
    # log-domain filtering stays defined; this is not a probabilistic model
    # choice and cannot be tuned from data.
    innovation.fill_(torch.finfo(innovation.dtype).tiny)


def static_frontier_probability(transport: nn.Module, reference: Tensor) -> Tensor:
    """Build a parameter-free bag-of-legal-frontiers distribution.

    Single-actor transports expose one successor-support vector.  Bimanual
    transports expose one per fixed role; their union defines whether a
    frontier admits any legal next event.  Binary support is used so graph
    branching multiplicity cannot act as an unregistered weight.
    """

    support = getattr(transport, "frontier_successor_support", None)
    if isinstance(support, Tensor):
        combined = support
    else:
        role_support = tuple(
            getattr(transport, f"frontier_successor_support_{role}", None)
            for role in range(2)
        )
        if not all(isinstance(value, Tensor) for value in role_support):
            raise TypeError("transport does not expose legal frontier support")
        combined = role_support[0] + role_support[1]
    if combined.ndim != 1 or not torch.isfinite(combined).all():
        raise ValueError("legal frontier support is malformed")
    legal = (combined > 0).to(device=reference.device, dtype=reference.dtype)
    if legal.sum() <= 0:
        raise ValueError("expert language contains no legal successor frontier")
    return legal / legal.sum()


def project_to_static_frontier(belief: Tensor, frontier: Tensor) -> Tensor:
    """Erase procedure progress while preserving every other latent marginal."""

    if belief.ndim not in {3, 5} or frontier.shape != (belief.shape[0],):
        raise ValueError("frontier projection and filter belief do not align")
    if (
        not torch.isfinite(belief).all()
        or torch.any(belief < 0)
        or belief.sum() <= 0
        or not torch.isfinite(frontier).all()
        or torch.any(frontier < 0)
        or frontier.sum() <= 0
    ):
        raise ValueError("frontier projection received an invalid probability")
    marginal = belief.sum(dim=0)
    normalized_frontier = frontier.to(belief) / frontier.sum().to(belief)
    shape = (len(normalized_frontier),) + (1,) * marginal.ndim
    projected = normalized_frontier.reshape(shape) * marginal.unsqueeze(0)
    return projected / projected.sum()


def marginalize_frontier_log_likelihood(
    log_likelihood: Tensor, frontier: Tensor
) -> Tensor:
    """Remove the frontier variable from a complete explanation likelihood.

    The penultimate likelihood axis is the latent frontier and the final axis
    is the explanation.  The exact log mixture is repeated across the now
    inert storage axis so existing mode/lifecycle filters can be reused without
    changing their learned parameters.
    """

    if log_likelihood.ndim < 2 or frontier.shape != (log_likelihood.shape[-2],):
        raise ValueError("frontier likelihood and mixture do not align")
    if (
        torch.isnan(log_likelihood).any()
        or torch.isposinf(log_likelihood).any()
        or not torch.isfinite(frontier).all()
        or torch.any(frontier < 0)
        or frontier.sum() <= 0
    ):
        raise ValueError("frontier likelihood mixture is malformed")
    probability = frontier.to(log_likelihood)
    probability = probability / probability.sum()
    log_probability = torch.log(probability).masked_fill(
        probability <= 0, -torch.inf
    )
    shape = (1,) * (log_likelihood.ndim - 2) + (len(probability), 1)
    collapsed = torch.logsumexp(
        log_likelihood + log_probability.reshape(shape), dim=-2
    )
    return collapsed.unsqueeze(-2).expand_as(log_likelihood)

