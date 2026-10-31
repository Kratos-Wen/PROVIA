"""Predictive-state transport for single-stream renewal cells.

Candidate topology determines whether observer origin is identifiable.  A
single-stream renewal decoder emits contiguous, non-overlapping cells over its
decoded temporal support; each cell consumes one global observation interval
rather than proposing an additional role-specific event.  This module therefore
models only execution mode (normal context, error, or correction).  Asynchronous
role proposals retain the separate null-origin law implemented by the bimanual
model because overlapping roles can otherwise count the same
temporal progress twice.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from .commitment_filter import ExhaustivePartitionCommitmentFilter
from .intent_state import (
    CoupledIntentPredictiveStateCompletionModel,
    CoupledIntentPredictiveStateMarkEnergy,
)
from .mode_filter import (
    CORRECTION,
    ERROR,
    EXPLANATION_NAMES,
    MODE_NAMES,
    NORMAL,
)


PROVIA_SINGLE_ACTOR_ARCHITECTURE = "provia_single_actor"
PROVIA_SHARED_SUCCESSOR_ARCHITECTURE = "provia_single_actor_shared_successor"
PARTITION_CANDIDATE_TOPOLOGY = "exhaustive_nonoverlapping_partition"
PARTITION_APPLICABILITY_SCHEMA = (
    "fold_training_partition_applicability_jeffreys"
)


def partition_mode_target(
    origin_target: Tensor,
    mode_target: Tensor,
    *,
    correction_supervised: bool,
) -> Tensor:
    """Map frozen renewal cells to the benchmark-observed mode ontology.

    An unmatched cell is normal open-world context, not an asserted procedural
    action.  Its reference/open-world posterior still decides whether the
    accepted procedure frontier can move.
    """

    if origin_target.ndim != 1 or mode_target.shape != origin_target.shape:
        raise ValueError("partition targets must be aligned vectors")
    origin = origin_target.to(dtype=torch.bool)
    mode = mode_target.to(device=origin.device, dtype=torch.long)
    if torch.any(mode[origin] < 0) or torch.any(mode[origin] >= len(MODE_NAMES)):
        raise ValueError("matched partition cells require a valid execution mode")
    if torch.any(mode[~origin] != -1):
        raise ValueError("unmatched partition cells must use mode target -1")
    latent = torch.where(origin, mode, torch.full_like(mode, NORMAL))
    if correction_supervised:
        return latent
    return (latent == ERROR).to(dtype=torch.long)


def observed_partition_mode_probability(
    explanation_probability: Tensor,
    *,
    correction_supervised: bool,
) -> Tensor:
    """Project the compatibility output onto the learned N/E/C ontology."""

    if (
        explanation_probability.ndim != 2
        or explanation_probability.shape[-1] != len(EXPLANATION_NAMES)
        or not torch.isfinite(explanation_probability).all()
        or torch.any(explanation_probability < 0)
    ):
        raise ValueError("partition explanation probability must have shape [events,4]")
    insertion = explanation_probability[:, 0]
    if not torch.allclose(
        insertion, torch.zeros_like(insertion), atol=1e-7, rtol=0.0
    ):
        raise ValueError("a non-overlapping renewal stream cannot carry insertion mass")
    mode = explanation_probability[:, 1:]
    if not torch.allclose(
        mode.sum(dim=-1), torch.ones(len(mode), device=mode.device, dtype=mode.dtype),
        atol=2e-5, rtol=2e-5,
    ):
        raise ValueError("partition mode probabilities must be normalized")
    if correction_supervised:
        return mode
    return torch.stack(
        (mode[:, NORMAL] + mode[:, CORRECTION], mode[:, ERROR]), dim=-1
    )


def jeffreys_partition_mode_probability(
    origin_target: Tensor,
    mode_target: Tensor,
    *,
    correction_supervised: bool,
) -> Tensor:
    """Fold-training posterior-predictive class law for partition modes."""

    target = partition_mode_target(
        origin_target,
        mode_target,
        correction_supervised=correction_supervised,
    )
    classes = len(MODE_NAMES) if correction_supervised else 2
    counts = torch.bincount(target.cpu(), minlength=classes).to(torch.float64)
    posterior = counts + 0.5
    return (posterior / posterior.sum()).to(torch.float32)


def partition_mode_nll(
    explanation_probability: Tensor,
    origin_target: Tensor,
    mode_target: Tensor,
    *,
    correction_supervised: bool,
) -> Tensor:
    """Proper log score for a renewal cell's observed execution mode."""

    probability = observed_partition_mode_probability(
        explanation_probability,
        correction_supervised=correction_supervised,
    )
    target = partition_mode_target(
        origin_target,
        mode_target,
        correction_supervised=correction_supervised,
    ).to(probability.device)
    tiny = torch.finfo(probability.dtype).tiny
    return -probability.clamp_min(tiny).log().gather(
        -1, target.unsqueeze(-1)
    ).mean()


def prior_preserving_partition_mode_log_score(
    explanation_probability: Tensor,
    origin_target: Tensor,
    mode_target: Tensor,
    class_probability: Tensor,
    *,
    correction_supervised: bool,
) -> tuple[Tensor, Tensor]:
    """Balanced proper score without introducing a null-origin class."""

    probability = observed_partition_mode_probability(
        explanation_probability,
        correction_supervised=correction_supervised,
    )
    target = partition_mode_target(
        origin_target,
        mode_target,
        correction_supervised=correction_supervised,
    ).to(probability.device)
    classes = probability.shape[-1]
    if (
        class_probability.shape != (classes,)
        or not torch.isfinite(class_probability).all()
        or torch.any(class_probability <= 0)
        or not torch.isclose(
            class_probability.sum(),
            class_probability.new_tensor(1.0),
            atol=1e-5,
        )
    ):
        raise ValueError("partition class probabilities must be positive and normalized")
    probability = probability.clamp_min(torch.finfo(probability.dtype).tiny)
    weight = class_probability.to(probability).reciprocal()
    weight = weight / weight.mean()
    transformed = probability * weight.unsqueeze(0)
    transformed = transformed / transformed.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(probability.dtype).tiny
    )
    sample_weight = weight.index_select(0, target)
    selected = transformed.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return -(sample_weight * selected.log()).sum(), sample_weight.sum()


class PartitionConditionedPredictiveStateMarkEnergy(
    CoupledIntentPredictiveStateMarkEnergy
):
    """Three-head N/E/C emission sharing one latent legal successor."""

    mode_likelihood_offset = 0
    has_insertion_likelihood = False

    def __init__(
        self,
        input_dim: int,
        factor_labels: Mapping[str, Sequence[str]],
        *,
        model_dim: int = 192,
    ) -> None:
        super().__init__(input_dim, factor_labels, model_dim=model_dim)
        self.mark_query = nn.Linear(
            self.model_dim,
            len(MODE_NAMES) * self.model_dim,
            bias=False,
        )
        self.mark_base = nn.Linear(self.model_dim, len(MODE_NAMES))
        nn.init.zeros_(self.mark_query.weight)
        nn.init.zeros_(self.mark_base.weight)
        nn.init.zeros_(self.mark_base.bias)


class ProviaSingleActorSharedSuccessor(
    CoupledIntentPredictiveStateCompletionModel
):
    """Topology-conditioned N/E/C model for one global renewal stream."""

    candidate_topology = PARTITION_CANDIDATE_TOPOLOGY

    def __init__(
        self,
        input_dim: int,
        factor_labels: Mapping[str, Sequence[str]],
        transition_probability: Tensor | Sequence[Sequence[float]],
        initial_probability: Tensor | Sequence[float],
        *,
        model_dim: int = 192,
    ) -> None:
        nn.Module.__init__(self)
        self.energy = PartitionConditionedPredictiveStateMarkEnergy(
            input_dim, factor_labels, model_dim=model_dim
        )
        self.filter = ExhaustivePartitionCommitmentFilter(
            transition_probability, initial_probability
        )


class ProviaSingleActor(
    ProviaSingleActorSharedSuccessor
):
    """Protect accepted procedure state from unverified visual evidence.

    The learned visual relation evaluates whether the current candidate fits a
    normative continuation. Accepted-state transport instead marginalizes
    only the reference-supported semantic/effect successor posterior. The two
    distributions have identical support and neural capacity, but visual
    compatibility cannot promote its own successor hypothesis into normative
    memory. This is the evidence-protection constraint used by PROVIA.
    """

    def __init__(
        self,
        input_dim: int,
        factor_labels: Mapping[str, Sequence[str]],
        transition_probability: Tensor | Sequence[Sequence[float]],
        initial_probability: Tensor | Sequence[float],
        *,
        model_dim: int = 192,
    ) -> None:
        super().__init__(
            input_dim,
            factor_labels,
            transition_probability,
            initial_probability,
            model_dim=model_dim,
        )
        self.energy.couple_successor = False

    @staticmethod
    def transport_node_log_bayes_factor(normal_node: Tensor) -> None:
        del normal_node
        return None




__all__ = [
    "PROVIA_SINGLE_ACTOR_ARCHITECTURE",
    "PROVIA_SINGLE_ACTOR_ARCHITECTURE",
    "PARTITION_APPLICABILITY_SCHEMA",
    "PARTITION_CANDIDATE_TOPOLOGY",
    "PROVIA_SHARED_SUCCESSOR_ARCHITECTURE",
    "ProviaSingleActor",
    "ProviaSingleActorSharedSuccessor",
    "PartitionConditionedPredictiveStateMarkEnergy",
    "jeffreys_partition_mode_probability",
    "observed_partition_mode_probability",
    "partition_mode_nll",
    "partition_mode_target",
    "prior_preserving_partition_mode_log_score",
]
