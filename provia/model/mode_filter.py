"""Reliability-aware causal state inference for predicted event streams.

The observer can emit a genuine event or an insertion.  Genuine events carry
one of three execution modes (normal, error, correction); insertions leave the
execution state unchanged.  Origin and mode are inferred in one normalized
Bayes update, so an observer confidence is never multiplied with an anomaly
score after the fact.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn




MODE_NAMES = ("normal", "error", "correction")
EXPLANATION_NAMES = ("insertion", *MODE_NAMES)
COARSE_EXPLANATION_NAMES = ("insertion", "non_error", "error")
NORMAL, ERROR, CORRECTION = range(len(MODE_NAMES))


class ReliabilityAwareModeFilter(nn.Module):
    """Marginalize insertion uncertainty and N/E/C dynamics exactly.

    ``origin_log_likelihood_ratio`` is evidence for genuine versus insertion.
    ``mode_log_likelihood_ratio`` contains observation evidence for each
    genuine execution mode.  Fold-training posterior-predictive probabilities
    provide the priors.  An insertion preserves the previous mode posterior;
    a genuine event applies the event-indexed transition law.
    """

    def __init__(
        self,
        transition_probability: Tensor | Sequence[Sequence[float]],
        initial_probability: Tensor | Sequence[float],
        genuine_probability: Tensor | float,
    ) -> None:
        super().__init__()
        transition = torch.as_tensor(transition_probability, dtype=torch.float32)
        initial = torch.as_tensor(initial_probability, dtype=torch.float32)
        genuine = torch.as_tensor(genuine_probability, dtype=torch.float32)
        if transition.shape != (len(MODE_NAMES), len(MODE_NAMES)):
            raise ValueError("mode transition probability must have shape [3,3]")
        if initial.shape != (len(MODE_NAMES),):
            raise ValueError("initial mode probability must have shape [3]")
        if genuine.numel() != 1:
            raise ValueError("genuine-event probability must be scalar")
        if (
            not torch.isfinite(transition).all()
            or not torch.isfinite(initial).all()
            or not torch.isfinite(genuine).all()
            or torch.any(transition <= 0)
            or torch.any(initial <= 0)
            or not 0.0 < float(genuine) < 1.0
        ):
            raise ValueError("filter priors must be finite and strictly positive")
        transition = transition / transition.sum(dim=-1, keepdim=True)
        initial = initial / initial.sum()
        self.register_buffer("transition_probability", transition)
        self.register_buffer("initial_probability", initial)
        self.register_buffer("prior_genuine_probability", genuine.reshape(()))

def joint_explanation_nll(
    explanation_probability: Tensor,
    origin_target: Tensor,
    mode_target: Tensor,
) -> Tensor:
    """Proper joint log score with mode observed only for genuine events."""

    target = explanation_target(origin_target, mode_target)
    if explanation_probability.ndim != 2 or explanation_probability.shape != (
        len(target),
        len(EXPLANATION_NAMES),
    ):
        raise ValueError("explanation probability must have shape [events,4]")
    tiny = torch.finfo(explanation_probability.dtype).tiny
    return -explanation_probability.clamp_min(tiny).log().gather(
        -1, target.to(explanation_probability.device).unsqueeze(-1)
    ).mean()


def explanation_target(origin_target: Tensor, mode_target: Tensor) -> Tensor:
    """Map observed origin and conditional mode labels to four explanations."""

    if origin_target.ndim != 1:
        raise ValueError("origin targets must be one-dimensional")
    events = len(origin_target)
    if origin_target.shape != (events,) or mode_target.shape != (events,):
        raise ValueError("joint targets must align with event explanations")
    origin = origin_target.to(dtype=torch.bool)
    mode = mode_target.to(device=origin.device, dtype=torch.long)
    if torch.any(mode[origin] < 0) or torch.any(mode[origin] >= len(MODE_NAMES)):
        raise ValueError("genuine events require a valid execution mode")
    if torch.any(mode[~origin] != -1):
        raise ValueError("insertion events must use mode target -1")
    target = torch.zeros(events, dtype=torch.long, device=origin.device)
    target[origin] = 1 + mode[origin]
    return target


def observed_explanation_probability(
    explanation_probability: Tensor,
    *,
    correction_supervised: bool,
) -> Tensor:
    """Project latent explanations onto the dataset's observed label ontology.

    Some benchmarks annotate only whether a genuine event is erroneous.  On
    those datasets, a correction is not an observed negative class; it is a
    latent non-error explanation.  Marginalizing ``normal`` and ``correction``
    is the exact likelihood for that coarsened supervision and avoids treating
    an unannotated recovery state as if it had been exhaustively labeled.
    """

    if explanation_probability.ndim != 2 or explanation_probability.shape[-1] != len(
        EXPLANATION_NAMES
    ):
        raise ValueError("explanation probability must have shape [events,4]")
    if correction_supervised:
        return explanation_probability
    return torch.stack(
        (
            explanation_probability[:, 0],
            explanation_probability[:, 1] + explanation_probability[:, 3],
            explanation_probability[:, 2],
        ),
        dim=-1,
    )


def observed_explanation_target(
    origin_target: Tensor,
    mode_target: Tensor,
    *,
    correction_supervised: bool,
) -> Tensor:
    """Map labels to the identity or binary-correctness observation space."""

    latent = explanation_target(origin_target, mode_target)
    if correction_supervised:
        return latent
    target = torch.zeros_like(latent)
    genuine = origin_target.to(device=latent.device, dtype=torch.bool)
    mode = mode_target.to(device=latent.device, dtype=torch.long)
    target[genuine] = 1
    target[genuine & (mode == ERROR)] = 2
    return target


def jeffreys_observed_explanation_probability(
    origin_target: Tensor,
    mode_target: Tensor,
    *,
    correction_supervised: bool,
) -> Tensor:
    """Training-fold posterior-predictive law in the observed ontology."""

    target = observed_explanation_target(
        origin_target,
        mode_target,
        correction_supervised=correction_supervised,
    )
    classes = (
        len(EXPLANATION_NAMES)
        if correction_supervised
        else len(COARSE_EXPLANATION_NAMES)
    )
    counts = torch.bincount(target.cpu(), minlength=classes).to(torch.float64)
    posterior = counts + 0.5
    return (posterior / posterior.sum()).to(torch.float32)


def prior_preserving_case_control_log_score(
    explanation_probability: Tensor,
    origin_target: Tensor,
    mode_target: Tensor,
    class_probability: Tensor,
    *,
    correction_supervised: bool = True,
) -> tuple[Tensor, Tensor]:
    """Return numerator and exposure of a calibrated balanced log score.

    Let ``q`` be the posterior under the true fold-training class prior and let
    ``w_c=1/pi_c``. Case-control sampling changes the posterior to
    ``q_cc(c|x) = w_c q(c|x) / sum_j w_j q(j|x)``. Scoring ``q_cc`` with the
    same case-control weights is strictly proper for the original ``q`` while
    giving each explanation equal expected exposure. Thus rare error and
    correction evidence affects optimization without replacing the calibrated
    mode and insertion priors used by the causal filter at inference.
    """

    probability = observed_explanation_probability(
        explanation_probability,
        correction_supervised=correction_supervised,
    )
    target = observed_explanation_target(
        origin_target,
        mode_target,
        correction_supervised=correction_supervised,
    ).to(
        probability.device
    )
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
        raise ValueError("explanation class probabilities must be positive and normalized")
    probability = probability.clamp_min(torch.finfo(probability.dtype).tiny)
    weight = class_probability.to(probability).reciprocal()
    weight = weight / weight.mean()
    transformed = probability * weight.unsqueeze(0)
    transformed = transformed / transformed.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(probability.dtype).tiny
    )
    sample_weight = weight.index_select(0, target)
    selected = transformed.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    numerator = -(sample_weight * selected.log()).sum()
    return numerator, sample_weight.sum()
