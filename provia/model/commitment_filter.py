"""Exact causal filtering for event proposal, commitment, and effect.

An event is not allowed to rewrite procedure state when it is first proposed.
The filter keeps its explanation latent until causal commitment evidence
arrives, applies the corresponding state transition at most once, and can
then refine that same explanation with post-commitment effect evidence.

The implementation is deliberately independent of a particular phase
observer.  A completion-only observer supplies commitment probability one at
its completion.  A streaming renewal observer can instead supply the
conditional commitment probability at every prefix.  No phase threshold or
score fusion is part of this state transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor, nn

from .mode_filter import ERROR, MODE_NAMES, NORMAL, ReliabilityAwareModeFilter


LIFECYCLE_NAMES = (
    "inactive",
    "insertion",
    "normal_uncommitted",
    "error_uncommitted",
    "correction_uncommitted",
    "normal_committed",
    "error_committed",
    "correction_committed",
)
INACTIVE, INSERTION = 0, 1


def _log_positive_mass(value: Tensor) -> Tensor:
    """Differentiate through positive mass while preserving exact zero support."""

    if not torch.is_floating_point(value):
        raise ValueError("probability mass must be floating point")
    tiny = torch.finfo(value.dtype).tiny
    return torch.log(value.clamp_min(tiny)).masked_fill(value <= 0, -torch.inf)


def uncommitted(mode: int) -> int:
    if not 0 <= int(mode) < len(MODE_NAMES):
        raise ValueError("unknown execution mode")
    return 2 + int(mode)


def committed(mode: int) -> int:
    if not 0 <= int(mode) < len(MODE_NAMES):
        raise ValueError("unknown execution mode")
    return 2 + len(MODE_NAMES) + int(mode)


@dataclass(frozen=True)
class CommitmentProcedureStep:
    """Posterior summary for one active single-actor event."""

    procedure_mode_lifecycle_probability: Tensor
    explanation_probability: Tensor
    procedure_probability: Tensor
    mode_probability: Tensor
    genuine_probability: Tensor
    commitment_probability: Tensor
    anomaly_probability: Tensor


class ReliabilityAwareCommitmentFilter(nn.Module):
    """Exact single-actor proposal--commitment--effect recursion."""

    def initial_belief(self, procedure_probability: Tensor) -> Tensor:
        if procedure_probability.ndim != 1:
            raise ValueError("initial procedure probability must have shape [states]")
        if (
            not torch.isfinite(procedure_probability).all()
            or torch.any(procedure_probability < 0)
            or procedure_probability.sum() <= 0
        ):
            raise ValueError("initial procedure probability is malformed")
        procedure = procedure_probability / procedure_probability.sum()
        belief = procedure.new_zeros(
            len(procedure), len(MODE_NAMES), len(LIFECYCLE_NAMES)
        )
        belief[:, :, INACTIVE] = (
            procedure[:, None] * self.initial_probability.to(procedure)[None, :]
        )
        return belief

    @staticmethod
    def _validate(probability: Tensor) -> Tensor:
        if probability.ndim != 3 or probability.shape[1:] != (
            len(MODE_NAMES),
            len(LIFECYCLE_NAMES),
        ):
            raise ValueError("lifecycle belief must have shape [states,3,8]")
        if (
            not torch.isfinite(probability).all()
            or torch.any(probability < 0)
            or probability.sum() <= 0
        ):
            raise ValueError("lifecycle belief is malformed")
        return probability / probability.sum()

    def summarize(self, probability: Tensor) -> CommitmentProcedureStep:
        belief = self._validate(probability)
        lifecycle = belief.sum(dim=(0, 1))
        explanation = torch.stack(
            (
                lifecycle[INSERTION],
                *(
                    lifecycle[uncommitted(mode)] + lifecycle[committed(mode)]
                    for mode in range(len(MODE_NAMES))
                ),
            )
        )
        commitment = torch.stack(
            tuple(lifecycle[committed(mode)] for mode in range(len(MODE_NAMES)))
        ).sum()
        return CommitmentProcedureStep(
            procedure_mode_lifecycle_probability=belief,
            explanation_probability=explanation,
            procedure_probability=belief.sum(dim=(1, 2)),
            mode_probability=belief.sum(dim=(0, 2)),
            genuine_probability=explanation[1:].sum(),
            commitment_probability=commitment,
            anomaly_probability=lifecycle[committed(ERROR)],
        )

    def decide(
        self,
        conditional_probability: Tensor | float,
        previous_probability: Tensor,
    ) -> CommitmentProcedureStep:
        """Commit explanation mass without applying its procedure effect.

        ``conditional_probability`` is the probability of becoming causally
        resolved at the current prefix, conditional on not having become
        resolved earlier.  This clock may fire before the action effect is
        complete.  The accepted frontier is therefore retained exactly; a
        later :meth:`finalize` call applies the procedure transition once.
        """

        previous = self._validate(previous_probability)
        probability = torch.as_tensor(
            conditional_probability,
            dtype=previous.dtype,
            device=previous.device,
        )
        if (
            probability.numel() != 1
            or not torch.isfinite(probability).all()
            or torch.any((probability < 0.0) | (probability > 1.0))
        ):
            raise ValueError(
                "conditional commitment probability must be a scalar in [0,1]"
            )
        value = probability.reshape(())
        following = torch.zeros_like(previous)
        following[:, :, INACTIVE] = previous[:, :, INACTIVE]
        following[:, :, INSERTION] = previous[:, :, INSERTION]
        for mode in range(len(MODE_NAMES)):
            following[:, :, committed(mode)] += previous[:, :, committed(mode)]
            source = previous[:, :, uncommitted(mode)]
            following[:, :, uncommitted(mode)] += (1.0 - value) * source
            # The lifecycle mode is the proposed destination mode.  Summing
            # the source-mode axis applies that transition without changing
            # the accepted procedure frontier.
            following[:, mode, committed(mode)] += value * source.sum(dim=1)
        return self.summarize(following)

    def finalize(
        self,
        previous_probability: Tensor,
        advance_probability: Callable[[Tensor], Tensor],
        reference_coverage_probability: Tensor | float = 1.0,
    ) -> Tensor:
        """Close an event and apply its procedure effect exactly once."""

        decided = self.decide(1.0, previous_probability)
        previous = decided.procedure_mode_lifecycle_probability
        coverage = torch.as_tensor(
            reference_coverage_probability,
            dtype=previous.dtype,
            device=previous.device,
        )
        if not torch.isfinite(coverage).all() or torch.any(
            (coverage < 0.0) | (coverage > 1.0)
        ):
            raise ValueError("reference coverage must lie in [0,1]")
        if coverage.numel() == 1:
            coverage = coverage.reshape(()).expand(len(previous))
        elif coverage.shape != (len(previous),):
            raise ValueError(
                "reference coverage must be scalar or frontier-conditioned"
            )
        following = torch.zeros_like(previous)
        following[:, :, INACTIVE] += previous[:, :, INACTIVE]
        following[:, :, INACTIVE] += previous[:, :, INSERTION]
        advance_all_genuine_modes = bool(
            getattr(self, "advance_all_genuine_modes", False)
        )
        for mode in range(len(MODE_NAMES)):
            source = previous[:, mode, committed(mode)]
            if mode == NORMAL or advance_all_genuine_modes:
                covered = coverage * source
                advanced = advance_probability(covered)
                if advanced.shape != source.shape:
                    raise ValueError("advanced frontier probability has wrong shape")
                if (
                    not torch.isfinite(advanced).all()
                    or torch.any(advanced < 0)
                    or advanced.sum() > covered.sum() + 2e-5
                ):
                    raise ValueError("advanced frontier probability is malformed")
                destination = advanced + (1.0 - coverage) * source
            else:
                destination = source
            following[:, mode, INACTIVE] += destination
        return self._validate(following)

class ExhaustivePartitionCommitmentFilter(ReliabilityAwareCommitmentFilter):
    """Three-mode recursion for an exhaustive, non-overlapping time partition.

    Every candidate cell consumes one non-overlapping interval of the decoded
    global stream, so it is not an additional optional role proposal.  The
    learned mark and probabilistic filter contain only normal context, error,
    and correction modes.  The inherited lifecycle tensor reserves its
    insertion coordinate solely for interface compatibility; this class never
    creates probability mass there and exposes no insertion prior or emission
    parameter.
    """

    candidate_topology = "exhaustive_nonoverlapping_partition"

    def __init__(
        self,
        transition_probability: Tensor | Sequence[Sequence[float]],
        initial_probability: Tensor | Sequence[float],
    ) -> None:
        nn.Module.__init__(self)
        template = ReliabilityAwareModeFilter(
            transition_probability, initial_probability, 0.5
        )
        self.register_buffer(
            "transition_probability", template.transition_probability.detach().clone()
        )
        self.register_buffer(
            "initial_probability", template.initial_probability.detach().clone()
        )

    @staticmethod
    def _mode_likelihood(
        mode_log_likelihood: Tensor, states: int
    ) -> Tensor:
        if mode_log_likelihood.shape == (len(MODE_NAMES),):
            likelihood = mode_log_likelihood.unsqueeze(0).expand(states, -1)
        elif mode_log_likelihood.shape == (states, len(MODE_NAMES)):
            likelihood = mode_log_likelihood
        else:
            raise ValueError(
                "partition mode likelihood must be global or frontier-conditioned"
            )
        if torch.isnan(likelihood).any() or torch.isposinf(likelihood).any():
            raise ValueError("partition mode likelihood is malformed")
        return likelihood

    def begin_from_log_likelihood(
        self,
        explanation_log_likelihood: Tensor,
        previous_probability: Tensor,
    ) -> CommitmentProcedureStep:
        """Begin one partition cell from a complete N/E/C likelihood."""

        previous = self._validate(previous_probability)
        if previous[:, :, 1:].sum() > 2e-6:
            raise ValueError("cannot begin a second single-actor event")
        states = len(previous)
        likelihood = self._mode_likelihood(explanation_log_likelihood, states)
        base = previous[:, :, INACTIVE]
        log_base = _log_positive_mass(base)
        log_following = torch.full_like(previous, -torch.inf)
        transition = self.transition_probability.to(base)
        for mode in range(len(MODE_NAMES)):
            log_following[:, :, uncommitted(mode)] = (
                log_base
                + torch.log(transition[:, mode])[None, :]
                + likelihood[:, mode, None].to(base)
            )
        normalizer = torch.logsumexp(log_following.flatten(), dim=0)
        if not torch.isfinite(normalizer):
            raise ValueError("partition mode likelihood has no support under the prior")
        return self.summarize(torch.exp(log_following - normalizer))

    def begin_many_from_log_likelihood(
        self,
        explanation_log_likelihood: Tensor,
        previous_probability: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Vectorize independent prefix reads for one partition cell."""

        previous = self._validate(previous_probability)
        if previous[:, :, 1:].sum() > 2e-6:
            raise ValueError("cannot begin a second single-actor event")
        states = len(previous)
        if (
            explanation_log_likelihood.ndim != 3
            or explanation_log_likelihood.shape[1:]
            != (states, len(MODE_NAMES))
            or not len(explanation_log_likelihood)
            or torch.isnan(explanation_log_likelihood).any()
            or torch.isposinf(explanation_log_likelihood).any()
        ):
            raise ValueError(
                "batched partition likelihood must have shape [time,states,3]"
            )
        likelihood = explanation_log_likelihood
        base = previous[:, :, INACTIVE]
        log_base = _log_positive_mass(base)
        log_following = likelihood.new_full(
            (
                len(likelihood),
                states,
                len(MODE_NAMES),
                len(LIFECYCLE_NAMES),
            ),
            -torch.inf,
        )
        transition = self.transition_probability.to(likelihood)
        for mode in range(len(MODE_NAMES)):
            log_following[:, :, :, uncommitted(mode)] = (
                log_base.unsqueeze(0)
                + torch.log(transition[:, mode]).reshape(1, 1, -1)
                + likelihood[:, :, mode].unsqueeze(-1)
            )
        normalizer = torch.logsumexp(
            log_following.flatten(start_dim=1), dim=1
        )
        if not torch.isfinite(normalizer).all():
            raise ValueError("one or more partition prefixes have no prior support")
        belief = torch.exp(
            log_following - normalizer.reshape(-1, 1, 1, 1)
        )
        lifecycle = belief.sum(dim=(1, 2))
        mode_probability = torch.stack(
            tuple(
                lifecycle[:, uncommitted(mode)]
                + lifecycle[:, committed(mode)]
                for mode in range(len(MODE_NAMES))
            ),
            dim=-1,
        )
        # The leading zero is an evaluation-interface coordinate, not a
        # modeled explanation.
        explanation = torch.cat(
            (mode_probability.new_zeros(len(mode_probability), 1), mode_probability),
            dim=-1,
        )
        return belief, explanation

