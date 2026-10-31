"""Bimanual filtering with effect-committed procedure state.

The decision clock and the effect clock are different stopping times.  A
prefix may identify an error and raise a warning, but no hypothesized normal
event is allowed to rewrite accepted procedure progress before its predicted
effect completion.  At completion, current soft semantics and the observed
consequence define one normalized reference/open-world likelihood and its
conditional state transition.  The likelihood is consumed once and the state
effect is applied once.

This file contains the single-actor and bimanual reductions of the same latent
step.  It introduces no score-fusion coefficient, phase target,
dataset head, duration threshold, or validation-selected gate.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor


from .mode_filter import CORRECTION, ERROR, NORMAL
from .interval_transport import ContinuationTransport
from .role_filter import (
    CORRECTION_LIFECYCLE,
    ERROR_LIFECYCLE,
    INACTIVE,
    INSERTION,
    NORMAL_LIFECYCLE,
    UNRESOLVED,
    RoleFilter,
)




class EffectCommittedRoleFilter(RoleFilter):
    """Bimanual latent decisions whose normal effects wait for completion."""

    def condition_normal(
        self,
        previous_probability: Tensor,
        role: int,
        normal_likelihood_ratio: Tensor,
    ) -> Tensor:
        previous = self._validate(previous_probability)
        canonical = self._canonical(previous, role)
        states = len(canonical)
        ratio = normal_likelihood_ratio.to(canonical)
        if (
            ratio.shape != (states,)
            or not torch.isfinite(ratio).all()
            or torch.any(ratio <= 0)
        ):
            raise ValueError("normal likelihood ratio must align with frontiers")
        following = canonical.clone()
        following[:, :, :, NORMAL_LIFECYCLE, :] *= ratio[:, None, None, None]
        return self._validate(self._restore(following, role))

    def _finalize_role_unnormalized(
        self,
        previous_probability: Tensor,
        role: int,
        factor_probability: Mapping[str, Tensor],
        transport: ContinuationTransport,
        *,
        node_effect_log_bayes_factor: Tensor | None = None,
    ) -> Tensor:
        previous = self._validate(previous_probability, normalize=False)
        canonical = self._canonical(previous, role)
        mixed, _, _, _ = transport.predictive_reference_statistics(
            factor_probability,
            role,
            node_effect_log_bayes_factor=node_effect_log_bayes_factor,
        )
        if canonical[:, :, :, UNRESOLVED, :].sum() > 2e-6:
            raise ValueError("effect completion cannot retain unresolved mass")
        following = torch.zeros_like(canonical)
        following[:, :, :, INACTIVE, :] += canonical[:, :, :, INACTIVE, :]
        following[:, :, :, INACTIVE, :] += canonical[:, :, :, INSERTION, :]
        following[:, :, :, INACTIVE, :] += canonical[
            :, :, :, ERROR_LIFECYCLE, :
        ]
        following[:, :, :, INACTIVE, :] += canonical[
            :, :, :, CORRECTION_LIFECYCLE, :
        ]
        advance_all_genuine_modes = bool(
            getattr(self, "advance_all_genuine_modes", False)
        )
        if advance_all_genuine_modes:
            for mode, lifecycle in (
                (ERROR, ERROR_LIFECYCLE),
                (CORRECTION, CORRECTION_LIFECYCLE),
            ):
                source = canonical[:, mode, :, lifecycle, :]
                # The transport contract consumes mass conditioned on the
                # same reference/open-world likelihood as the normal branch.
                conditioned = source * mixed.to(canonical)[:, None, None]
                destination = transport.conditional_predictive_advance_probability(
                    conditioned,
                    role,
                    factor_probability,
                    node_effect_log_bayes_factor=node_effect_log_bayes_factor,
                )
                following[:, mode, :, INACTIVE, :] -= source
                following[:, mode, :, INACTIVE, :] += destination
        # Test forbidden support directly; a difference of two large float32
        # reductions can otherwise report mass that is not present.
        non_normal_mode = torch.cat(
            (
                canonical[:, :NORMAL, :, NORMAL_LIFECYCLE, :].reshape(-1),
                canonical[:, NORMAL + 1 :, :, NORMAL_LIFECYCLE, :].reshape(-1),
            )
        ).sum()
        if non_normal_mode > 2e-6:
            raise RuntimeError("normal lifecycle and execution mode disagree")
        normal = (
            canonical[:, NORMAL, :, NORMAL_LIFECYCLE, :]
            * mixed.to(canonical)[:, None, None]
        )
        destination = transport.conditional_predictive_advance_probability(
            normal,
            role,
            factor_probability,
            node_effect_log_bayes_factor=node_effect_log_bayes_factor,
        )
        following[:, NORMAL, :, INACTIVE, :] += destination
        restored = self._restore(following, role)
        if restored.sum() <= 0 or not torch.isfinite(restored).all():
            raise ValueError("effect completion has no finite posterior support")
        return restored

    def finalize_joint(
        self,
        previous_probability: Tensor,
        observations: Mapping[
            int, tuple[Mapping[str, Tensor], Tensor | None]
        ],
        transport: ContinuationTransport,
    ) -> Tensor:
        if not observations or any(role not in {0, 1} for role in observations):
            raise ValueError("effect completion requires one or two fixed roles")

        def apply(source: Tensor, role: int) -> Tensor:
            factors, effect = observations[role]
            return self._finalize_role_unnormalized(
                source,
                role,
                factors,
                transport,
                node_effect_log_bayes_factor=effect,
            )

        roles = sorted(observations)
        if len(roles) == 1:
            following = apply(previous_probability, roles[0])
        elif len(roles) == 2:
            following = 0.5 * (
                apply(apply(previous_probability, roles[0]), roles[1])
                + apply(apply(previous_probability, roles[1]), roles[0])
            )
        else:
            raise ValueError("only two roles are defined")
        return self._validate(following)


__all__ = [
    "EffectCommittedRoleFilter",
]
