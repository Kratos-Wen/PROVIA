"""Empirical-Bayes visual effect laws induced from successful demonstrations.

Each continuation node receives a diagonal posterior-predictive distribution
over causal path statistics.  Node means are partially pooled through a
random-effects model whose within- and between-node variances are estimated
from the demonstrations themselves.  This keeps nodes with one observation
well defined without a hand-chosen shrinkage coefficient.

The resulting evidence is a normalized composite log Bayes factor between the
current viable-successor mixture and the task-level expert background.  It is
an observation term for the causal state-space model, not a separately
thresholded anomaly score.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn


EFFECT_EVIDENCE_NAMES = (
    "viable_successor_log_probability",
    "expert_background_log_probability",
    "viable_successor_log_bayes_factor",
)


class ExpertEffectPosterior(nn.Module):
    """Data-derived node-conditional posterior predictive effect model."""

    def __init__(
        self,
        node_ids: Sequence[str],
        global_mean: Tensor,
        global_scale: Tensor,
        node_location: Tensor,
        node_predictive_variance: Tensor,
        node_observation_count: Tensor,
        within_node_variance: Tensor,
        between_node_variance: Tensor,
        *,
        source_statistic_dim: int | None = None,
        coordinate_index: Tensor | None = None,
        background_node_probability: Tensor | None = None,
        background_location: Tensor | None = None,
        background_variance: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.node_ids = tuple(str(value) for value in node_ids)
        if not self.node_ids or len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("effect nodes must be non-empty and unique")
        nodes = len(self.node_ids)
        if global_mean.ndim != 1 or global_scale.shape != global_mean.shape:
            raise ValueError("effect normalization must be one-dimensional")
        dimension = len(global_mean)
        if node_location.shape != (nodes, dimension) or (
            node_predictive_variance.shape != (nodes, dimension)
        ):
            raise ValueError("node effect parameters have incompatible shapes")
        if node_observation_count.shape != (nodes,):
            raise ValueError("node observation counts have an incompatible shape")
        if within_node_variance.shape != (dimension,) or (
            between_node_variance.shape != (dimension,)
        ):
            raise ValueError("effect variance components have incompatible shapes")
        finite = (
            global_mean,
            global_scale,
            node_location,
            node_predictive_variance,
            within_node_variance,
            between_node_variance,
        )
        if any(not value.is_floating_point() for value in finite):
            raise ValueError("effect parameters must be floating point")
        if any(not torch.isfinite(value).all() for value in finite):
            raise ValueError("effect parameters must be finite")
        if (
            torch.any(global_scale <= 0)
            or torch.any(node_predictive_variance <= 0)
            or torch.any(within_node_variance <= 0)
            or torch.any(between_node_variance <= 0)
        ):
            raise ValueError("effect scales and variances must be positive")
        if node_observation_count.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        } or torch.any(node_observation_count < 0):
            raise ValueError("node observation counts must be non-negative integers")
        source_dimension = (
            dimension if source_statistic_dim is None else int(source_statistic_dim)
        )
        coordinates = (
            torch.arange(dimension, dtype=torch.long)
            if coordinate_index is None
            else coordinate_index.detach().long().clone()
        )
        if (
            source_dimension < dimension
            or coordinates.shape != (dimension,)
            or torch.any(coordinates < 0)
            or torch.any(coordinates >= source_dimension)
            or len(torch.unique(coordinates)) != dimension
        ):
            raise ValueError("effect-coordinate projection is malformed")
        if background_node_probability is not None:
            background_node_probability = (
                background_node_probability.detach().float().clone()
            )
            if (
                background_node_probability.shape != (nodes,)
                or not torch.isfinite(background_node_probability).all()
                or torch.any(background_node_probability <= 0)
                or not torch.isclose(
                    background_node_probability.sum(),
                    background_node_probability.new_tensor(1.0),
                    atol=2e-5,
                    rtol=2e-5,
                )
            ):
                raise ValueError("expert background-node probability is malformed")
        if (background_location is None) != (background_variance is None):
            raise ValueError("background location and variance must be paired")
        if background_node_probability is not None and background_location is not None:
            raise ValueError("effect posterior has two incompatible background laws")
        if background_location is not None:
            background_location = background_location.detach().float().clone()
            assert background_variance is not None
            background_variance = background_variance.detach().float().clone()
            if (
                background_location.shape != (dimension,)
                or background_variance.shape != (dimension,)
                or not torch.isfinite(background_location).all()
                or not torch.isfinite(background_variance).all()
                or torch.any(background_variance <= 0)
            ):
                raise ValueError("generic transition background is malformed")
        self.statistic_dim = dimension
        self.source_statistic_dim = source_dimension
        self.register_buffer("global_mean", global_mean.detach().float().clone())
        self.register_buffer("global_scale", global_scale.detach().float().clone())
        self.register_buffer("node_location", node_location.detach().float().clone())
        self.register_buffer(
            "node_predictive_variance",
            node_predictive_variance.detach().float().clone(),
        )
        self.register_buffer(
            "node_observation_count",
            node_observation_count.detach().long().clone(),
        )
        self.register_buffer(
            "within_node_variance", within_node_variance.detach().float().clone()
        )
        self.register_buffer(
            "between_node_variance", between_node_variance.detach().float().clone()
        )
        self.register_buffer("_coordinate_index", coordinates, persistent=False)
        self.register_buffer(
            "background_node_probability", background_node_probability
        )
        self.register_buffer("background_location", background_location)
        self.register_buffer("background_variance", background_variance)
        # Algebraically equivalent Gaussian natural parameters avoid
        # materializing a [batch,nodes,dimensions] tensor. They are derived
        # from the posterior and are intentionally not serialized.
        precision = self.node_predictive_variance.reciprocal()
        self.register_buffer("_node_precision", precision, persistent=False)
        self.register_buffer(
            "_node_precision_location",
            precision * self.node_location,
            persistent=False,
        )
        self.register_buffer(
            "_node_log_normalizer",
            (
                math.log(2.0 * math.pi)
                + self.node_predictive_variance.log()
                + self.node_location.square() * precision
            ).sum(dim=-1),
            persistent=False,
        )

    @classmethod
    def fit(
        cls,
        node_ids: Sequence[str],
        statistic: Tensor,
        node_index: Tensor,
    ) -> "ExpertEffectPosterior":
        """Fit a diagonal random-effects posterior without tuned constants."""

        nodes = len(node_ids)
        if nodes <= 0 or len(set(str(value) for value in node_ids)) != nodes:
            raise ValueError("effect nodes must be non-empty and unique")
        if statistic.ndim != 2 or not len(statistic):
            raise ValueError("effect statistics must have shape [events,dimensions]")
        if node_index.shape != (len(statistic),) or node_index.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("effect node indices must align with observations")
        if (
            not torch.isfinite(statistic).all()
            or torch.any(node_index < 0)
            or torch.any(node_index >= nodes)
        ):
            raise ValueError("effect observations are malformed")
        values = statistic.detach().to(dtype=torch.float64, device="cpu")
        index = node_index.detach().to(dtype=torch.long, device="cpu")
        mean = values.mean(dim=0)
        centered = values - mean
        variance = centered.square().mean(dim=0)
        positive = variance[variance > 0]
        reference = positive.mean() if len(positive) else variance.new_tensor(1.0)
        floor = torch.finfo(variance.dtype).eps * reference
        scale = variance.clamp_min(floor).sqrt()
        standardized = centered / scale

        count = torch.bincount(index, minlength=nodes)
        total = standardized.new_zeros(nodes, standardized.shape[-1])
        total.index_add_(0, index, standardized)
        observed = count > 0
        sample_mean = total / count.clamp_min(1).to(total).unsqueeze(-1)
        residual = standardized - sample_mean.index_select(0, index)
        within_sum = residual.square().sum(dim=0)
        within_degrees = int(len(standardized) - int(observed.sum()))
        if within_degrees > 0:
            within = within_sum / float(within_degrees)
        else:
            within = standardized.square().mean(dim=0)
        total_variance = standardized.square().mean(dim=0)
        variance_reference = total_variance[total_variance > 0]
        variance_floor = (
            torch.finfo(total_variance.dtype).eps
            * (
                variance_reference.mean()
                if len(variance_reference)
                else total_variance.new_tensor(1.0)
            )
        )
        within = within.clamp_min(variance_floor)
        between = (total_variance - within).clamp_min(variance_floor)

        precision = between.reciprocal().unsqueeze(0) + (
            count.to(total).unsqueeze(-1) / within.unsqueeze(0)
        )
        posterior_variance = precision.reciprocal()
        location = posterior_variance * (
            total / within.unsqueeze(0)
        )
        predictive = within.unsqueeze(0) + posterior_variance
        return cls(
            node_ids,
            mean.float(),
            scale.float(),
            location.float(),
            predictive.float(),
            count,
            within.float(),
            between.float(),
        )

    @classmethod
    def fit_transition_effects(
        cls,
        node_ids: Sequence[str],
        path_statistic: Tensor,
        node_index: Tensor,
        input_dim: int,
    ) -> "ExpertEffectPosterior":
        """Fit endpoint-displacement effects with an expert-mixture null.

        The finite-variation path statistic is partitioned by construction:
        endpoint displacement is the state effect, while oriented path shape,
        quadratic variation, and duration remain execution dynamics.  The
        denominator is the Jeffreys posterior-predictive mixture over all
        expert action effects, not a hand-selected reference distribution.
        """

        input_dimension = int(input_dim)
        expected = 3 * input_dimension + 1
        if input_dimension <= 0 or path_statistic.ndim != 2 or (
            path_statistic.shape[-1] != expected
        ):
            raise ValueError("transition effects require a full path statistic")
        fitted = cls.fit(
            node_ids,
            path_statistic[:, :input_dimension],
            node_index,
        )
        count = fitted.node_observation_count.to(dtype=torch.float64)
        background = (count + 0.5) / (count.sum() + 0.5 * len(count))
        return cls(
            fitted.node_ids,
            fitted.global_mean,
            fitted.global_scale,
            fitted.node_location,
            fitted.node_predictive_variance,
            fitted.node_observation_count,
            fitted.within_node_variance,
            fitted.between_node_variance,
            source_statistic_dim=expected,
            coordinate_index=torch.arange(input_dimension),
            background_node_probability=background.float(),
        )

    @staticmethod
    def _composite_normal_log_probability(
        value: Tensor, location: Tensor, variance: Tensor
    ) -> Tensor:
        return -0.5 * (
            math.log(2.0 * math.pi)
            + variance.log()
            + (value - location).square() / variance
        ).mean(dim=-1)

    def _background_log_probability(
        self, standardized: Tensor, node_log_probability: Tensor
    ) -> Tensor:
        if self.background_node_probability is not None:
            return torch.logsumexp(
                self.background_node_probability.to(standardized).log()[None, :]
                + node_log_probability,
                dim=-1,
            )
        if self.background_location is not None:
            assert self.background_variance is not None
            return self._composite_normal_log_probability(
                standardized,
                self.background_location.to(standardized),
                self.background_variance.to(standardized),
            )
        return self._composite_normal_log_probability(
            standardized,
            torch.zeros_like(standardized),
            torch.ones_like(standardized),
        )

    def node_log_bayes_factor(self, statistic: Tensor) -> Tensor:
        """Return each expert node's effect likelihood ratio to the null.

        This is the un-marginalized form of :meth:`evidence`.  It is useful
        when a state-space transition must retain the successor-node identity
        while incorporating an endpoint effect.  The denominator is exactly
        the same posterior-predictive expert background used by
        :meth:`evidence`; no extra calibration scale or fusion weight is
        introduced.
        """

        if statistic.ndim == 1:
            statistic = statistic.unsqueeze(0)
        batch = len(statistic)
        if statistic.shape == (batch, self.source_statistic_dim):
            statistic = statistic.index_select(
                -1, self._coordinate_index.to(device=statistic.device)
            )
        if statistic.shape != (batch, self.statistic_dim) or not torch.isfinite(
            statistic
        ).all():
            raise ValueError("effect statistic has an incompatible shape")
        standardized = (
            statistic - self.global_mean.to(statistic)
        ) / self.global_scale.to(statistic)
        node_quadratic = (
            standardized.square() @ self._node_precision.to(statistic).T
            - 2.0
            * standardized
            @ self._node_precision_location.to(statistic).T
            + self._node_log_normalizer.to(statistic)[None, :]
        )
        node_log_probability = -0.5 * node_quadratic / float(self.statistic_dim)
        background = self._background_log_probability(
            standardized, node_log_probability
        )
        return node_log_probability - background.unsqueeze(-1)

    def artifact(self, metadata: Mapping[str, object]) -> dict[str, object]:
        normative_effect = self.background_node_probability is not None
        contrastive_effect = self.background_location is not None
        return {
            "metadata": {
                **dict(metadata),
                "schema": (
                    "contrastive_transition_effect_posterior"
                    if contrastive_effect
                    else (
                        "normative_transition_effect_posterior"
                        if normative_effect
                        else "expert_effect_random_effects"
                    )
                ),
                "node_ids": list(self.node_ids),
                "statistic_dim": self.statistic_dim,
                "source_statistic_dim": self.source_statistic_dim,
                "coordinate_index": self._coordinate_index.tolist(),
                "effect_evidence": list(EFFECT_EVIDENCE_NAMES),
                "manual_shrinkage_hyperparameters": 0,
                
            },
            "state": self.state_dict(),
        }

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, object]) -> "ExpertEffectPosterior":
        metadata = artifact.get("metadata")
        state = artifact.get("state")
        if not isinstance(metadata, Mapping) or not isinstance(state, Mapping):
            raise ValueError("effect artifact is malformed")
        schema = metadata.get("schema")
        if schema not in {
            "expert_effect_random_effects",
            "normative_transition_effect_posterior",
            "contrastive_transition_effect_posterior",
        }:
            raise ValueError("unknown expert-effect schema")
        node_ids = metadata.get("node_ids")
        if not isinstance(node_ids, Sequence) or isinstance(node_ids, (str, bytes)):
            raise ValueError("effect artifact lacks node IDs")
        required = {
            "global_mean",
            "global_scale",
            "node_location",
            "node_predictive_variance",
            "node_observation_count",
            "within_node_variance",
            "between_node_variance",
        }
        if schema == "normative_transition_effect_posterior":
            required.add("background_node_probability")
        if schema == "contrastive_transition_effect_posterior":
            required.update(("background_location", "background_variance"))
        if set(state) != required or any(not isinstance(state[name], Tensor) for name in required):
            raise ValueError("effect artifact state is malformed")
        source_statistic_dim = int(
            metadata.get("source_statistic_dim", state["global_mean"].numel())
        )
        coordinate_index = metadata.get("coordinate_index")
        if coordinate_index is None and schema == "expert_effect_random_effects":
            coordinate_index = list(range(state["global_mean"].numel()))
        if not isinstance(coordinate_index, Sequence) or isinstance(
            coordinate_index, (str, bytes)
        ):
            raise ValueError("effect artifact lacks its coordinate projection")
        return cls(
            tuple(str(value) for value in node_ids),
            state["global_mean"],  # type: ignore[index]
            state["global_scale"],  # type: ignore[index]
            state["node_location"],  # type: ignore[index]
            state["node_predictive_variance"],  # type: ignore[index]
            state["node_observation_count"],  # type: ignore[index]
            state["within_node_variance"],  # type: ignore[index]
            state["between_node_variance"],  # type: ignore[index]
            source_statistic_dim=source_statistic_dim,
            coordinate_index=torch.tensor(coordinate_index, dtype=torch.long),
            background_node_probability=(
                state["background_node_probability"]  # type: ignore[index]
                if schema == "normative_transition_effect_posterior"
                else None
            ),
            background_location=(
                state["background_location"]  # type: ignore[index]
                if schema == "contrastive_transition_effect_posterior"
                else None
            ),
            background_variance=(
                state["background_variance"]  # type: ignore[index]
                if schema == "contrastive_transition_effect_posterior"
                else None
            ),
        )
