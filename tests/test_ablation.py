from types import SimpleNamespace

import pytest
import torch

from provia.model.ablation import (POSTERIOR_NORMAL_HISTORY, configure_model_ablation, marginalize_frontier_log_likelihood, project_to_static_frontier, static_frontier_probability)


TRANSITION = (
    (0.8, 0.1, 0.1),
    (0.1, 0.8, 0.1),
    (0.2, 0.1, 0.7),
)
INITIAL = (0.7, 0.2, 0.1)


def test_static_frontier_uses_binary_union_support_without_branch_weighting() -> None:
    single = SimpleNamespace(
        frontier_successor_support=torch.tensor((0.2, 0.0, 0.9, 0.0))
    )
    expected = torch.tensor((0.5, 0.0, 0.5, 0.0))
    torch.testing.assert_close(
        static_frontier_probability(single, torch.tensor(0.0)), expected
    )

    roles = SimpleNamespace(
        frontier_successor_support_0=torch.tensor((0.0, 0.4, 0.0, 0.0)),
        frontier_successor_support_1=torch.tensor((0.1, 0.0, 0.8, 0.0)),
    )
    torch.testing.assert_close(
        static_frontier_probability(roles, torch.tensor(0.0)),
        torch.tensor((1.0 / 3, 1.0 / 3, 1.0 / 3, 0.0)),
    )


def test_frontier_projection_preserves_every_nonfrontier_marginal() -> None:
    generator = torch.Generator().manual_seed(4)
    for shape in ((4, 3, 8), (4, 3, 3, 8, 8)):
        belief = torch.rand(shape, generator=generator)
        belief = belief / belief.sum()
        frontier = torch.tensor((0.2, 0.0, 0.3, 0.5))
        projected = project_to_static_frontier(belief, frontier)
        torch.testing.assert_close(projected.sum(dim=0), belief.sum(dim=0))
        torch.testing.assert_close(
            projected.flatten(1).sum(dim=1), frontier
        )
        torch.testing.assert_close(projected.sum(), torch.tensor(1.0))


def test_frontier_likelihood_marginalization_is_exact_and_state_independent() -> None:
    likelihood = torch.tensor(
        (
            ((0.2, -0.1, 0.7, -0.4), (1.2, 0.3, -0.2, 0.5)),
            ((-0.3, 0.4, 0.1, 0.8), (0.6, -0.7, 0.9, 0.0)),
        )
    )
    frontier = torch.tensor((0.25, 0.75))
    actual = marginalize_frontier_log_likelihood(likelihood, frontier)
    expected = torch.logsumexp(
        likelihood + torch.log(frontier).reshape(1, 2, 1), dim=1
    )
    torch.testing.assert_close(actual[:, 0], expected)
    torch.testing.assert_close(actual[:, 1], expected)


















def test_posterior_normal_history_is_registered_at_model_level() -> None:
    model = SimpleNamespace(posterior_normal_history=False)
    configure_model_ablation(model, POSTERIOR_NORMAL_HISTORY)
    assert model.posterior_normal_history is True
    with pytest.raises(ValueError, match="unconditional candidate history"):
        configure_model_ablation(model, POSTERIOR_NORMAL_HISTORY)


