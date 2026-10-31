import torch

from provia.model.mode_filter import (jeffreys_observed_explanation_probability, joint_explanation_nll, observed_explanation_probability, observed_explanation_target, prior_preserving_case_control_log_score)
































def test_joint_log_score_and_gradients_are_finite() -> None:
    probability = torch.tensor(
        ((0.7, 0.1, 0.1, 0.1), (0.1, 0.2, 0.6, 0.1)),
        requires_grad=True,
    )
    loss = joint_explanation_nll(
        probability,
        torch.tensor((False, True)),
        torch.tensor((-1, 1)),
    )
    torch.testing.assert_close(loss.detach(), -torch.log(torch.tensor(0.42)) / 2.0)
    loss.backward()
    assert probability.grad is not None
    assert torch.isfinite(probability.grad).all()


def test_case_control_score_preserves_uniform_prior_and_has_finite_gradients() -> None:
    probability = torch.tensor(
        ((0.7, 0.1, 0.1, 0.1), (0.1, 0.2, 0.6, 0.1)),
        requires_grad=True,
    )
    origin = torch.tensor((False, True))
    mode = torch.tensor((-1, 1))
    numerator, exposure = prior_preserving_case_control_log_score(
        probability,
        origin,
        mode,
        torch.full((4,), 0.25),
    )
    torch.testing.assert_close(
        numerator / exposure,
        joint_explanation_nll(probability, origin, mode),
    )
    numerator.backward()
    assert probability.grad is not None
    assert torch.isfinite(probability.grad).all()




def test_coarse_ontology_marginalizes_unannotated_correction() -> None:
    latent = torch.tensor(
        ((0.1, 0.2, 0.3, 0.4), (0.4, 0.1, 0.2, 0.3))
    )
    observed = observed_explanation_probability(
        latent, correction_supervised=False
    )
    torch.testing.assert_close(
        observed,
        torch.tensor(((0.1, 0.6, 0.3), (0.4, 0.4, 0.2))),
    )
    target = observed_explanation_target(
        torch.tensor((True, True, False)),
        torch.tensor((0, 2, -1)),
        correction_supervised=False,
    )
    torch.testing.assert_close(target, torch.tensor((1, 1, 0)))


def test_coarse_case_control_score_is_proper_for_observed_categories() -> None:
    latent = torch.tensor(
        ((0.2, 0.3, 0.1, 0.4), (0.1, 0.2, 0.6, 0.1)),
        requires_grad=True,
    )
    origin = torch.tensor((True, True))
    mode = torch.tensor((0, 1))
    prior = jeffreys_observed_explanation_probability(
        origin, mode, correction_supervised=False
    )
    assert prior.shape == (3,)
    numerator, exposure = prior_preserving_case_control_log_score(
        latent,
        origin,
        mode,
        prior,
        correction_supervised=False,
    )
    assert torch.isfinite(numerator / exposure)
    numerator.backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
    # Normal and correction are observationally indistinguishable here: only
    # their sum can affect the supervised score.
    torch.testing.assert_close(latent.grad[0, 1], latent.grad[0, 3])










