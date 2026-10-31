import math

import torch

from provia.model.step_effects import ExpertEffectPosterior








def test_effect_artifact_round_trip() -> None:
    model = ExpertEffectPosterior.fit(
        ("a", "b"),
        torch.tensor([[0.0, 1.0], [0.5, 1.5], [2.0, 3.0]]),
        torch.tensor([0, 0, 1]),
    )
    restored = ExpertEffectPosterior.from_artifact(
        model.artifact({"test_split_accessed": False})
    )
    assert restored.node_ids == model.node_ids
    for name, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)


def test_natural_parameter_evaluation_matches_direct_gaussian() -> None:
    generator = torch.Generator().manual_seed(17)
    samples = torch.randn(12, 9, generator=generator)
    node = torch.arange(12) % 4
    model = ExpertEffectPosterior.fit(tuple(f"n{i}" for i in range(4)), samples, node)
    query = torch.randn(5, 9, generator=generator)
    standardized = (query - model.global_mean) / model.global_scale
    direct = -0.5 * (
        math.log(2.0 * math.pi)
        + model.node_predictive_variance.log()[None, :, :]
        + (
            standardized[:, None, :] - model.node_location[None, :, :]
        ).square()
        / model.node_predictive_variance[None, :, :]
    ).mean(dim=-1)
    natural = -0.5 * (
        standardized.square() @ model._node_precision.T
        - 2.0 * standardized @ model._node_precision_location.T
        + model._node_log_normalizer[None, :]
    ) / model.statistic_dim
    assert torch.allclose(natural, direct, atol=2e-6, rtol=2e-6)






