from __future__ import annotations

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from provia.observer.completion_process import (
    MarkedCompletionProcess,
    decode_completion_posterior_medians,
)
from provia.training.train_single_actor_observer import (
    _completion_dilations,
    _event_nll,
)
from provia.training.completion_metrics import (_binary_metrics)


LABELS = {
    "verb": ("<unk>", "pick"),
    "noun": ("<unk>", "<none>"),
    "tool": ("<unk>", "<none>"),
}


def test_zero_initialized_completion_rate_equals_empirical_base() -> None:
    model = MarkedCompletionProcess(4, LABELS, 0.2)
    features = torch.randn(6, 4)
    change = torch.zeros(6, 2, 4)
    timestamps = torch.arange(6, dtype=torch.float32) * 0.5
    output = model(features, change, timestamps)
    assert torch.allclose(
        output.intensity_per_second,
        torch.full((6,), 0.2),
        atol=1e-6,
    )
    probability = model.event_probability(output.intensity_per_second, timestamps)
    assert probability[0] == 0
    assert torch.allclose(
        probability[1:], torch.full((5,), 1.0 - torch.exp(torch.tensor(-0.1)))
    )


def test_completion_decoder_uses_fixed_posterior_median() -> None:
    probability = torch.tensor([0.0, 0.2, 0.2, 0.2, 0.2, 0.2])
    logits = {
        name: torch.zeros(len(probability), len(labels))
        for name, labels in LABELS.items()
    }
    priors = {
        name: torch.full((len(labels),), 1.0 / len(labels))
        for name, labels in LABELS.items()
    }
    events = decode_completion_posterior_medians(probability, logits, priors)
    # 0.8^4 <= 0.5, while 0.8^3 > 0.5.
    assert [event.onset_index for event in events] == [4]
    assert all(event.role == 0 for event in events)


def test_completion_model_is_causal_under_future_perturbation() -> None:
    torch.manual_seed(4)
    model = MarkedCompletionProcess(4, LABELS, 0.1).eval()
    features = torch.randn(8, 4)
    change = torch.randn(8, 2, 4)
    timestamps = torch.arange(8, dtype=torch.float32) / 4
    first = model(features, change, timestamps)
    altered_features = features.clone()
    altered_change = change.clone()
    altered_features[5:] += 100
    altered_change[5:] -= 100
    second = model(altered_features, altered_change, timestamps)
    assert torch.equal(first.intensity_per_second[:5], second.intensity_per_second[:5])
    for name in LABELS:
        assert torch.equal(first.factor_logits[name][:5], second.factor_logits[name][:5])


def test_long_context_completion_model_is_causal() -> None:
    torch.manual_seed(5)
    model = MarkedCompletionProcess(
        4,
        LABELS,
        0.1,
        model_dim=8,
        dilations=(1, 2),
        dropout=0.0,
    ).eval()
    features = torch.randn(10, 4)
    change = torch.randn(10, 2, 4)
    timestamps = torch.arange(10, dtype=torch.float32) / 4
    first = model(features, change, timestamps)
    altered_features = features.clone()
    altered_change = change.clone()
    altered_features[6:] += 100
    altered_change[6:] -= 100
    second = model(altered_features, altered_change, timestamps)
    assert torch.equal(first.intensity_per_second[:6], second.intensity_per_second[:6])
    for name in LABELS:
        assert torch.equal(first.factor_logits[name][:6], second.factor_logits[name][:6])


@pytest.mark.parametrize("dilations", (None, (1, 2, 4)))
def test_completion_stream_step_matches_causal_batch_forward(dilations) -> None:
    torch.manual_seed(13)
    kwargs = (
        {}
        if dilations is None
        else {"model_dim": 8, "dilations": dilations, "dropout": 0.0}
    )
    model = MarkedCompletionProcess(4, LABELS, 0.1, **kwargs).eval()
    features = torch.randn(12, 4)
    change = torch.randn(12, 2, 4)
    timestamps = torch.arange(12, dtype=torch.float32) / 8
    batch = model(features, change, timestamps)
    state = None
    intensities = []
    factors = {name: [] for name in LABELS}
    for index in range(len(features)):
        evidence, state = model.step(
            features[index : index + 1],
            change[index : index + 1],
            timestamps[index : index + 1],
            state,
        )
        intensities.append(evidence.intensity_per_second)
        for name in LABELS:
            factors[name].append(evidence.factor_logits[name])
    assert torch.allclose(torch.cat(intensities), batch.intensity_per_second, atol=1e-6)
    for name in LABELS:
        assert torch.allclose(
            torch.cat(factors[name]), batch.factor_logits[name], atol=1e-6
        )


def test_completion_dilations_minimally_cover_training_inter_event_gap() -> None:
    streams = (
        SimpleNamespace(event_indices=torch.tensor((3, 10))),
        SimpleNamespace(event_indices=torch.tensor((2, 6))),
    )
    dilations = _completion_dilations(streams)  # type: ignore[arg-type]
    receptive_field = 1 + 2 * sum(dilations)
    previous_field = 1 + 2 * sum(dilations[:-1])
    assert receptive_field >= 8
    assert previous_field < 8


def test_completion_likelihood_is_finite_and_differentiable() -> None:
    raw = torch.tensor([-4.0, -3.0, -2.0, -1.0], requires_grad=True)
    intensity = torch.nn.functional.softplus(raw)
    timestamps = torch.arange(4, dtype=torch.float32) / 8
    loss, probability, target = _event_nll(
        intensity, timestamps, torch.tensor([2])
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(probability).all()
    assert target.tolist() == [False, True, False]
    loss.backward()
    assert torch.isfinite(raw.grad).all()


def test_completion_binary_metrics_report_auditable_operating_point() -> None:
    result = _binary_metrics(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.1, 0.8, 0.7, 0.9]),
        threshold=0.5,
    )
    assert result["accuracy_at_operating_point"] == pytest.approx(0.75)
    assert result["precision_at_operating_point"] == pytest.approx(2 / 3)
    assert result["recall_at_operating_point"] == pytest.approx(1.0)
    assert result["confusion"] == {
        "true_positive": 2,
        "false_positive": 1,
        "true_negative": 1,
        "false_negative": 0,
    }


