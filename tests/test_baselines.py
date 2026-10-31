from __future__ import annotations

import numpy as np
import torch

from provia.baselines.models import (
    CausalTCN,
    MiniROAD,
    MistSenseRGB,
)
from provia.baselines.run_single_actor import (
    CAUSAL_TCN_RECEPTIVE_FIELD,
    _causal_window,
    _select_threshold,
)
from provia.baselines.run_bimanual import (
    RoleActionVocabulary,
    _frame_categories,
    _rasterize_event_scores,
    _role_window,
)
from provia.data.role_stream import CompetitiveRoleStream
from provia.data.streaming_events import StreamingEventInterval
from provia.observer.factors import EVENT_FACTORS


def test_causal_window_cannot_read_future_features() -> None:
    features = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    expected = _causal_window(features, 5, 4)
    changed = features.clone()
    changed[6:] = -1000
    assert torch.equal(expected, _causal_window(changed, 5, 4))
    assert CAUSAL_TCN_RECEPTIVE_FIELD == 127


def test_fast_threshold_selection_matches_exhaustive_definition() -> None:
    generator = np.random.default_rng(9)
    target = generator.integers(0, 2, size=71)
    score = np.round(generator.random(71), 2)
    candidates = np.unique(np.r_[np.nextafter(0.0, 1.0), 0.5, score])
    best_key = (-1.0, -1.0, -1.0)
    expected = 0.5
    for threshold in candidates:
        prediction = score >= threshold
        values = []
        for class_value in (False, True):
            truth = target.astype(bool) == class_value
            predicted = prediction == class_value
            true_positive = int(np.sum(truth & predicted))
            false_positive = int(np.sum(~truth & predicted))
            false_negative = int(np.sum(truth & ~predicted))
            values.append(
                2
                * true_positive
                / max(2 * true_positive + false_positive + false_negative, 1)
            )
        recall = float(np.mean(prediction[target.astype(bool)]))
        key = (float(np.mean(values)), recall, -abs(float(threshold) - 0.5))
        if key > best_key:
            best_key = key
            expected = float(threshold)
    assert _select_threshold(target, score) == expected


def test_neural_baseline_shapes() -> None:
    torch.manual_seed(0)
    value = torch.randn(2, 12, 16)
    assert CausalTCN(16, width=8, dilations=(1, 2))(value).shape == (2,)
    assert MiniROAD(16, 5, embedding_dim=8, hidden_dim=8)(value).shape == (
        2,
        12,
        5,
    )
    assert MistSenseRGB(16, width=16, queries=4, layers=1)(value).shape == (2,)


def _role_stream() -> CompetitiveRoleStream:
    intervals = (
        StreamingEventInterval("a", 0, 0, 1, 2, 0, ()),
        StreamingEventInterval("b", 1, 3, 4, 5, 1, ("wrong_tool",)),
    )
    factors = {
        name: torch.tensor(
            [[1, 0], [1, 0], [1, 0], [0, 2], [0, 2], [0, 2]],
            dtype=torch.long,
        )
        for name in EVENT_FACTORS
    }
    return CompetitiveRoleStream(
        episode_id="data:video",
        video_id="video",
        global_features=torch.arange(12, dtype=torch.float32).reshape(6, 2),
        role_features=torch.arange(24, dtype=torch.float32).reshape(6, 2, 2),
        role_change_features=None,
        frame_ids=torch.arange(6),
        timestamps_sec=torch.arange(6, dtype=torch.float64),
        phase_targets=torch.zeros(6, 2, dtype=torch.long),
        factor_targets=factors,
        factor_weights=torch.zeros(6, 2),
        anomaly_target=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.bool),
        recovery_target=torch.zeros(6, dtype=torch.bool),
        intervals=intervals,
    )


def test_role_window_is_role_specific_and_past_only() -> None:
    stream = _role_stream()
    expected = _role_window(stream, 2, 0, 3)
    changed = CompetitiveRoleStream(
        **{
            **stream.__dict__,
            "global_features": stream.global_features.clone(),
            "role_features": stream.role_features.clone(),
        }
    )
    changed.global_features[3:] = -1000
    changed.role_features[3:] = -1000
    assert torch.equal(expected, _role_window(changed, 2, 0, 3))
    assert not torch.equal(expected, _role_window(stream, 2, 1, 3))


def test_role_vocabulary_and_frame_categories_preserve_hand_and_subtype() -> None:
    stream = _role_stream()
    second = CompetitiveRoleStream(
        **{
            **stream.__dict__,
            "intervals": (
                StreamingEventInterval("b", 1, 3, 4, 5, 0, ()),
            ),
        }
    )
    vocabulary = RoleActionVocabulary.from_normal_streams((stream, second))
    assert vocabulary.values == ((0, 1, 1, 1), (1, 2, 2, 2))
    categories = _frame_categories(stream)
    assert categories[:3] == ((), (), ())
    assert categories[3:] == (("wrong_tool",),) * 3


def test_role_score_raster_starts_at_predicted_onset_not_approach() -> None:
    event = type(
        "Event",
        (),
        {"start_index": 1, "onset_index": 3, "end_index": 5},
    )()
    score, support = _rasterize_event_scores(
        7, (event,), np.asarray([0.8], dtype=np.float64)
    )
    np.testing.assert_array_equal(support, [0, 0, 0, 1, 1, 1, 0])
    np.testing.assert_allclose(score, [0.0, 0.0, 0.0, 0.8, 0.8, 0.8, 0.0])
