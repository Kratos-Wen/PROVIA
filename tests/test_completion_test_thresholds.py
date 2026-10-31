from __future__ import annotations

import json

import pytest

from provia.evaluation.evaluate_single_actor import (
    FROC_BUDGETS,
    _load_frozen_operating_thresholds,
)
from provia.model.intent_state import (
    CAUSAL_CANDIDATE_DECISION_CLOCK,
)


def _artifact() -> dict[str, object]:
    checkpoints = {}
    for checkpoint in ("completion", "anytime"):
        checkpoints[checkpoint] = {
            "free_response": {
                "operating_points": {
                    budget: {"validation_threshold": index / 10.0}
                    for index, budget in enumerate(FROC_BUDGETS, start=1)
                }
            }
        }
    return {
        "split": "validation",
        "test_split_accessed": False,
        "checkpoint_sha256": "checkpoint",
        "examples_sha256": "examples",
        "inference_ablation": "full",
        "anytime_readout": CAUSAL_CANDIDATE_DECISION_CLOCK,
        "metrics": {
            "future_frames_used_by_score": False,
            "checkpoints": checkpoints,
        },
    }


def test_test_thresholds_are_loaded_only_from_matching_validation(tmp_path) -> None:
    path = tmp_path / "validation.json"
    path.write_text(json.dumps(_artifact()))
    value = _load_frozen_operating_thresholds(
        path,
        checkpoint_sha256="checkpoint",
        examples_sha256="examples",
        inference_ablation="full",
        anytime_readout=CAUSAL_CANDIDATE_DECISION_CLOCK,
    )
    assert value["completion"]["0.1"] == pytest.approx(0.1)
    assert value["anytime"]["2.0"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("split", "test"),
        ("test_split_accessed", True),
        ("checkpoint_sha256", "different"),
        ("examples_sha256", "different"),
        ("inference_ablation", "no_procedure_frontier"),
        ("anytime_readout", "different"),
    ),
)
def test_test_thresholds_reject_mismatched_provenance(
    tmp_path, field: str, value: object
) -> None:
    artifact = _artifact()
    artifact[field] = value
    path = tmp_path / "validation.json"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="incompatible"):
        _load_frozen_operating_thresholds(
            path,
            checkpoint_sha256="checkpoint",
            examples_sha256="examples",
            inference_ablation="full",
            anytime_readout=CAUSAL_CANDIDATE_DECISION_CLOCK,
        )
