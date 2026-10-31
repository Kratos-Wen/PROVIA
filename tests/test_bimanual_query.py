from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from provia.evaluation.bimanual_query import (
    ROLE_EVALUATION_ONLY_FIELDS,
    ROLE_INFERENCE_FIELDS,
    ROLE_QUERY_PREFIX_SCHEMA,
    ROLE_QUERY_SCHEMA,
    load_role_query_artifact,
    load_role_query_prefix_registry,
    role_inference_view,
)
from provia.automaton.decoded_events import DecodedEvent
from provia.observer.factors import EVENT_FACTORS
from provia.training.role_transport_fit import _decoded_events
from provia.training.common import _sha256


def test_role_query_artifacts_require_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"development")
    query = tmp_path / "query.pt"
    torch.save(
        {
            "metadata": {
                "schema": ROLE_QUERY_SCHEMA,
                "source_examples_sha256": _sha256(source),
                "split": "test",
                "parameters_fitted": False,
                "query_annotations_used_by_candidate_decoder": False,
                "query_annotations_used_for_evaluation_only": True,
                "test_split_accessed": True,
            },
            "test": [{"video_id": "video"}],
        },
        query,
    )
    _, sequence = load_role_query_artifact(
        query, source_examples=source, split="test"
    )
    assert sequence[0]["video_id"] == "video"

    root = tmp_path / "prefix"
    root.mkdir()
    evidence = root / "video.npz"
    np.savez(evidence, value=np.asarray([1.0]))
    registry = {
        "schema": ROLE_QUERY_PREFIX_SCHEMA,
        "query_examples_sha256": _sha256(query),
        "source_examples_sha256": _sha256(source),
        "split": "test",
        "causal": True,
        "parameters_fitted": False,
        "annotations_used_to_compute_evidence": False,
        "test_split_accessed": True,
        "videos": {
            "video": {"file": evidence.name, "sha256": _sha256(evidence)}
        },
    }
    (root / "registry.json").write_text(json.dumps(registry))
    loaded = load_role_query_prefix_registry(
        root,
        query_examples=query,
        source_examples=source,
        split="test",
    )
    assert "video" in loaded["videos"]


def test_role_query_rejects_candidate_decoder_label_access(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"development")
    query = tmp_path / "query.pt"
    torch.save(
        {
            "metadata": {
                "schema": ROLE_QUERY_SCHEMA,
                "source_examples_sha256": _sha256(source),
                "split": "test",
                "parameters_fitted": False,
                "query_annotations_used_by_candidate_decoder": True,
                "query_annotations_used_for_evaluation_only": True,
                "test_split_accessed": True,
            },
            "test": [{"video_id": "video"}],
        },
        query,
    )
    with pytest.raises(ValueError):
        load_role_query_artifact(query, source_examples=source, split="test")


def test_target_free_query_decoding_does_not_read_stream_annotations() -> None:
    event = DecodedEvent(
        role=0,
        start_index=1,
        onset_index=2,
        end_index=3,
        factor_log_probability={name: (0.0,) for name in EVENT_FACTORS},
    )
    labels = {name: (f"{name}_value",) for name in EVENT_FACTORS}
    decoded = _decoded_events(  # type: ignore[arg-type]
        object(), (event,), labels, attach_targets=False
    )
    assert decoded[0].correctness == -1
    assert decoded[0].error_types == ()


def test_role_inference_view_is_invariant_to_query_gt() -> None:
    allowed = {
        "video_id": "v1",
        "start_index": torch.tensor([1, 3]),
        "onset_index": torch.tensor([2, 4]),
        "end_index": torch.tensor([2, 5]),
        "role_index": torch.tensor([0, 1]),
        "statistic": torch.randn(2, 7),
        "factor_verb": torch.softmax(torch.randn(2, 3), dim=-1),
        "factor_noun": torch.softmax(torch.randn(2, 4), dim=-1),
        "factor_tool": torch.softmax(torch.randn(2, 2), dim=-1),
    }
    first = {
        **allowed,
        "origin_target": torch.tensor([True, False]),
        "mode_target": torch.tensor([0, -1]),
        "native_assignment": torch.tensor([0, -1]),
        "native_error_types": ((), ()),
        "native_intervals": ({"mode": 0},),
    }
    perturbed = {
        **allowed,
        "origin_target": torch.tensor([False, True]),
        "mode_target": torch.tensor([2, 1]),
        "native_assignment": torch.tensor([-1, 8]),
        "native_error_types": (("changed",),),
        "native_intervals": ({"mode": 1}, {"mode": 1}),
    }

    first_view = role_inference_view(first)
    perturbed_view = role_inference_view(perturbed)

    assert set(first_view) == ROLE_INFERENCE_FIELDS
    assert not (set(first_view) & ROLE_EVALUATION_ONLY_FIELDS)
    assert set(perturbed_view) == set(first_view)
    for name in ROLE_INFERENCE_FIELDS:
        left = first_view[name]
        right = perturbed_view[name]
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        else:
            assert left == right
    with pytest.raises(TypeError):
        first_view["native_intervals"] = ()  # type: ignore[index]
