import json
from pathlib import Path

import pytest
import torch

from provia.evaluation.single_actor_query import (
    COMPLETION_EVALUATION_ONLY_FIELDS,
    COMPLETION_INFERENCE_FIELDS,
    COMPLETION_QUERY_PREFIX_SCHEMA,
    COMPLETION_QUERY_SCHEMA,
    completion_inference_view,
    load_completion_query_artifact,
    load_completion_query_prefix_registry,
)
from provia.training.common import _sha256


def _source(path: Path) -> None:
    torch.save({"metadata": {"test_split_accessed": False}}, path)


def test_query_examples_are_linked_and_evaluation_only(tmp_path: Path) -> None:
    source = tmp_path / "trainval.pt"
    query = tmp_path / "test.pt"
    _source(source)
    torch.save(
        {
            "metadata": {
                "schema": COMPLETION_QUERY_SCHEMA,
                "source_examples_sha256": _sha256(source),
                "split": "test",
                "parameters_fitted": False,
                "query_annotations_used_by_candidate_decoder": False,
                "query_annotations_used_for_evaluation_only": True,
                "test_split_accessed": True,
            },
            "test": [{"video_id": "v1"}],
        },
        query,
    )
    metadata, sequences = load_completion_query_artifact(
        query, source_examples=source, split="test"
    )
    assert metadata["parameters_fitted"] is False
    assert sequences[0]["video_id"] == "v1"


def test_query_examples_reject_candidate_decoding_from_annotations(tmp_path: Path) -> None:
    source = tmp_path / "trainval.pt"
    query = tmp_path / "test.pt"
    _source(source)
    torch.save(
        {
            "metadata": {
                "schema": COMPLETION_QUERY_SCHEMA,
                "source_examples_sha256": _sha256(source),
                "split": "test",
                "parameters_fitted": False,
                "query_annotations_used_by_candidate_decoder": True,
                "query_annotations_used_for_evaluation_only": True,
                "test_split_accessed": True,
            },
            "test": [{"video_id": "v1"}],
        },
        query,
    )
    with pytest.raises(ValueError, match="query artifact"):
        load_completion_query_artifact(query, source_examples=source, split="test")


def test_query_prefix_registry_is_immutable_and_annotation_free(tmp_path: Path) -> None:
    source = tmp_path / "trainval.pt"
    query = tmp_path / "test.pt"
    root = tmp_path / "prefix"
    root.mkdir()
    _source(source)
    torch.save({"metadata": {}}, query)
    evidence = root / "v1.npz"
    evidence.write_bytes(b"causal evidence")
    registry = {
        "schema": COMPLETION_QUERY_PREFIX_SCHEMA,
        "query_examples_sha256": _sha256(query),
        "source_examples_sha256": _sha256(source),
        "split": "test",
        "causal": True,
        "parameters_fitted": False,
        "annotations_used_to_compute_evidence": False,
        "test_split_accessed": True,
        "videos": {
            "v1": {"file": evidence.name, "sha256": _sha256(evidence)}
        },
    }
    (root / "registry.json").write_text(json.dumps(registry))
    loaded = load_completion_query_prefix_registry(
        root,
        query_examples=query,
        source_examples=source,
        split="test",
    )
    assert loaded["annotations_used_to_compute_evidence"] is False
    evidence.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after compilation"):
        load_completion_query_prefix_registry(
            root,
            query_examples=query,
            source_examples=source,
            split="test",
        )


def test_inference_view_is_invariant_to_query_gt() -> None:
    allowed = {
        "video_id": "v1",
        "task_index": 0,
        "decoded_index": torch.tensor([3, 8], dtype=torch.long),
        "stream_start_sec": 0.0,
        "statistic": torch.randn(2, 5),
        "factor_verb": torch.softmax(torch.randn(2, 3), dim=-1),
        "factor_noun": torch.softmax(torch.randn(2, 4), dim=-1),
        "factor_tool": torch.softmax(torch.randn(2, 2), dim=-1),
    }
    first = {
        **allowed,
        "origin_target": torch.tensor([True, False]),
        "mode_target": torch.tensor([0, -1]),
        "native_assignment": torch.tensor([0, 1]),
        "native_event_index": torch.tensor([3, 8]),
        "native_time_sec": torch.tensor([0.4, 1.0]),
        "native_mode": torch.tensor([0, 1]),
        "native_error_types": ((), ("wrong_order",)),
    }
    perturbed = {
        **allowed,
        "origin_target": torch.tensor([False, True]),
        "mode_target": torch.tensor([2, 2]),
        "native_assignment": torch.tensor([7, -1]),
        "native_event_index": torch.tensor([1, 2, 7]),
        "native_time_sec": torch.tensor([0.1, 0.2, 0.9]),
        "native_mode": torch.tensor([1, 1, 1]),
        "native_error_types": (("changed",),) * 3,
    }

    first_view = completion_inference_view(first)
    perturbed_view = completion_inference_view(perturbed)

    assert set(first_view) == COMPLETION_INFERENCE_FIELDS
    assert not (set(first_view) & COMPLETION_EVALUATION_ONLY_FIELDS)
    assert set(perturbed_view) == set(first_view)
    for name in COMPLETION_INFERENCE_FIELDS:
        left = first_view[name]
        right = perturbed_view[name]
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        else:
            assert left == right
    with pytest.raises(TypeError):
        first_view["native_mode"] = torch.tensor([1])  # type: ignore[index]
