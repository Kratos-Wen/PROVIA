"""Validation of query-only artifacts evaluated by a frozen trained model."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import torch

from provia.training.common import _sha256


COMPLETION_QUERY_SCHEMA = "completion_query_examples"
COMPLETION_QUERY_PREFIX_SCHEMA = "completion_query_prefix_evidence"

# This is the complete query-time interface of the single-stream reasoner.
# In particular, no annotation, native-event assignment, correctness mode, or
# error category is admitted.  Evaluation code must construct this view before
# invoking any model-scoring operation.
COMPLETION_INFERENCE_FIELDS = frozenset(
    {
        "video_id",
        "task_index",
        "decoded_index",
        "stream_start_sec",
        "statistic",
        "factor_verb",
        "factor_noun",
        "factor_tool",
    }
)
COMPLETION_EVALUATION_ONLY_FIELDS = frozenset(
    {
        "origin_target",
        "mode_target",
        "native_assignment",
        "native_event_index",
        "native_time_sec",
        "native_mode",
        "native_error_types",
    }
)



def _examples_binding_hashes(examples):
    """sha256 of an examples file and, for derived examples (identity fusion, gap candidates), of the base examples
    whose registries they inherit (recorded as gap_base_examples_sha256 by the deriving tool)."""
    hashes = {_sha256(examples)}
    try:
        import torch as _torch
        meta = _torch.load(examples, map_location="cpu", weights_only=False).get("metadata", {})
        base = meta.get("gap_base_examples_sha256") if isinstance(meta, dict) else None
        if base: hashes.add(str(base))
    except Exception:
        pass
    return hashes


def completion_inference_view(
    sequence: Mapping[str, object],
) -> Mapping[str, object]:
    """Return an immutable, annotation-free query view for model scoring.

    The source artifact intentionally co-locates predictions and labels for
    reproducible metric computation.  Passing the source mapping to a model
    would nevertheless leave an unnecessary path from query annotations to
    inference.  This allowlist is the runtime firewall between scoring and
    post-hoc evaluation.
    """

    missing = COMPLETION_INFERENCE_FIELDS - set(sequence)
    if missing:
        raise ValueError(
            f"completion query sequence lacks inference fields: {sorted(missing)}"
        )
    view = {
        name: sequence[name]
        for name in sorted(COMPLETION_INFERENCE_FIELDS)
    }
    if set(view) & COMPLETION_EVALUATION_ONLY_FIELDS:
        raise RuntimeError("evaluation-only fields entered the inference view")
    return MappingProxyType(view)


def load_completion_query_artifact(
    path: Path,
    *,
    source_examples: Path,
    split: str,
) -> tuple[Mapping[str, object], Sequence[Mapping[str, object]]]:
    """Load labels and support only after model selection is frozen."""

    artifact = torch.load(path, map_location="cpu", weights_only=False)
    metadata = artifact.get("metadata") if isinstance(artifact, Mapping) else None
    sequences = artifact.get(split) if isinstance(artifact, Mapping) else None
    expected_test_access = split == "test"
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("schema") != COMPLETION_QUERY_SCHEMA
        or metadata.get("source_examples_sha256") not in _examples_binding_hashes(source_examples)
        or metadata.get("split") != split
        or metadata.get("parameters_fitted") is not False
        or metadata.get("query_annotations_used_by_candidate_decoder") is not False
        or metadata.get("query_annotations_used_for_evaluation_only") is not True
        or metadata.get("test_split_accessed") is not expected_test_access
        or not isinstance(sequences, Sequence)
        or not sequences
    ):
        raise ValueError("completion query artifact is malformed or incompatible")
    return metadata, sequences  # type: ignore[return-value]


def load_completion_query_prefix_registry(
    root: Path,
    *,
    query_examples: Path,
    source_examples: Path,
    split: str,
) -> Mapping[str, object]:
    """Load annotation-free prefix evidence compiled for a frozen query split."""

    path = root / "registry.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    registry = json.loads(path.read_text())
    expected_test_access = split == "test"
    if (
        not isinstance(registry, Mapping)
        or registry.get("schema") != COMPLETION_QUERY_PREFIX_SCHEMA
        or registry.get("query_examples_sha256") not in _examples_binding_hashes(query_examples)
        or registry.get("source_examples_sha256") not in _examples_binding_hashes(source_examples)
        or registry.get("split") != split
        or registry.get("causal") is not True
        or registry.get("parameters_fitted") is not False
        or registry.get("annotations_used_to_compute_evidence") is not False
        or registry.get("test_split_accessed") is not expected_test_access
        or not isinstance(registry.get("videos"), Mapping)
    ):
        raise ValueError("query prefix-evidence registry is malformed or incompatible")
    for video_id, raw in registry["videos"].items():  # type: ignore[union-attr]
        if not isinstance(raw, Mapping):
            raise ValueError(f"{video_id}: query prefix-evidence record is malformed")
        file = root / str(raw.get("file"))
        if not file.is_file() or _sha256(file) != raw.get("sha256"):
            raise ValueError(f"{video_id}: query prefix evidence changed after compilation")
    return registry
