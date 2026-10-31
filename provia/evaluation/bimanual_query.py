"""Validation of query-only artifacts for frozen bimanual evaluation."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import torch




def _examples_binding_hashes(examples):
    """sha256 of the examples file and, for gap-candidate examples, of the base examples whose registries they inherit."""
    import torch as _torch
    hashes = {_sha256(examples)}
    try:
        meta = _torch.load(examples, map_location="cpu", weights_only=False).get("metadata", {})
        base = meta.get("gap_base_examples_sha256") if isinstance(meta, dict) else None
        if base: hashes.add(str(base))
    except Exception:
        pass
    return hashes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


ROLE_QUERY_SCHEMA = "role_query_examples"
ROLE_QUERY_PREFIX_SCHEMA = "role_query_prefix_evidence"

ROLE_INFERENCE_FIELDS = frozenset(
    {
        "video_id",
        "start_index",
        "onset_index",
        "end_index",
        "role_index",
        "statistic",
        "factor_verb",
        "factor_noun",
        "factor_tool",
    }
)
ROLE_EVALUATION_ONLY_FIELDS = frozenset(
    {
        "origin_target",
        "mode_target",
        "native_assignment",
        "native_error_types",
        "native_intervals",
    }
)


def role_inference_view(
    sequence: Mapping[str, object],
) -> Mapping[str, object]:
    """Return the immutable annotation-free input to bimanual scoring."""

    missing = ROLE_INFERENCE_FIELDS - set(sequence)
    if missing:
        raise ValueError(
            f"role query sequence lacks inference fields: {sorted(missing)}"
        )
    view = {name: sequence[name] for name in sorted(ROLE_INFERENCE_FIELDS)}
    if set(view) & ROLE_EVALUATION_ONLY_FIELDS:
        raise RuntimeError("evaluation-only fields entered the role inference view")
    return MappingProxyType(view)


def load_role_query_artifact(
    path: Path,
    *,
    source_examples: Path,
    split: str,
) -> tuple[Mapping[str, object], Sequence[Mapping[str, object]]]:
    """Load predicted candidates and evaluation labels after model selection."""

    artifact = torch.load(path, map_location="cpu", weights_only=False)
    metadata = artifact.get("metadata") if isinstance(artifact, Mapping) else None
    sequences = artifact.get(split) if isinstance(artifact, Mapping) else None
    expected_test_access = split == "test"
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("schema") != ROLE_QUERY_SCHEMA
        or metadata.get("source_examples_sha256") not in _examples_binding_hashes(source_examples)
        or metadata.get("split") != split
        or metadata.get("parameters_fitted") is not False
        or metadata.get("query_annotations_used_by_candidate_decoder") is not False
        or metadata.get("query_annotations_used_for_evaluation_only") is not True
        or metadata.get("test_split_accessed") is not expected_test_access
        or not isinstance(sequences, Sequence)
        or not sequences
    ):
        raise ValueError("role query artifact is malformed or incompatible")
    return metadata, sequences  # type: ignore[return-value]


def load_role_query_prefix_registry(
    root: Path,
    *,
    query_examples: Path,
    source_examples: Path,
    split: str,
) -> Mapping[str, object]:
    """Load annotation-free role evidence compiled for a frozen query split."""

    path = root / "registry.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    registry = json.loads(path.read_text())
    expected_test_access = split == "test"
    if (
        not isinstance(registry, Mapping)
        or registry.get("schema") != ROLE_QUERY_PREFIX_SCHEMA
        or registry.get("query_examples_sha256") != _sha256(query_examples)
        or registry.get("source_examples_sha256") not in _examples_binding_hashes(source_examples)
        or registry.get("split") != split
        or registry.get("causal") is not True
        or registry.get("parameters_fitted") is not False
        or registry.get("annotations_used_to_compute_evidence") is not False
        or registry.get("test_split_accessed") is not expected_test_access
        or not isinstance(registry.get("videos"), Mapping)
    ):
        raise ValueError("role query prefix-evidence registry is incompatible")
    for video_id, raw in registry["videos"].items():  # type: ignore[union-attr]
        if not isinstance(raw, Mapping):
            raise ValueError(f"{video_id}: malformed role query evidence record")
        file = root / str(raw.get("file"))
        if not file.is_file() or _sha256(file) != raw.get("sha256"):
            raise ValueError(f"{video_id}: role query evidence changed after compilation")
    return registry
