"""Shared, fail-fast utilities for native dataset adapters."""

from __future__ import annotations

import json
from abc import ABC
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..schema import clean_text


class AdapterError(ValueError):
    """A native annotation does not match the documented dataset schema."""


def load_json(path: str | Path) -> Any:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        return json.loads(source.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise AdapterError(f"Malformed JSON {source}: {exc}") from exc


def require(config: Mapping[str, Any], key: str) -> Any:
    value = config.get(key)
    if value is None or value == "":
        raise AdapterError(f"Required adapter configuration key is absent: {key}")
    return value


def build_split_index(payload: Any) -> dict[str, str]:
    """Convert official ``{split: [video IDs]}`` JSON to an ID lookup."""

    if not isinstance(payload, Mapping):
        raise AdapterError("Split file must contain a JSON object")
    index: dict[str, str] = {}
    for split, raw_ids in payload.items():
        if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
            raise AdapterError(f"Split {split!r} is not a list of IDs")
        normalized = "val" if str(split).lower() in {"validation", "valid"} else str(split)
        normalized = "train" if normalized.lower() == "training" else normalized.lower()
        for raw_id in raw_ids:
            video_id = str(raw_id).strip()
            previous = index.get(video_id)
            if previous is not None and previous != normalized:
                raise AdapterError(
                    f"Video {video_id!r} appears in both {previous!r} and {normalized!r}"
                )
            index[video_id] = normalized
    return index


def load_text_split_files(split_files: Mapping[str, str]) -> dict[str, str]:
    index: dict[str, str] = {}
    for split, path in split_files.items():
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        normalized = "val" if str(split).lower() in {"validation", "valid"} else str(split)
        normalized = "train" if normalized.lower() == "training" else normalized.lower()
        for line in source.read_text(encoding="utf-8-sig").splitlines():
            video_id = line.strip()
            if not video_id:
                continue
            previous = index.get(video_id)
            if previous is not None and previous != normalized:
                raise AdapterError(
                    f"Video {video_id!r} appears in both {previous!r} and {normalized!r}"
                )
            index[video_id] = normalized
    return index


def resolve_video_path(
    video_root: Optional[str | Path],
    patterns: Sequence[str],
    **values: object,
) -> Optional[str]:
    if not video_root:
        return None
    root = Path(video_root)
    formatted = [root / pattern.format(**values) for pattern in patterns]
    for candidate in formatted:
        if candidate.exists():
            return str(candidate.resolve())
    return str(formatted[0].resolve()) if formatted else str(root.resolve())


def read_config(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise AdapterError("Adapter configuration must be a JSON object")
    return payload


def parse_action_text(value: object) -> tuple[Optional[str], Optional[str]]:
    """Split common ``verb-description`` strings without discarding raw text."""

    text = clean_text(value)
    if text is None:
        return None, None
    head, separator, _ = text.partition("-")
    verb = clean_text(head) if separator else None
    return text, verb.lower() if verb else None


class DatasetAdapter(ABC):
    dataset_name: str

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
