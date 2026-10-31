"""Attach fixed-receptive-field causal visual windows to event examples.

The 127-frame support is the analytic receptive field of six kernel-three
causal blocks with dilations 1--32.  It is an architecture property shared by
all datasets, not a duration or validation-selected window.  Every window ends
at the candidate's causal checkpoint and only train/validation streams are
loaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor

from provia.automaton.example_io import (
    _atomic_torch,
    _episodes_and_intents,
    _load_source,
)
from provia.data.role_stream import load_competitive_role_streams
from provia.training.common import _sha256


CAUSAL_WINDOW_LENGTH = 127


def _indices(end_index: Tensor, length: int) -> Tensor:
    if end_index.ndim != 1 or end_index.dtype != torch.long:
        raise ValueError("candidate indices must be a long vector")
    if torch.any(end_index < 0) or length <= 0:
        raise ValueError("candidate indices and stream length are malformed")
    offsets = torch.arange(
        -CAUSAL_WINDOW_LENGTH + 1, 1, dtype=torch.long
    )
    result = (end_index[:, None].cpu() + offsets[None, :]).clamp_min(0)
    if torch.any(result >= length):
        raise ValueError("a candidate checkpoint lies outside its stream")
    return result


def _attach_role(examples: Path, output: Path) -> dict[str, object]:
    artifact, metadata = _load_source(examples)
    if metadata.get("schema") != "role_examples":
        raise ValueError("causal windows require role-onset examples")
    train_episode, validation_episode, intent = _episodes_and_intents(metadata)
    feature_root = Path(str(metadata["feature_root"]))
    streams = {
        "train": load_competitive_role_streams(
            train_episode, feature_root, intent
        ),
        "validation": load_competitive_role_streams(
            validation_episode, feature_root, intent
        ),
    }
    window_dim = None
    for split, split_streams in streams.items():
        lookup = {stream.video_id: stream for stream in split_streams}
        for sequence in artifact[split]:
            if not isinstance(sequence, dict):
                raise ValueError("role sequence is not mutable")
            stream = lookup.get(str(sequence["video_id"]))
            if stream is None:
                raise ValueError("role stream is unavailable")
            onset = sequence["onset_index"]
            role = sequence["role_index"]
            if (
                not isinstance(role, Tensor)
                or role.dtype != torch.long
                or role.shape != onset.shape
                or torch.any((role < 0) | (role >= stream.role_features.shape[1]))
            ):
                raise ValueError("candidate roles are malformed")
            onset_index = _indices(onset, len(stream.global_features))
            end_index = _indices(
                sequence["end_index"], len(stream.global_features)
            )
            global_feature = stream.global_features.to(torch.float16)
            global_window = global_feature[onset_index]
            global_end_window = global_feature[end_index]
            role_feature = stream.role_features.to(torch.float16)
            role_window = role_feature[onset_index, role.cpu()[:, None]]
            role_end_window = role_feature[end_index, role.cpu()[:, None]]
            window = torch.cat((global_window, role_window), dim=-1)
            end_window = torch.cat(
                (global_end_window, role_end_window), dim=-1
            )
            sequence["causal_window"] = window
            sequence["causal_window_end"] = end_window
            window_dim = int(window.shape[-1])
    if window_dim is None:
        raise ValueError("role examples contain no causal windows")
    artifact["metadata"] = {
        **dict(metadata),
        "schema": "role_windowed_examples",
        "source_examples": str(examples.resolve()),
        "source_examples_sha256": _sha256(examples),
        "causal_window_length": CAUSAL_WINDOW_LENGTH,
        "causal_window_dim": window_dim,
        "causal_window_checkpoints": [
            "predicted interaction onset",
            "predicted event end",
        ],
        "causal_window_future_frames": False,
        "causal_window_storage_dtype": "float16",
        "test_split_accessed": False,
    }
    _atomic_torch(output, artifact)
    return artifact["metadata"]  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    metadata = _attach_role(args.examples, args.output)
    print(
        {
            "output": str(args.output),
            "schema": metadata["schema"],
            "causal_window_length": metadata["causal_window_length"],
            "causal_window_dim": metadata["causal_window_dim"],
            "test_split_accessed": metadata["test_split_accessed"],
        }
    )


if __name__ == "__main__":
    main()
