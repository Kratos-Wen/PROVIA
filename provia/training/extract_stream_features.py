"""Cache causal global and competitive-role features for development streams."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from provia.data.folds import materialize_cross_validation_fold
from provia.data.io import read_jsonl
from provia.data.schema import CanonicalEpisode

from .backbone import _load_backbone, _preprocess


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@torch.inference_mode()
def _extract_episode(
    episode: CanonicalEpisode,
    backbone,
    device: torch.device,
    *,
    sample_rate_hz: float,
    batch_size: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    try:
        import decord
    except ImportError as exc:
        raise RuntimeError("decord is required for stream extraction") from exc
    if not episode.video_path or not Path(episode.video_path).is_file():
        raise FileNotFoundError(episode.video_path or episode.episode_id)
    reader = decord.VideoReader(
        str(episode.video_path), ctx=decord.cpu(0), num_threads=2
    )
    native_fps = float(reader.get_avg_fps())
    if not np.isfinite(native_fps) or native_fps <= 0:
        raise ValueError(f"{episode.episode_id}: invalid native fps")
    duration = len(reader) / native_fps
    timestamps = np.arange(
        0.0, duration, 1.0 / sample_rate_hz, dtype=np.float64
    )
    frame_ids = np.unique(
        np.clip(np.rint(timestamps * native_fps).astype(np.int64), 0, len(reader) - 1)
    )
    timestamps = frame_ids.astype(np.float64) / native_fps
    global_values: list[np.ndarray] = []
    role_values: list[np.ndarray] = []
    role_change_values: list[np.ndarray] = []
    previous_global: torch.Tensor | None = None
    for start in range(0, len(frame_ids), batch_size):
        indices = frame_ids[start : start + batch_size]
        decoded = reader.get_batch(indices.tolist()).asnumpy()
        frames = _preprocess(decoded, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = backbone.forward_features(frames)
            global_features = output["x_norm_clstoken"]
            current = global_features.float()
            preceding = torch.cat(
                (
                    current[:1]
                    if previous_global is None
                    else previous_global.unsqueeze(0),
                    current[:-1],
                ),
                dim=0,
            )
            change = current - preceding
            role_features = torch.zeros(
                len(current), 2, current.shape[-1],
                dtype=current.dtype, device=current.device,
            )
            role_features[:, 0] = current
            role_change_features = torch.zeros_like(role_features)
            role_change_features[:, 0] = change
            previous_global = current[-1].detach()
        global_values.append(
            global_features.float().cpu().numpy().astype(np.float16)
        )
        role_values.append(role_features.float().cpu().numpy().astype(np.float16))
        role_change_values.append(role_change_features.cpu().numpy())
    return (
        frame_ids,
        timestamps,
        np.concatenate(global_values),
        np.concatenate(role_values),
        np.concatenate(role_change_values),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--partition", action="append", default=[])
    parser.add_argument("--sample-rate-hz", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backbone",
        choices=("dinov2_vits14",),
        default="dinov2_vits14",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path.home()
        / ".cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth",
    )
    args = parser.parse_args()
    if args.sample_rate_hz <= 0 or args.batch_size <= 0:
        parser.error("sample rate and batch size must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must lie in [0,num-shards)")
    partitions = set(args.partition or ("train", "validation"))
    if "test" in partitions:
        if partitions != {"test"}:
            parser.error("test extraction requires only --partition test")
    elif not partitions <= {"train", "validation"}:
        parser.error("partitions must be train/validation or test only")
    device = torch.device(args.device)
    materialized = materialize_cross_validation_fold(
        tuple(read_jsonl(args.manifest)), args.fold
    )
    train = tuple(item for item in materialized if item.split == "train")
    all_episodes = tuple(item for item in materialized if item.split in partitions)
    episodes = tuple(
        item
        for index, item in enumerate(all_episodes)
        if index % args.num_shards == args.shard_index
    )
    if not train or not episodes:
        raise ValueError("development partition is empty")
    backbone = _load_backbone(
        args.repo, args.checkpoint, device, backbone=args.backbone
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    total_frames = 0
    for index, episode in enumerate(episodes, start=1):
        output = args.output_root / f"{episode.video_id}_stream_features.npz"
        if output.is_file():
            with np.load(output, allow_pickle=False) as payload:
                if set(payload.files) != {
                    "frame_ids",
                    "timestamps_sec",
                    "global_features",
                    "role_features",
                    "role_change_features",
                }:
                    raise ValueError(f"{output}: malformed stream cache")
                total_frames += len(payload["frame_ids"])
            continue
        (
            frame_ids,
            timestamps,
            global_features,
            role_features,
            role_change_features,
        ) = _extract_episode(
            episode,
            backbone,
            device,
            sample_rate_hz=args.sample_rate_hz,
            batch_size=args.batch_size,
        )
        temporary = output.with_name(output.stem + ".tmp.npz")
        np.savez_compressed(
            temporary,
            frame_ids=frame_ids,
            timestamps_sec=timestamps,
            global_features=global_features,
            role_features=role_features,
            role_change_features=role_change_features,
        )
        os.replace(temporary, output)
        total_frames += len(frame_ids)
        print(
            json.dumps(
                {
                    "episodes_complete": index,
                    "episodes_total": len(episodes),
                    "frames_complete": total_frames,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    manifest_name = (
        "manifest.json"
        if args.num_shards == 1
        else f"manifest.shard{args.shard_index:03d}_of_{args.num_shards:03d}.json"
    )
    _atomic_json(
        args.output_root / manifest_name,
        {
            "schema": "role_stream_features",
            "backbone": args.backbone,
            "backbone_checkpoint": str(args.checkpoint),
            "role_mode": "single_actor",
            "fold": args.fold,
            "partitions": sorted(partitions),
            "sample_rate_hz": args.sample_rate_hz,
            "episodes": len(episodes),
            "episodes_total_before_sharding": len(all_episodes),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "frames": total_frames,
            "global_dim": 384,
            "roles": ["left", "right"],
            "role_interpretation": {
                "left": "single observed actor",
                "right": "structural null role",
            },
            "role_dim": 384,
            "role_change_dim": 384,
            "role_change_operator": (
                "causal global descriptor difference in active actor; zero in null role"
            ),
            "dtype": {
                "global_features": "float16",
                "role_features": "float16",
                "role_change_features": "float32",
            },
            "contains_annotations": False,
            "test_split_accessed": "test" in partitions,
        },
    )


if __name__ == "__main__":
    main()
