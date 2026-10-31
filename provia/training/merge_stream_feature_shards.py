"""Validate and merge sharded stream-feature manifests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    destination = args.root / "manifest.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    paths = sorted(args.root.glob("manifest.shard*_of_*.json"))
    if not paths:
        raise FileNotFoundError("no shard manifests found")
    values = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            values.append(json.load(handle))
    shard_counts = {int(item["num_shards"]) for item in values}
    if len(shard_counts) != 1:
        raise ValueError("shard manifests disagree on num_shards")
    count = shard_counts.pop()
    indices = {int(item["shard_index"]) for item in values}
    if len(values) != count or indices != set(range(count)):
        raise ValueError("shard set is incomplete or duplicated")
    invariant_keys = (
        "schema", "backbone", "backbone_checkpoint", "role_mode",
        "observer_checkpoint", "observer_sha256", "fold", "partitions",
        "sample_rate_hz", "episodes_total_before_sharding", "global_dim",
        "roles", "role_interpretation", "role_dim", "role_change_dim",
        "role_change_operator", "dtype", "contains_annotations",
        "test_split_accessed",
    )
    for key in invariant_keys:
        if any(item.get(key) != values[0].get(key) for item in values[1:]):
            raise ValueError(f"shard manifests disagree on {key}")
    merged = {key: values[0].get(key) for key in invariant_keys}
    merged.update(
        {
            "episodes": sum(int(item["episodes"]) for item in values),
            "frames": sum(int(item["frames"]) for item in values),
            "num_shards": count,
            "shard_manifests": [path.name for path in paths],
        }
    )
    if merged["episodes"] != merged["episodes_total_before_sharding"]:
        raise ValueError("merged episode count disagrees with the source partition")
    feature_files = list(args.root.glob("*_stream_features.npz"))
    if len(feature_files) != merged["episodes"]:
        raise ValueError(
            f"found {len(feature_files)} feature files for {merged['episodes']} episodes"
        )
    temporary = destination.with_name(destination.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    print(json.dumps(merged, sort_keys=True))


if __name__ == "__main__":
    main()
