"""Create a label-blind internal validation split and keep the public test split untouched."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

from provia.data.io import read_jsonl, write_jsonl


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _group(video_id: str, subject_id: str | None, pattern: re.Pattern[str]) -> str:
    if subject_id:
        return f"subject:{subject_id}"
    match = pattern.search(video_id)
    if match is None:
        raise ValueError(f"cannot derive a leakage group from {video_id!r}")
    return f"video_group:{match.group(1)}"


def _is_validation(group: str, salt: str, fraction: float) -> bool:
    value = int(hashlib.sha256(f"{salt}|{group}".encode()).hexdigest(), 16)
    return value / float(1 << 256) < fraction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--salt", default="provia_nested_selection_v1")
    parser.add_argument("--video-group-regex", default=r"^([^-]+)")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not 0 < args.validation_fraction < 1:
        parser.error("validation-fraction must lie in (0,1)")
    pattern = re.compile(args.video_group_regex)
    episodes = tuple(read_jsonl(args.input))
    groups = {
        episode.episode_id: _group(
            episode.video_id, episode.subject_id, pattern
        )
        for episode in episodes
        if episode.split == "train"
    }
    assignment = {
        group: (
            "validation"
            if _is_validation(group, args.salt, args.validation_fraction)
            else "train"
        )
        for group in sorted(set(groups.values()))
    }
    result = []
    for episode in episodes:
        if episode.split == "train":
            group = groups[episode.episode_id]
            split = assignment[group]
            result.append(
                replace(
                    episode,
                    split=split,
                    attributes={
                        **episode.attributes,
                        "nested_selection_split": {
                            "schema": "label_blind_group_hash",
                            "group": group,
                            "partition": split,
                        },
                    },
                )
            )
        elif episode.split in {"val", "validation"}:
            result.append(
                replace(
                    episode,
                    split="test",
                    attributes={
                        **episode.attributes,
                        "public_test_holdout": True,
                    },
                )
            )
        else:
            result.append(episode)
    counts = Counter(episode.split for episode in result)
    if not all(counts[name] for name in ("train", "validation", "test")):
        raise ValueError("nested split must contain train, validation, and test")
    task_partitions = {
        task: Counter(
            episode.split for episode in result if episode.task_id == task
        )
        for task in sorted({episode.task_id for episode in result})
    }
    incomplete = {
        task: dict(count)
        for task, count in task_partitions.items()
        if count["train"] == 0 or count["validation"] == 0 or count["test"] == 0
    }
    if incomplete:
        raise ValueError(f"task coverage is incomplete: {incomplete}")
    report = write_jsonl(tuple(result), args.output, require_paths=True)
    audit = {
        "schema": "nested_selection_split",
        "source": str(args.input.resolve()),
        "source_sha256": _sha256(args.input),
        "output": str(args.output.resolve()),
        "validation_fraction": args.validation_fraction,
        "salt": args.salt,
        "video_group_regex": args.video_group_regex,
        "assignment_uses_labels": False,
        "existing_validation_reclassified_as_test": True,
        "partition_episodes": dict(sorted(counts.items())),
        "group_partition": dict(sorted(assignment.items())),
        "task_partition_episodes": {
            task: dict(sorted(count.items())) for task, count in task_partitions.items()
        },
        "episodes": report.episodes,
        "segments": report.segments,
    }
    audit_path = args.output.with_name(args.output.stem + ".split.json")
    with open(audit_path, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
