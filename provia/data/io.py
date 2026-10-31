"""Safe JSONL I/O and corpus-level validation."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from .schema import CanonicalEpisode, SchemaError


@dataclass
class CorpusReport:
    episodes: int = 0
    segments: int = 0
    observed_segments: int = 0
    untimed_segments: int = 0
    errors: int = 0
    corrections: int = 0
    interventions: int = 0
    decision_windows: int = 0
    datasets: Counter[str] = field(default_factory=Counter)
    splits: Counter[str] = field(default_factory=Counter)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "episodes": self.episodes,
            "segments": self.segments,
            "observed_segments": self.observed_segments,
            "untimed_segments": self.untimed_segments,
            "errors": self.errors,
            "corrections": self.corrections,
            "interventions": self.interventions,
            "decision_windows": self.decision_windows,
            "datasets": dict(self.datasets),
            "splits": dict(self.splits),
            "issues": list(self.issues),
        }


def read_jsonl(path: str | os.PathLike[str]) -> Iterator[CanonicalEpisode]:
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield CanonicalEpisode.from_dict(json.loads(line))
            except Exception as exc:
                raise SchemaError(f"{path}:{line_number}: {exc}") from exc


def validate_corpus(
    episodes: Iterable[CanonicalEpisode],
    *,
    require_paths: bool = False,
    enforce_subject_disjoint: bool = False,
) -> CorpusReport:
    report = CorpusReport()
    seen_ids: set[str] = set()
    subject_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for episode in episodes:
        report.episodes += 1
        report.datasets[episode.dataset] += 1
        report.splits[episode.split] += 1
        if episode.episode_id in seen_ids:
            report.issues.append(f"duplicate episode_id {episode.episode_id}")
        seen_ids.add(episode.episode_id)
        report.issues.extend(
            f"{episode.episode_id}: {issue}" for issue in episode.validate()
        )
        if require_paths:
            if not episode.video_path:
                report.issues.append(f"{episode.episode_id}: video_path is absent")
            elif not os.path.exists(episode.video_path):
                report.issues.append(
                    f"{episode.episode_id}: video_path does not exist: {episode.video_path}"
                )
            for source in episode.source_files:
                if not os.path.exists(source):
                    report.issues.append(
                        f"{episode.episode_id}: source file does not exist: {source}"
                    )
        if episode.subject_id:
            subject_splits[(episode.dataset, episode.subject_id)].add(episode.split)
        report.segments += len(episode.segments)
        report.observed_segments += sum(item.observed for item in episode.segments)
        report.untimed_segments += sum(
            item.start_sec is None for item in episode.segments
        )
        report.errors += sum(item.correctness == "error" for item in episode.segments)
        report.corrections += sum(
            item.correctness == "correction" for item in episode.segments
        )
        report.interventions += len(episode.interventions)
        report.decision_windows += len(episode.decision_windows)
    if enforce_subject_disjoint:
        for (dataset, subject), splits in sorted(subject_splits.items()):
            if len(splits) > 1:
                report.issues.append(
                    f"subject leakage: {dataset}:{subject} appears in {sorted(splits)}"
                )
    return report


def write_jsonl(
    episodes: Iterable[CanonicalEpisode],
    path: str | os.PathLike[str],
    *,
    require_paths: bool = False,
    enforce_subject_disjoint: bool = False,
) -> CorpusReport:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(episodes)
    report = validate_corpus(
        materialized,
        require_paths=require_paths,
        enforce_subject_disjoint=enforce_subject_disjoint,
    )
    if report.issues:
        raise SchemaError("Corpus validation failed:\n- " + "\n- ".join(report.issues))
    temporary = destination.with_name(destination.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        for episode in materialized:
            json.dump(episode.to_dict(), handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    summary = destination.with_name(destination.name + ".summary.json")
    summary_tmp = summary.with_name(summary.name + ".tmp")
    with open(summary_tmp, "w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(summary_tmp, summary)
    return report
