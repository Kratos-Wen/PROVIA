"""Convert the public IMPACT v1.1 release (ego view) into the per-hand parsed-event schema.

The two-hand pipeline consumes ``<video_id>_parsed_annotations.json`` files
(one per recording) plus a ``splits.json`` with participant folds, through the
``impact`` adapter (``ImpactAdapter``).  This converter writes that
schema from the canonical TAS-B per-hand segment manifests of the public release:

* events   = non-null TAS-B action segments of either hand (``entity`` left/right);
* phase    = normal -> ``normal``; anomaly -> first active ATR type; recovery -> ``recovery``;
* decision = the public release carries no contact-onset annotation, so the decision
             frame (``contact_onset_frame``) is placed at a fixed fraction of the segment
             (default 0.5 = midpoint); approach = [start, onset),
             interaction = [onset, end] as the chain's phase model requires;
* splits   = official PPR splits restricted to the ego view and to product A
             (Disassembly_A / Reassembly_A), the procedures that have anomaly-free
             reference executions in their training partitions.

Nothing is synthesized: null-phase anomalies that are not attached to an action
(idle-period errors) are counted and reported, not converted.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

ANOMALY_ORDER = (
    "error_temporal",
    "error_spatial",
    "error_handling",
    "error_wrong_part",
    "error_wrong_tool",
    "error_procedural",
)
PRODUCT_A = ("Disassembly_A", "Reassembly_A")
HAND = {"left": "Left_hand", "right": "Right_hand"}


def _participant_disjoint_fold(out_ann: Path, kept: set[str], index: int) -> dict:
    """Deterministic participant-disjoint fold (3 unseen test
    participants, 2 unseen validation participants, the rest train).

    Rule, fixed before any run: participants who contribute an anomaly-free *Disassembly* execution
    stay in train (they are the only possible disassembly references); among the remaining
    participants every (3 test, 2 val) assignment is enumerated and the one whose execution and error
    shares are closest to test ~23% / val ~16% is chosen (ties: lexicographic).
    """
    import itertools

    stats: dict[str, dict] = {}
    for video_id in sorted(kept):
        record = json.loads((out_ann / f"{video_id}_parsed_annotations.json").read_text())
        entry = stats.setdefault(record["participant"], {"n": 0, "err": 0, "clean_dis": 0, "videos": []})
        entry["n"] += 1
        errors = sum(1 for e in record["events"] if e["has_anomaly"])
        entry["err"] += errors
        entry["videos"].append(video_id)
        if errors == 0 and record["procedure"].startswith("Disassembly"):
            entry["clean_dis"] += 1
    total_n = sum(e["n"] for e in stats.values())
    total_err = sum(e["err"] for e in stats.values())
    anchored = sorted(p for p, e in stats.items() if e["clean_dis"] > 0)
    free = sorted(p for p in stats if p not in anchored)
    best = None
    for test in itertools.combinations(free, 3):
        rest = [p for p in free if p not in test]
        for val in itertools.combinations(rest, 2):
            share = lambda group, key: sum(stats[p][key] for p in group) / max(1, (total_n if key == "n" else total_err))
            cost = abs(share(test, "n") - 7 / 31) + abs(share(val, "n") - 5 / 31) + abs(share(test, "err") - 7 / 31) + abs(share(val, "err") - 5 / 31)
            key = (round(cost, 9), test, val)
            if best is None or key < best:
                best = key
    _, test, val = best
    train = [p for p in stats if p not in test and p not in val]
    fold = {
        "fold": index,
        "official_split": "participant_disjoint",
        "rule": "anchored disassembly-reference participants in train; exhaustive (3 test, 2 val) search minimizing deviation of execution and error shares from 7/31 and 5/31",
        "participants": {"train": sorted(train), "val": sorted(val), "test": sorted(test)},
    }
    for partition, group in (("train", train), ("val", val), ("test", test)):
        fold[partition] = sorted(v for p in group for v in stats[p]["videos"])
    return fold


def _read_split(path: Path) -> list[str]:
    ids = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.endswith("_ego"):
            ids.append(line)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True, help="IMPACT-v1.1/annotations root")
    parser.add_argument("--video-dir", type=Path, required=True, help="directory with <id>_ego.mp4")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["1", "2"], help="official PPR splits to expose as folds")
    parser.add_argument("--procedures", nargs="+", default=list(PRODUCT_A))
    parser.add_argument("--onset-fraction", type=float, default=0.5, help="decision frame position within the segment")
    parser.add_argument(
        "--participant-disjoint",
        action="store_true",
        help="prepend a deterministic participant-disjoint fold (3 unseen test, 2 unseen val participants)",
    )
    args = parser.parse_args()

    tasb = args.release / "TAS-B" / "ego"
    out_ann = args.out / "parsed_annotations"
    out_ann.mkdir(parents=True, exist_ok=True)
    report = {
        "source": "IMPACT v1.1 public release (KratosWen/IMPACT), ego view",
        "license": "CC BY-NC-SA 4.0",
        "decision_frame": f"start + {args.onset_fraction} * (end - start) (no contact-onset annotation in the public release)",
        "procedures": list(args.procedures),
        "videos": 0,
        "events": 0,
        "anomalies": collections.Counter(),
        "recoveries": 0,
        "multi_type_anomalies": 0,
        "excluded_null_phase_anomalies": 0,
        "excluded_videos_other_product": 0,
        "missing_videos": [],
    }
    kept: set[str] = set()
    for manifest_path in sorted(tasb.glob("*_ego.json")):
        payload = json.loads(manifest_path.read_text())
        video_id = str(payload["video_id"]) + "_ego" if not str(payload["video_id"]).endswith("_ego") else str(payload["video_id"])
        execution = video_id[: -len("_ego")]
        procedure = "_".join(execution.split("_")[1:3])
        if procedure not in args.procedures:
            report["excluded_videos_other_product"] += 1
            continue
        meta = payload["meta_data"]
        fps = float(meta["fps"])
        frame_count = int(meta["num_frames"])
        verbs = {int(v["id"]): str(v["name"]) for v in payload["verbs"]}
        nouns = {int(n["id"]): str(n["name"]) for n in payload["nouns"]}
        actions = {int(a["id"]): str(a["name"]) for a in payload["action_labels"]}
        type_names = [str(t["name"]) for t in payload["anomaly_types"]]
        events = []
        counts = collections.Counter()
        for segment in sorted(payload["segments"], key=lambda s: (int(s["start_frame"]), s["entity"])):
            phase = str(segment["phase"]).lower()
            active = [type_names[i] for i, flag in enumerate(segment["anomaly_type"]) if flag]
            if int(segment["action_label"]) == 0:
                if phase == "anomaly":
                    report["excluded_null_phase_anomalies"] += 1
                continue
            if phase == "anomaly":
                ordered = [t for t in ANOMALY_ORDER if t in active] or active or ["error_procedural"]
                label = ordered[0]
                has_anomaly = True
                report["anomalies"][label] += 1
                report["multi_type_anomalies"] += int(len(active) > 1)
            elif phase == "recovery":
                label, has_anomaly = "recovery", False
                report["recoveries"] += 1
            else:
                label, has_anomaly = "normal", False
            counts[label] += 1
            verb = verbs.get(int(segment["verb"]), "") if int(segment["verb"]) >= 0 else ""
            noun = nouns.get(int(segment["noun"]), "") if int(segment["noun"]) >= 0 else ""
            start = int(segment["start_frame"])
            end = int(segment["end_frame"])
            if end < start or end >= frame_count:
                raise ValueError(f"{video_id}: segment {start}-{end} outside {frame_count} frames")
            onset = start + int(round(args.onset_fraction * (end - start)))
            events.append(
                {
                    "event_id": len(events) + 1,
                    "hand": HAND[str(segment["entity"]).lower()],
                    "start_frame": start,
                    "contact_onset_frame": onset,
                    "end_frame": end,
                    "verb": verb,
                    "noun_object_id": int(segment["noun"]),
                    "target_object_id": int(segment["noun"]),
                    "tool_object_id": None,
                    "anomaly_label": label,
                    "anomaly_types": active,
                    "has_anomaly": has_anomaly,
                    "noun_object_name": noun,
                    "target_object_name": noun,
                    "tool_object_name": "",
                    "action_name": actions[int(segment["action_label"])],
                    "action_label": int(segment["action_label"]),
                    "phase": phase,
                }
            )
        video_path = args.video_dir / f"{video_id}.mp4"
        if not video_path.is_file():
            report["missing_videos"].append(video_id)
        record = {
            "video_id": video_id,
            "video_path": str(video_path),
            "fps": fps,
            "frame_count": frame_count,
            "procedure": procedure,
            "participant": execution.split("_")[0],
            "object_library": {str(k): v for k, v in nouns.items()},
            "verb_library": {str(k): v for k, v in verbs.items()},
            "events": events,
            "anomaly_label_counts": dict(counts),
        }
        (out_ann / f"{video_id}_parsed_annotations.json").write_text(json.dumps(record, indent=2))
        kept.add(video_id)
        report["videos"] += 1
        report["events"] += len(events)

    folds = []
    summary = []
    if args.participant_disjoint:
        folds.append(_participant_disjoint_fold(out_ann, kept, len(folds)))
        summary.append({k: (len(v) if isinstance(v, list) else v) for k, v in folds[-1].items()})
    for split in args.splits:
        index = len(folds)
        fold = {"fold": index, "official_split": f"S{split}"}
        for partition in ("train", "val", "test"):
            ids = [v for v in _read_split(args.release / "PPR" / "splits_PPR_L" / f"{partition}.split{split}.bundle") if v in kept]
            fold[partition] = sorted(ids)
        folds.append(fold)
        summary.append({k: (len(v) if isinstance(v, list) else v) for k, v in fold.items()})
    splits = {
        "schema_version": 1,
        "fold_count": len(folds),
        "participant_disjoint": bool(args.participant_disjoint),
        "source": "official IMPACT v1.1 PPR splits (ego view, product A)",
        "anomaly_types": list(ANOMALY_ORDER),
        "folds": folds,
        "summary": summary,
    }
    (args.out / "splits.json").write_text(json.dumps(splits, indent=2))
    report["anomalies"] = dict(report["anomalies"])
    report["folds"] = summary
    (args.out / "preparation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
