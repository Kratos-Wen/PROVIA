"""Alarm-level evaluation for untrimmed online mistake detection.

Input: a per-video candidate dump (``--candidate-score-output`` of the frozen evaluators): for each video the
decision times of the causal decoder, the anytime mistake score at each decision, the native (annotated) step each
decision is assigned to, and the native step statuses (0 correct, 1 mistake).  An alarm rule turns the score stream
into alarm times.  Nothing after the first mistake is discarded.

Metrics (all causal, no ground-truth boundaries used by the rule):
  * false alarms per minute of correct operation (alarms on correct steps / minutes of correct executions) and per
    minute of all video;
  * recall of all mistakes, of first mistakes, and of mistakes after the first, a mistake counting as detected when an
    alarm falls on a decision assigned to it or within ``grace`` later decisions;
  * detection delay (seconds between the mistake's annotated end and the alarm decision) and the share of detections
    issued before the following step's decision;
  * fraction of correct executions with zero alarms; false alarms per minute before/after the first mistake.
Two rules: ``instantaneous`` (threshold on the score) and ``sequential`` (Shiryaev-Roberts statistic on the
posterior-to-prior odds, alarm-reset).  A table at matched budgets (0.1/0.5/1/2 false alarms per minute of correct
operation) is produced by sweeping the threshold; a fixed threshold can also be supplied (``--threshold``) for a
calibrated, one-shot evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ERROR = 1


def evidence(scores: np.ndarray, prior_odds: float) -> np.ndarray:
    s = np.clip(np.asarray(scores, dtype=np.float64), 1e-6, 1 - 1e-6)
    return (s / (1 - s)) / prior_odds


def alarms_for(statistic_input: np.ndarray, threshold: float, rule: str) -> list[int]:
    if rule == "instantaneous":
        return [i for i, x in enumerate(statistic_input) if x >= threshold]
    out, m = [], 0.0
    for i, x in enumerate(statistic_input):
        m = (1.0 + m) * x
        if m >= threshold:
            out.append(i)
            m = 0.0
    return out


def video_stats(v: dict, alarms: list[int], grace: int) -> dict:
    assignment = np.asarray(v["assignment"], dtype=int)
    native_mode = np.asarray(v["native_mode"], dtype=int)
    native_time = np.asarray(v["native_time"], dtype=float)
    decoded = np.asarray(v["decoded_time"], dtype=float)
    mistakes = [j for j in range(len(native_mode)) if native_mode[j] == ERROR]
    first_time = native_time[mistakes[0]] if mistakes else np.inf
    duration = float(decoded[-1] - decoded[0]) if len(decoded) > 1 else 0.0
    # correct-operation time: whole video if it has no mistake, otherwise time not attributed to mistake steps
    step_time = np.diff(np.concatenate(([decoded[0]], native_time))) if len(native_time) else np.zeros(0)
    mistake_time = float(step_time[mistakes].clip(min=0).sum()) if mistakes and len(step_time) == len(native_time) else 0.0
    correct_minutes = max(duration - mistake_time, 0.0) / 60.0
    alarm_set = set(alarms)
    false_alarms = [i for i in alarms if native_mode[assignment[i]] != ERROR]
    detected, delays, before_next = {}, [], []
    for j in mistakes:
        idx = np.flatnonzero(assignment == j)
        if not len(idx):
            continue
        window = set(range(int(idx[0]), min(int(idx[-1]) + grace + 1, len(decoded))))
        hits = sorted(window & alarm_set)
        if hits:
            detected[j] = hits[0]
            delays.append(float(decoded[hits[0]] - native_time[j]))
            next_end = native_time[j + 1] if j + 1 < len(native_time) else np.inf
            before_next.append(bool(decoded[hits[0]] <= next_end))
    fa_before = sum(1 for i in false_alarms if decoded[i] < first_time)
    fa_after = len(false_alarms) - fa_before
    min_before = (min(first_time, decoded[-1]) - decoded[0]) / 60.0 if mistakes else duration / 60.0
    min_after = max(decoded[-1] - first_time, 0.0) / 60.0 if mistakes else 0.0
    covered = [j for j in mistakes if (assignment == j).any()]
    types = v.get("native_error_types") or []
    by_type = {}
    for j in mistakes:
        key = (types[j][0] if j < len(types) and len(types[j]) else "unspecified")
        n_det, n_all = by_type.get(key, (0, 0))
        by_type[key] = (n_det + int(j in detected), n_all + 1)
    return {
        "by_type": by_type,
        "has_mistake": bool(mistakes),
        "n_mistakes": len(mistakes),
        "n_covered": len(covered),
        "first_detected": bool(mistakes and mistakes[0] in detected),
        "n_detected": len(detected),
        "n_after_first": max(len(covered) - 1, 0) if covered else 0,
        "n_after_first_detected": sum(1 for j in detected if covered and j != covered[0]),
        "false_alarms": len(false_alarms),
        "zero_alarms": len(alarms) == 0,
        "duration_min": duration / 60.0,
        "correct_minutes": correct_minutes,
        "delays": delays,
        "before_next": before_next,
        "fa_before": fa_before,
        "fa_after": fa_after,
        "min_before": min_before,
        "min_after": min_after,
    }


def aggregate(stats: list[dict]) -> dict:
    correct = [s for s in stats if not s["has_mistake"]]
    mist = [s for s in stats if s["has_mistake"]]
    delays = [d for s in stats for d in s["delays"]]
    before = [b for s in stats for b in s["before_next"]]
    fa_correct = sum(s["false_alarms"] for s in correct)
    min_correct = sum(s["correct_minutes"] for s in correct)
    per_type = {}
    for s in stats:
        for key, (d, n) in s.get("by_type", {}).items():
            pd, pn = per_type.get(key, (0, 0)); per_type[key] = (pd + d, pn + n)
    return {
        "recall_by_type": {k: (d / n if n else float("nan")) for k, (d, n) in per_type.items()},
        "count_by_type": {k: n for k, (d, n) in per_type.items()},
        "fa_per_min_correct_executions": fa_correct / min_correct if min_correct else float("nan"),
        "fa_per_min_all_video": sum(s["false_alarms"] for s in stats) / max(sum(s["duration_min"] for s in stats), 1e-9),
        "fa_per_min_correct_operation": sum(s["false_alarms"] for s in stats) / max(sum(s["correct_minutes"] for s in stats), 1e-9),
        "recall_all": sum(s["n_detected"] for s in stats) / max(sum(s["n_mistakes"] for s in stats), 1),
        "recall_all_covered": sum(s["n_detected"] for s in stats) / max(sum(s["n_covered"] for s in stats), 1),
        "recall_first": sum(1 for s in mist if s["first_detected"]) / len(mist) if mist else float("nan"),
        "recall_after_first": (sum(s["n_after_first_detected"] for s in stats) / sum(s["n_after_first"] for s in stats))
        if sum(s["n_after_first"] for s in stats) else float("nan"),
        "n_after_first": sum(s["n_after_first"] for s in stats),
        "median_delay_sec": float(np.median(delays)) if delays else float("nan"),
        "p90_delay_sec": float(np.percentile(delays, 90)) if delays else float("nan"),
        "before_next_step_share": float(np.mean(before)) if before else float("nan"),
        "zero_alarm_correct_share": sum(1 for s in correct if s["zero_alarms"]) / max(len(correct), 1),
        "fa_per_min_before_first_mistake": sum(s["fa_before"] for s in mist) / max(sum(s["min_before"] for s in mist), 1e-9),
        "fa_per_min_after_first_mistake": sum(s["fa_after"] for s in mist) / max(sum(s["min_after"] for s in mist), 1e-9),
        "n_videos": len(stats),
        "n_correct_videos": len(correct),
        "n_mistakes": sum(s["n_mistakes"] for s in stats),
    }


def evaluate(videos: list[dict], prior_odds: float, rule: str, threshold: float, grace: int) -> dict:
    stats = []
    for v in videos:
        e = evidence(v["anytime_score"], prior_odds) if rule == "sequential" else np.asarray(v["anytime_score"], dtype=float)
        stats.append(video_stats(v, alarms_for(e, threshold, rule), grace))
    out = aggregate(stats)
    out.update({"rule": rule, "threshold": float(threshold)})
    return out


def sweep(videos: list[dict], prior_odds: float, rule: str, grace: int, budgets, budget_key: str) -> list[dict]:
    if rule == "sequential":
        # candidate thresholds from the running statistic values themselves
        values = []
        for v in videos:
            m = 0.0
            for x in evidence(v["anytime_score"], prior_odds):
                m = (1.0 + m) * x
                values.append(m)
    else:
        values = [float(x) for v in videos for x in v["anytime_score"]]
    grid = np.unique(np.quantile(np.asarray(values), np.linspace(0.0, 1.0, 400)))
    rows = [evaluate(videos, prior_odds, rule, float(t), grace) for t in grid]
    table = []
    for b in budgets:
        feasible = [r for r in rows if r[budget_key] <= b]
        best = max(feasible, key=lambda r: (r["recall_all"], -r["threshold"])) if feasible else None
        table.append({"budget": b, **(best or {"rule": rule, })})
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--prevalence", type=float, help="mistake prevalence used for the prior odds (default: from the dump's native steps)")
    ap.add_argument("--grace", type=int, default=1)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.5, 1.0, 2.0])
    ap.add_argument("--budget-key", default="fa_per_min_correct_operation", choices=["fa_per_min_correct_operation", "fa_per_min_correct_executions", "fa_per_min_all_video"])
    ap.add_argument("--threshold", type=float, help="evaluate one fixed (pre-calibrated) sequential threshold instead of sweeping")
    ap.add_argument("--output")
    a = ap.parse_args()
    z = np.load(a.dump, allow_pickle=True)
    videos = list(z["videos"])
    modes = np.concatenate([np.asarray(v["native_mode"]) for v in videos])
    prevalence = a.prevalence if a.prevalence is not None else float((modes == ERROR).mean())
    prior_odds = prevalence / (1 - prevalence)
    result = {"dump": a.dump, "prevalence": prevalence, "grace": a.grace, "budget_key": a.budget_key}
    if a.threshold is not None:
        result["fixed"] = evaluate(videos, prior_odds, "sequential", a.threshold, a.grace)
    else:
        result["sequential"] = sweep(videos, prior_odds, "sequential", a.grace, a.budgets, a.budget_key)
        result["instantaneous"] = sweep(videos, prior_odds, "instantaneous", a.grace, a.budgets, a.budget_key)
    keys = ("budget", "rule", a.budget_key, "recall_all", "recall_first", "recall_after_first", "median_delay_sec", "before_next_step_share", "zero_alarm_correct_share", "fa_per_min_after_first_mistake")
    for rule in ("sequential", "instantaneous"):
        for r in result.get(rule, []):
            line = " ".join(f"{k}={r[k]:.3f}" if isinstance(r.get(k), float) else f"{k}={r.get(k)}" for k in keys)
            types = r.get("recall_by_type", {})
            if len(types) > 1:
                line += " | " + " ".join(f"{k}={v:.2f}(n={r['count_by_type'][k]})" for k, v in sorted(types.items()))
            print(line)
    if "fixed" in result:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in result["fixed"].items()}))
    if a.output:
        Path(a.output).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
