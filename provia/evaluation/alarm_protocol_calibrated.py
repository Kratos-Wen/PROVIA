"""Validation-calibrated alarm-level evaluation for untrimmed online mistake detection.

The threshold of the alarm rule is selected on the validation dump only, one per false-alarm budget, and the
test dump is scored once at that fixed threshold.  The test false-alarm rate is reported as achieved, not as
targeted.  Two rules are evaluated (``sequential``: Shiryaev--Roberts statistic on the likelihood ratio of the
candidate's mistake probability, alarm-reset; ``instantaneous``: threshold on the mistake probability), with two
priors for the likelihood ratio (``global``: the training-fold mistake prevalence; ``filter``: the filter's own
predictive prior of the mistake status for the candidate, available when the dump carries the belief marginal),
and two position controls calibrated the same way (decision index, elapsed time).

Beyond the recall of all, first, and later mistakes, the timing of every detection is measured against the
annotated mistake: the delay from the mistake's end (negative: alarm during the mistake), the share of detections
raised before the mistake ends, and the share raised before the following step's decision.  ``--grace 0`` removes
the one-decision tolerance of the protocol.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

ERROR = 1
NORMAL = 0


def load_videos(path: str) -> list[dict]:
    return list(np.load(path, allow_pickle=True)["videos"])


def prior_probability(v: dict, prevalence: float, prior: str, transition: np.ndarray | None) -> np.ndarray:
    """Per-candidate prior probability of the mistake status."""

    n = len(v["anytime_score"])
    if prior == "filter":
        marginal = np.asarray(v.get("belief_in_mode_marginal", np.zeros((0, 0))), dtype=np.float64)
        if marginal.ndim != 2 or len(marginal) != n or transition is None:
            raise ValueError("the dump carries no compatible belief marginal for the filter prior")
        modes = transition.shape[0]
        if marginal.shape[1] != modes:
            # single-stream models dump the belief over (status, lifecycle) with the lifecycle axis
            # fastest; the status prior sums out the lifecycle
            if marginal.shape[1] % modes:
                raise ValueError("the dump carries no compatible belief marginal for the filter prior")
            marginal = marginal.reshape(n, modes, -1).sum(axis=2)
        marginal = marginal / np.clip(marginal.sum(axis=1, keepdims=True), 1e-12, None)
        return marginal @ transition[:, ERROR]
    return np.full(n, prevalence, dtype=np.float64)


def evidence(scores: np.ndarray, prior: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(scores, dtype=np.float64), 1e-6, 1 - 1e-6)
    q = np.clip(np.asarray(prior, dtype=np.float64), 1e-6, 1 - 1e-6)
    return (s / (1 - s)) / (q / (1 - q))


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
    duration = float(decoded[-1] - decoded[0]) if len(decoded) > 1 else 0.0
    step_time = np.diff(np.concatenate(([decoded[0]], native_time))) if len(native_time) else np.zeros(0)
    mistake_time = float(step_time[mistakes].clip(min=0).sum()) if mistakes and len(step_time) == len(native_time) else 0.0
    correct_minutes = max(duration - mistake_time, 0.0) / 60.0
    alarm_set = set(alarms)
    false_alarms = [i for i in alarms if native_mode[assignment[i]] != ERROR]
    detected, delay_end, delay_start, before_end, before_next = {}, [], [], [], []
    covered = [j for j in mistakes if (assignment == j).any()]
    for j in mistakes:
        idx = np.flatnonzero(assignment == j)
        if not len(idx):
            continue
        window = set(range(int(idx[0]), min(int(idx[-1]) + grace + 1, len(decoded))))
        hits = sorted(window & alarm_set)
        if hits:
            detected[j] = hits[0]
            t_alarm = float(decoded[hits[0]])
            end = float(native_time[j])
            start = float(native_time[j - 1]) if j > 0 else float(decoded[0])
            delay_end.append(t_alarm - end)
            delay_start.append(t_alarm - start)
            before_end.append(bool(t_alarm <= end))
            next_end = native_time[j + 1] if j + 1 < len(native_time) else np.inf
            before_next.append(bool(t_alarm <= next_end))
    # mistakes that follow a mistake (second or later in the execution) and mistakes after a correction
    later = [j for j in covered[1:]] if covered else []
    after_correction = [j for j in covered if any(native_mode[k] == 2 for k in range(j))]
    return {
        "has_mistake": bool(mistakes),
        "n_mistakes": len(mistakes),
        "n_covered": len(covered),
        "first_detected": bool(mistakes and mistakes[0] in detected),
        "n_detected": len(detected),
        "n_later": len(later),
        "n_later_detected": sum(1 for j in later if j in detected),
        "n_after_correction": len(after_correction),
        "n_after_correction_detected": sum(1 for j in after_correction if j in detected),
        "false_alarms": len(false_alarms),
        "zero_alarms": len(alarms) == 0,
        "duration_min": duration / 60.0,
        "correct_minutes": correct_minutes,
        "delay_end": delay_end,
        "delay_start": delay_start,
        "before_end": before_end,
        "before_next": before_next,
    }


def aggregate(stats: list[dict]) -> dict:
    correct = [s for s in stats if not s["has_mistake"]]
    mist = [s for s in stats if s["has_mistake"]]
    delay_end = [d for s in stats for d in s["delay_end"]]
    delay_start = [d for s in stats for d in s["delay_start"]]
    before_end = [b for s in stats for b in s["before_end"]]
    before_next = [b for s in stats for b in s["before_next"]]

    def ratio(num: str, den: str) -> float:
        d = sum(s[den] for s in stats)
        return sum(s[num] for s in stats) / d if d else float("nan")

    return {
        "fa_per_min_correct_operation": sum(s["false_alarms"] for s in stats) / max(sum(s["correct_minutes"] for s in stats), 1e-9),
        "recall_all": ratio("n_detected", "n_mistakes"),
        "recall_first": sum(1 for s in mist if s["first_detected"]) / len(mist) if mist else float("nan"),
        "recall_later": ratio("n_later_detected", "n_later"),
        "n_later": sum(s["n_later"] for s in stats),
        "recall_after_correction": ratio("n_after_correction_detected", "n_after_correction"),
        "n_after_correction": sum(s["n_after_correction"] for s in stats),
        "median_delay_end_sec": float(np.median(delay_end)) if delay_end else float("nan"),
        "p90_delay_end_sec": float(np.percentile(delay_end, 90)) if delay_end else float("nan"),
        "median_delay_start_sec": float(np.median(delay_start)) if delay_start else float("nan"),
        "before_end_share": float(np.mean(before_end)) if before_end else float("nan"),
        "before_next_step_share": float(np.mean(before_next)) if before_next else float("nan"),
        "zero_alarm_correct_share": sum(1 for s in correct if s["zero_alarms"]) / max(len(correct), 1),
        "n_videos": len(stats),
        "n_correct_videos": len(correct),
        "n_mistakes": sum(s["n_mistakes"] for s in stats),
    }


def statistic_inputs(videos: list[dict], rule: str, prevalence: float, prior: str, transition) -> list[np.ndarray]:
    out = []
    for v in videos:
        if rule == "sequential":
            out.append(evidence(v["anytime_score"], prior_probability(v, prevalence, prior, transition)))
        else:
            out.append(np.asarray(v["anytime_score"], dtype=float))
    return out


def evaluate(videos: list[dict], inputs: list[np.ndarray], rule: str, threshold: float, grace: int) -> dict:
    stats = [video_stats(v, alarms_for(x, threshold, rule), grace) for v, x in zip(videos, inputs)]
    out = aggregate(stats)
    out.update({"rule": rule, "threshold": float(threshold)})
    return out


def threshold_grid(inputs: list[np.ndarray], rule: str) -> np.ndarray:
    if rule == "sequential":
        values = []
        for x in inputs:
            m = 0.0
            for e in x:
                m = (1.0 + m) * e
                values.append(m)
    else:
        values = [float(e) for x in inputs for e in x]
    return np.unique(np.quantile(np.asarray(values, dtype=float), np.linspace(0.0, 1.0, 400)))


def calibrate(videos: list[dict], inputs: list[np.ndarray], rule: str, grace: int, budgets, key: str) -> dict:
    """Select, per budget, the threshold with the highest validation recall whose validation rate meets the budget."""

    rows = [evaluate(videos, inputs, rule, float(t), grace) for t in threshold_grid(inputs, rule)]
    chosen = {}
    for b in budgets:
        feasible = [r for r in rows if r[key] <= b]
        best = max(feasible, key=lambda r: (r["recall_all"], -r["threshold"])) if feasible else max(rows, key=lambda r: r["threshold"])
        chosen[str(b)] = {"threshold": best["threshold"], "validation": best}
    return chosen


def control_videos(videos: list[dict], kind: str, training_prior: dict | None = None) -> list[dict]:
    vs = [copy.deepcopy(v) for v in videos]
    for v in vs:
        n = len(v["anytime_score"])
        t = np.asarray(v["decoded_time"], dtype=float)
        if kind == "candidate_index":
            score = 1.0 - 1.0 / (np.arange(n) + 2.0)
        elif kind == "elapsed_time":
            score = 1.0 - 1.0 / (1.0 + (t - t[0]) / 60.0)
        elif kind == "training_index":
            # training-fold mistake rate at the decision index within the execution's task (task identity and
            # training statistics only; no frame is read)
            if training_prior is None:
                raise ValueError("the training_index control needs --training-prior")
            task = training_prior["video_task"].get(str(v["video_id"]))
            rates = training_prior["by_index"].get(task, training_prior["global"])
            score = np.asarray([rates[min(i, len(rates) - 1)] for i in range(n)], dtype=float)
        else:
            raise ValueError(kind)
        v["anytime_score"] = np.clip(score, 1e-4, 1 - 1e-4)
        v.pop("belief_in_mode_marginal", None)
    return vs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validation-dump", required=True)
    ap.add_argument("--test-dump", required=True)
    ap.add_argument("--prevalence", type=float, required=True, help="training-fold mistake prevalence")
    ap.add_argument("--prior", choices=("global", "filter"), default="global")
    ap.add_argument("--checkpoint", help="PROVIA checkpoint whose execution-status transition defines the filter prior")
    ap.add_argument("--grace", type=int, default=1)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    ap.add_argument("--budget-key", default="fa_per_min_correct_operation")
    ap.add_argument("--rules", nargs="+", default=["sequential", "instantaneous"])
    ap.add_argument("--controls", nargs="*", default=["candidate_index", "elapsed_time"])
    ap.add_argument("--training-prior", help="JSON of build_training_position_prior.py for the training_index control")
    ap.add_argument("--output")
    a = ap.parse_args()

    transition = None
    if a.prior == "filter":
        import torch

        state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)["model"]
        transition = state["filter.transition_probability"].double().numpy()
    val = load_videos(a.validation_dump)
    test = load_videos(a.test_dump)
    result = {"validation_dump": a.validation_dump, "test_dump": a.test_dump, "prevalence": a.prevalence, "prior": a.prior, "grace": a.grace, "budget_key": a.budget_key}
    for rule in a.rules:
        val_in = statistic_inputs(val, rule, a.prevalence, a.prior, transition)
        test_in = statistic_inputs(test, rule, a.prevalence, a.prior, transition)
        chosen = calibrate(val, val_in, rule, a.grace, a.budgets, a.budget_key)
        result[rule] = {b: {"threshold": c["threshold"], "validation": c["validation"], "test": evaluate(test, test_in, rule, c["threshold"], a.grace)} for b, c in chosen.items()}
    training_prior = json.load(open(a.training_prior)) if a.training_prior else None
    for kind in a.controls:
        cv, ct = control_videos(val, kind, training_prior), control_videos(test, kind, training_prior)
        rule = "sequential"
        val_in = statistic_inputs(cv, rule, a.prevalence, "global", None)
        test_in = statistic_inputs(ct, rule, a.prevalence, "global", None)
        chosen = calibrate(cv, val_in, rule, a.grace, a.budgets, a.budget_key)
        result[f"control:{kind}"] = {b: {"threshold": c["threshold"], "validation": c["validation"], "test": evaluate(ct, test_in, rule, c["threshold"], a.grace)} for b, c in chosen.items()}
    keys = ("recall_all", "recall_first", "recall_later", "fa_per_min_correct_operation", "before_end_share", "median_delay_end_sec", "zero_alarm_correct_share")
    for name, table in result.items():
        if not isinstance(table, dict) or "0.1" not in table and str(a.budgets[0]) not in table:
            continue
        for b, r in table.items():
            t = r["test"]
            print(f"{name:26s} budget {b}: tau={r['threshold']:.4g} " + " ".join(f"{k}={t[k]:.3f}" for k in keys))
    if a.output:
        Path(a.output).write_text(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
