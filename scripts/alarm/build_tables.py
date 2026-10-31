"""Alarm-level results for every method, dataset and seed.  The validation scores fix one threshold per false-alarm
budget and the test scores are evaluated once at that threshold; results are averaged over seeds.
Usage: build_tables.py [--grace 1]
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np

root = Path("runs")
TRAIN_PREV = {"captaincook4d": 915 / 2821, "industreal": 12 / 206, "holoassist": 5235 / 102330, "impact_ego": 468 / 4726}
METHODS = ("provia", "causal_tcn", "prego", "mistsense_rgb")
BUDGETS = ["0.1", "0.5", "1.0"]
KEYS = ("recall_all", "recall_first", "recall_later", "fa_per_min_correct_operation", "before_end_share", "before_next_step_share", "median_delay_end_sec", "median_delay_start_sec", "zero_alarm_correct_share")


def dumps_for(ds: str, method: str, seed: int):
    """(validation dump, test dump, checkpoint) for one run."""
    d = root / ds / f"model_seed{seed}" if method == "provia" else root / ds / "baselines" / f"{method}_seed{seed}"
    return d / "validation_candidates.npz", d / "test_candidates.npz", d / "best.pt" if method == "provia" else None


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--grace", type=int, default=1)
    ap.add_argument("--datasets", nargs="+", default=list(TRAIN_PREV))
    a = ap.parse_args()
    out_json = root / "alarm_tables.json"
    summary = {}
    for ds in a.datasets:
        prev = TRAIN_PREV[ds]; summary[ds] = {}
        for method in METHODS:
            runs = []
            for seed in (0, 1, 2):
                val, test, ckpt = dumps_for(ds, method, seed)
                if not val.exists() or not test.exists():
                    continue
                prior = "filter" if method == "provia" and ds != "impact_ego" else "global"
                out = root / ds / "alarm" / f"{method}_seed{seed}.json"; out.parent.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, "-m", "provia.evaluation.alarm_protocol_calibrated", "--validation-dump", str(val), "--test-dump", str(test), "--prevalence", str(prev), "--prior", prior, "--grace", str(a.grace), "--budgets", *BUDGETS, "--output", str(out)]
                if prior == "filter":
                    cmd += ["--checkpoint", str(ckpt)]
                if method == "provia":
                    prior_file = root / ds / "training_position_prior.json"
                    has_prior = prior_file.exists() and bool(json.load(open(prior_file)).get("by_index"))
                    cmd += ["--controls", "candidate_index", "elapsed_time"] + (["training_index", "--training-prior", str(prior_file)] if has_prior else [])
                else:
                    cmd += ["--controls"]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"FAILED {ds} {method} seed{seed}: {r.stderr[-400:]}"); continue
                runs.append(json.load(open(out)))
            if not runs:
                continue
            table = {}
            for name in ("sequential", "instantaneous", "control:candidate_index", "control:elapsed_time", "control:training_index"):
                if (method != "provia" and name.startswith("control")) or any(name not in r for r in runs):
                    continue
                table[name] = {}
                for b in BUDGETS:
                    vals = {k: [r[name][b]["test"][k] for r in runs if isinstance(r[name][b]["test"].get(k), float)] for k in KEYS}
                    table[name][b] = {k: (float(np.nanmean(v)) if v else float("nan")) for k, v in vals.items()}
                    table[name][b]["threshold"] = [r[name][b]["threshold"] for r in runs]
                    table[name][b]["n_runs"] = len(runs)
            summary[ds][method] = table
            seq = table["sequential"]
            print(f"{ds:14s} {method:14s} seq  " + " | ".join(f"{b}: R={seq[b]['recall_all']:.3f} FA={seq[b]['fa_per_min_correct_operation']:.3f} first={seq[b]['recall_first']:.3f} later={seq[b]['recall_later']:.3f} before_end={seq[b]['before_end_share']:.2f}" for b in BUDGETS))
            ins = table["instantaneous"]
            print(f"{'':14s} {'':14s} inst " + " | ".join(f"{b}: R={ins[b]['recall_all']:.3f} FA={ins[b]['fa_per_min_correct_operation']:.3f}" for b in BUDGETS))
    json.dump(summary, open(out_json, "w"), indent=1, default=float); print("written", out_json)


if __name__ == "__main__":
    main()
