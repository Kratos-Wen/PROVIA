"""LaTeX table of the alarm-level results: recall at 0.1 / 0.5 / 1.0 false alarms per minute with thresholds fixed on
validation, recall of the mistakes after the first one, and the position controls.
Usage: build_latex_table.py [--out FILE]
"""
import argparse, json
from pathlib import Path

root = Path("runs")
DATASETS = [("captaincook4d", "CaptainCook4D"), ("industreal", "IndustReal"), ("holoassist", "HoloAssist"), ("impact_ego", "IMPACT-ego")]
METHODS = [("causal_tcn", "Causal-TCN"), ("prego", "Controlled PREGO"), ("mistsense_rgb", "MistSense-RGB-style"), ("provia", "\\provia{}")]
CONTROLS = [("control:candidate_index", "Position control (index)"), ("control:elapsed_time", "Position control (time)"), ("control:training_index", "Position control (training)")]
COLS = [("0.1", "recall_all"), ("0.5", "recall_all"), ("1.0", "recall_all"), ("0.5", "recall_later")]


def fmt(x):
    return "--" if x is None or x != x else ("%.3f" % x).lstrip("0") if x < 1 else "%.2f" % x


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(root / "alarm_table.tex"))
    a = ap.parse_args()
    S = json.load(open(root / "alarm_tables.json"))
    rows = []
    values = {}
    for m, _ in METHODS:
        for ds, _ in DATASETS:
            t = S.get(ds, {}).get(m, {}).get("sequential")
            for c, (b, k) in enumerate(COLS):
                values[(m, ds, c)] = t[b][k] if t else None
    best = {}
    for ds, _ in DATASETS:
        for c in range(len(COLS)):
            cand = {m: values[(m, ds, c)] for m, _ in METHODS if values.get((m, ds, c)) is not None}
            best[(ds, c)] = max(cand, key=cand.get) if cand else None
    for m, name in METHODS:
        cells = []
        for ds, _ in DATASETS:
            for c in range(len(COLS)):
                v = values.get((m, ds, c)); txt = fmt(v)
                cells.append("\\textbf{" + txt + "}" if best[(ds, c)] == m else txt)
        rows.append(name + " & " + " & ".join(cells) + " \\\\")
    control_rows = []
    for key, name in CONTROLS:
        cells = []; present = False
        for ds, _ in DATASETS:
            t = S.get(ds, {}).get("provia", {}).get(key)
            for b, k in COLS:
                v = t[b][k] if t else None; present = present or v is not None
                cells.append(fmt(v))
        if present:
            control_rows.append(name + " & " + " & ".join(cells) + " \\\\")
    head = " & ".join("\\multicolumn{4}{c%s}{%s}" % ("|" if i < 3 else "", n) for i, (_, n) in enumerate(DATASETS))
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Alarm-level test results with thresholds fixed on validation: recall of all mistakes at 0.1/0.5/1 false alarms per minute of correct operation and recall of the mistakes after the first one at 0.5 per minute (Later); sequential change detector with the filter prior (training-prevalence prior on IMPACT-ego), mean of seeds 0--2.  The position controls score decisions by their index, by elapsed time, or by the training-fold mistake rate at that index.}",
             "\\label{tab:alarm_test}", "\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}",
             "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lcccc|cccc|cccc|cccc}", "\\toprule",
             "& " + head + " \\\\",
             "Method & " + " & ".join(["0.1 & 0.5 & 1.0 & Later"] * 4) + " \\\\", "\\midrule", *rows, "\\midrule", *control_rows,
             "\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\n% measured test false alarms per minute at each budget (sequential rule)")
    for m, name in METHODS:
        cells = []
        for ds, _ in DATASETS:
            t = S.get(ds, {}).get(m, {}).get("sequential")
            cells.append(" / ".join(fmt(t[b]["fa_per_min_correct_operation"]) if t else "--" for b in ("0.1", "0.5", "1.0")))
        print(f"% {name:22s} " + " | ".join(cells))
    print("\n% PROVIA timeliness and later-mistake recall at 0.5/min (sequential rule, filter prior where available)")
    for ds, dn in DATASETS:
        t = S.get(ds, {}).get("provia", {}).get("sequential")
        if not t:
            continue
        b = t["0.5"]
        print(f"% {dn:14s} recall all {fmt(b['recall_all'])} first {fmt(b['recall_first'])} later {fmt(b['recall_later'])} | before end {fmt(b['before_end_share'])} before next step {fmt(b.get('before_next_step_share'))} | median delay after mistake end {b['median_delay_end_sec']:.1f}s, after start {b['median_delay_start_sec']:.1f}s | zero-alarm correct executions {fmt(b['zero_alarm_correct_share'])}")
        for m, name in METHODS[:-1]:
            tb = S.get(ds, {}).get(m, {}).get("sequential")
            if tb:
                bb = tb["0.5"]; print(f"%   {name:20s} later {fmt(bb['recall_later'])} before end {fmt(bb['before_end_share'])} delay end {bb['median_delay_end_sec']:.1f}s")


if __name__ == "__main__":
    main()
