"""Training-fold mistake statistics by step index, per task, for a perception-free position control.

Reads the benchmark manifest, keeps the training split, and records for every task the fraction of annotated
steps at each step index that are mistakes (Jeffreys-smoothed).  The control scores a decoded decision by the
training rate at its own decision index within the execution's task, so it uses task identity and training
statistics but no frame.  Output: JSON with ``by_index`` (task -> rates), ``global`` (rates over all tasks) and
``video_task`` (video id -> task) for every split.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--train-split", default="train"); ap.add_argument("--max-index", type=int, default=60)
    a = ap.parse_args()
    counts = defaultdict(lambda: [[0, 0] for _ in range(a.max_index + 1)])
    total = [[0, 0] for _ in range(a.max_index + 1)]
    video_task = {}
    with open(a.manifest) as handle:
        for line in handle:
            rec = json.loads(line)
            task = str(rec.get("task_id")); video_task[str(rec["video_id"])] = task
            if rec.get("split") != a.train_split:
                continue
            segments = [s for s in rec.get("segments", []) if s.get("observed", True)]
            for hand in sorted({s.get("hand") for s in segments}, key=str):
                ordered = sorted((s for s in segments if s.get("hand") == hand), key=lambda s: s.get("end_sec", s.get("onset_sec", 0.0)))
                for index, s in enumerate(ordered):
                    i = min(index, a.max_index)
                    mistake = int(str(s.get("correctness", "correct")).lower() not in ("correct", "normal", "0", "true"))
                    counts[task][i][0] += mistake; counts[task][i][1] += 1
                    total[i][0] += mistake; total[i][1] += 1
    rate = lambda c: [(m + 0.5) / (n + 1.0) for m, n in c]
    out = {"by_index": {t: rate(c) for t, c in counts.items()}, "global": rate(total), "video_task": video_task, "train_split": a.train_split}
    Path(a.output).write_text(json.dumps(out))
    print("tasks", len(counts), "videos", len(video_task), "global rate by index (first 8):", [round(x, 3) for x in out["global"][:8]])


if __name__ == "__main__":
    main()
