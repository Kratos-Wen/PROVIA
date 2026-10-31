"""Qualitative figure: one complete CaptainCook4D test execution (18_27, Zoodles) with the seed-0 model, at the threshold
fixed on validation for 0.5 false alarms per minute.  Rows: frames from six steps with their annotated error types and a
line to the frame time, the annotated steps with the predicted completions, the reported mistake probability per frame,
and the Shiryaev-Roberts statistic with the alarm regions shaded.  Run after scripts/alarm/run.sh.
Usage: qualitative.py [--seed 0] [--video 18_27]
"""
import argparse, json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import ConnectionPatch
import numpy as np
import torch

from provia.evaluation.alarm_protocol_calibrated import ERROR, alarms_for, evidence, load_videos, prior_probability

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--video", default="18_27")
ap.add_argument("--output", type=Path, default=Path("runs/captaincook4d/qualitative"))
a = ap.parse_args()
DS, SEED, VID, OUT = "captaincook4d", a.seed, a.video, a.output
PREV = 915 / 2821
SHOWN_STEPS = (2, 4, 5, 7, 8, 9)  # zero-based indices of the annotated steps shown as frames
OUT.mkdir(parents=True, exist_ok=True)
run = Path("runs") / DS

record = next(json.loads(l) for l in open(f"data/{DS}.jsonl") if f'"video_id": "{VID}"' in l)
thr = float(json.load(open(run / "alarm" / f"provia_seed{SEED}.json"))["sequential"]["0.5"]["threshold"])
state = torch.load(run / f"model_seed{SEED}" / "best.pt", map_location="cpu", weights_only=False)["model"]
transition = state["filter.transition_probability"].double().numpy()
v = next(w for w in load_videos(run / f"model_seed{SEED}" / "test_candidates.npz") if w["video_id"] == VID)
prior = prior_probability(v, PREV, "filter", transition)
xi = evidence(v["anytime_score"], prior)
S, m = [], 0.0
for x in xi:
    m = (1.0 + m) * x; S.append(m)
    if m >= thr: m = 0.0
S = np.asarray(S); alarms = alarms_for(xi, thr, "sequential")
dec = np.asarray(v["decoded_time"], float); assign = np.asarray(v["assignment"]); nmode = np.asarray(v["native_mode"]); ntime = np.asarray(v["native_time"], float)
false = [i for i in alarms if nmode[assign[i]] != ERROR]
segs = record["segments"]
print("steps", len(segs), "decisions", len(dec), "threshold", round(thr, 3), "alarms", [round(dec[i], 1) for i in alarms], "false", false)
json.dump(dict(video=VID, seed=SEED, threshold=thr, decoded_time=dec.tolist(), S=S.tolist(), alarms=[int(i) for i in alarms], false_alarms=false), open(OUT / f"{VID}_seed{SEED}.json", "w"), indent=1)

T0, T1 = 0.0, float(record["duration_sec"])
comp = np.asarray([s["end_sec"] for s in segs], float)
edges = np.concatenate(([T0], 0.5 * (comp[1:] + comp[:-1]), [T1]))  # completion cells

# frames: three seconds before the completion of each shown step, inside its cell
cap = cv2.VideoCapture(record["video_path"]); frames = []
for k in SHOWN_STEPS:
    t = max(comp[k] - 3.0, edges[k] + 0.5)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0); ok, img = cap.read()
    if not ok: raise RuntimeError(f"no frame at {t}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); h, w = img.shape[:2]; img = img[:, (w - h * 4 // 3) // 2:(w + h * 4 // 3) // 2]
    frames.append((k, t, img))
cap.release()

def label(k):
    s = segs[k]
    if s["correctness"] != "error": return f"step {k + 1}\ncorrect"
    return f"step {k + 1}\n" + "\n".join(e.lower() for e in (s.get("error_types") or []))

plt.rcParams.update({"font.size": 7, "font.family": "sans-serif", "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "pdf.fonttype": 42})
fig = plt.figure(figsize=(3.45, 2.45))
outer = gridspec.GridSpec(4, 1, height_ratios=[0.42, 0.4, 0.75, 0.6], hspace=0.16, left=0.12, right=0.985, top=0.905, bottom=0.15)
fgs = gridspec.GridSpecFromSubplotSpec(1, len(frames), subplot_spec=outer[0], wspace=0.06)
axg = fig.add_subplot(outer[1]); axa = fig.add_subplot(outer[2], sharex=axg); axs = fig.add_subplot(outer[3], sharex=axg)
frame_axes = []
for j, (k, t, img) in enumerate(frames):
    ax = fig.add_subplot(fgs[0, j]); ax.imshow(img); ax.axis("off"); frame_axes.append(ax)
    col = "#d62728" if segs[k]["correctness"] == "error" else "#2ca02c"
    ax.text(0.5, 1.04, label(k), transform=ax.transAxes, ha="center", va="bottom", fontsize=4.0, color=col, linespacing=1.05)
# annotated steps as completion cells, decoded completions as ticks
for k, s in enumerate(segs):
    c = "#d62728" if s["correctness"] == "error" else "#2ca02c"
    axg.axvspan(edges[k], edges[k + 1], ymin=0.35, ymax=1.0, color=c, alpha=0.85, lw=0)
    axg.axvline(edges[k + 1], ymin=0.35, ymax=1.0, color="white", lw=0.4)
    if edges[k + 1] - edges[k] > 0.025 * (T1 - T0):
        axg.text(0.5 * (edges[k] + edges[k + 1]), 0.675, str(k + 1), ha="center", va="center", fontsize=5.5, color="white")
for t in dec:
    axg.plot([t, t], [0.0, 0.28], color="k", lw=0.6)
axg.set_ylim(0, 1); axg.set_yticks([]); axg.tick_params(labelbottom=False); axg.set_ylabel("steps", fontsize=7)
for sp in ("top", "right", "left"): axg.spines[sp].set_visible(False)
axg.set_xlim(T0, T1)
# leader lines from each frame to its time on the step bar
for ax, (k, t, img) in zip(frame_axes, frames):
    fig.add_artist(ConnectionPatch(xyA=(0.5, 0.0), coordsA=ax.transAxes, xyB=(t, 1.0), coordsB=axg.transData, color="0.25", lw=1.0, zorder=5))
# reported mistake probability per frame (dump times are relative to the assigned step's completion)
for i, po in enumerate(v["prefix_options"]):
    po = np.asarray(po); axa.plot(po[:, 1] + ntime[assign[i]], po[:, 0], color="#1f77b4", lw=0.8)
axa.set_ylabel(r"$a_{i,t}$", fontsize=7); axa.set_ylim(0, 1.05 * max(np.asarray(p)[:, 0].max() for p in v["prefix_options"]))
axa.tick_params(labelbottom=False, labelsize=6); axa.spines["top"].set_visible(False); axa.spines["right"].set_visible(False)
# Shiryaev-Roberts statistic; the regions above the threshold are the alarms
axs.axhline(thr, color="0.4", lw=0.6, ls="--"); axs.text(T1, thr, r"$\tau$", ha="right", va="bottom", fontsize=7, color="0.3")
axs.plot(dec, S, color="k", lw=0.7, marker="o", ms=1.8)
axs.fill_between(dec, thr, S, where=S >= thr, interpolate=True, color="#d62728", alpha=0.75, lw=0)
axs.set_ylabel(r"$S_i$", fontsize=7); axs.set_xlabel("time (s)", fontsize=7); axs.tick_params(labelsize=6)
axs.spines["top"].set_visible(False); axs.spines["right"].set_visible(False); axs.set_ylim(0, 1.15 * float(S.max()))
fig.savefig(OUT / f"qualitative_{DS}_{VID}.pdf"); fig.savefig(OUT / f"qualitative_{DS}_{VID}.png", dpi=200)
print("figure written", OUT)
