"""Converts hand-object features into the two-hand feature cache read by the IMPACT-ego pipeline: slot 0/1 = left/right
hand-object crop descriptor, plus its per-second temporal difference (zero when the hand is absent).
"""
import argparse, json
from pathlib import Path
import numpy as np

def convert(hand_object_file, out_file, rate):
    z = np.load(hand_object_file); hf = z["hand_features"].astype(np.float32); pres = z["hand_presence"]
    change = np.zeros_like(hf); ok = pres[1:] & pres[:-1]; change[1:][ok] = (hf[1:] - hf[:-1])[ok] * rate
    np.savez(out_file, frame_ids=z["frame_ids"], timestamps_sec=z["timestamps_sec"], global_features=z["global_features"].astype(np.float16),
             role_features=hf.astype(np.float16), role_change_features=change.astype(np.float32))
    return len(z["frame_ids"])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=("query", "expert"), required=True)
    ap.add_argument("--hand-object", type=Path, required=True); ap.add_argument("--source", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--references", type=Path)
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    src = json.load(open(a.source / "manifest.json")); rate = float(src["sample_rate_hz"])
    if a.mode == "query":
        files = sorted(a.hand_object.glob("*_hand_features.npz")); frames = 0
        for f in files: frames += convert(f, a.output / f.name.replace("_hand_features", "_stream_features"), rate)
        m = dict(src); m.update({"role_mode": "hand_object_crops", "role_change_operator": "per-second temporal difference of the hand-object crop descriptor; zero when the hand is absent",
                                 "role_interpretation": "slot 0 = left hand-object region, slot 1 = right hand-object region", "crop_scale": 1.6, "episodes": len(files), "frames": int(frames)})
        json.dump(m, open(a.output / "manifest.json", "w"), indent=2, sort_keys=True); print("query cache", len(files), "videos", frames, "frames")
    else:
        ids = [l.strip() for l in open(a.references) if l.strip()]; frames = 0
        for v in ids: frames += convert(a.hand_object / f"{v}_hand_features.npz", a.output / f"{v}_stream_features.npz", rate)
        m = {"backbone": src["backbone"], "backbone_checkpoint": src["backbone_checkpoint"], "contains_annotations": False, "uses_expert_annotations": False, "frames": int(frames),
             "references": len(ids), "videos": len(ids), "role_change_evidence": True, "role_mode": "hand_object_crops", "sample_rate_hz": rate, "schema": "expert_role_stream_features",
             "test_split_accessed": False, "visual_change_units": "per_second"}
        json.dump(m, open(a.output / "manifest.json", "w"), indent=2, sort_keys=True); print("expert cache", len(ids), "references", frames, "frames")

if __name__ == "__main__":
    main()
