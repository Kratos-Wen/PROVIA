"""Hand-object descriptors aligned to an existing feature cache: detect hands, crop an enlarged square region per hand,
and encode each crop with DINOv2-S/14 (CLS token).
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np, torch, cv2
if torch.cuda.is_available():
    torch.zeros(1).cuda()  # decord must load after the CUDA runtime
import decord
from ultralytics import YOLO
from provia.training.backbone import _load_backbone, _preprocess

def crops_for(frame, boxes, scale, size=224):
    H, W = frame.shape[:2]; out = []
    for b in boxes:
        if b is None: out.append(None); continue
        x0, y0, x1, y1 = b; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; s = max(x1 - x0, y1 - y0) * scale; s = max(s, 48)
        X0, Y0, X1, Y1 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2)), int(min(W, cx + s / 2)), int(min(H, cy + s / 2))
        crop = frame[Y0:Y1, X0:X1]
        out.append(cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA) if crop.size else None)
    return out

def assign(boxes, confs, W):
    if len(boxes) == 0: return [None, None]
    order = np.argsort(-confs)[:2]; bs = [boxes[i] for i in order]
    if len(bs) == 2:
        bs = sorted(bs, key=lambda b: (b[0] + b[2]) / 2); return [bs[0], bs[1]]
    b = bs[0]; return [b, None] if (b[0] + b[2]) / 2 < W / 2 else [None, b]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True); ap.add_argument("--source-cache", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True); ap.add_argument("--partitions", nargs="+", default=["train", "validation"])
    ap.add_argument("--fold", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--detector", help="hand detector weights; not needed with --reuse-boxes"); ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25); ap.add_argument("--scale", type=float, default=1.6); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--repo", type=Path, default=Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main")
    ap.add_argument("--checkpoint", type=Path, default=Path.home() / ".cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth")
    ap.add_argument("--reuse-boxes", type=Path, default=None, help="existing hand-object cache: reuse its hand boxes and skip the detector")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); a.output_root.mkdir(parents=True, exist_ok=True); dev = torch.device(a.device)
    episodes = []
    for line in open(a.manifest):
        e = json.loads(line); roles = {r["fold"]: r["partition"] for r in e["attributes"].get("cross_validation_roles", [])}
        part = roles.get(a.fold) or e.get("split"); part = {"val": "validation"}.get(part, part)
        if part in a.partitions: episodes.append((e["video_id"], e["video_path"], part))
    episodes = sorted(episodes)[a.shard_index::a.num_shards]
    detector = None if a.reuse_boxes else YOLO(a.detector); backbone = _load_backbone(a.repo, a.checkpoint, dev, backbone="dinov2_vits14"); backbone.eval()
    done = 0; t_start = time.time()
    for vid, path, part in episodes:
        out = a.output_root / f"{vid}_hand_features.npz"
        if out.exists(): done += 1; continue
        src = np.load(a.source_cache / f"{vid}_stream_features.npz"); frame_ids = src["frame_ids"]; T = len(frame_ids)
        if a.reuse_boxes:
            prev = np.load(a.reuse_boxes / f"{vid}_hand_features.npz"); prev_boxes = prev["hand_boxes"]; prev_pres = prev["hand_presence"]; assert (prev["frame_ids"] == frame_ids).all()
        reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=4)
        feats = np.zeros((T, 2, 384), np.float16); boxes_out = np.zeros((T, 2, 4), np.float32); pres = np.zeros((T, 2), bool)
        for s in range(0, T, a.batch):
            idx = frame_ids[s:s + a.batch]; frames = reader.get_batch(idx.tolist()).asnumpy()  # RGB
            H, Wd = frames.shape[1:3]
            crop_list, owner = [], []
            if a.reuse_boxes:
                lrs = []
                for k in range(len(frames)):
                    t = s + k; lrs.append([None if not prev_pres[t, h] else [prev_boxes[t, h, 0] * Wd, prev_boxes[t, h, 1] * H, prev_boxes[t, h, 2] * Wd, prev_boxes[t, h, 3] * H] for h in (0, 1)])
            else:
                res = detector.predict([f[:, :, ::-1] for f in frames], imgsz=a.imgsz, conf=a.conf, verbose=False, device=0 if dev.type == "cuda" else "cpu", half=dev.type == "cuda")
                lrs = [assign(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), Wd) for r in res]
            for k, lr in enumerate(lrs):
                cr = crops_for(frames[k], lr, a.scale)
                for h in (0, 1):
                    if lr[h] is not None and cr[h] is not None:
                        b = lr[h]; boxes_out[s + k, h] = [b[0] / Wd, b[1] / H, b[2] / Wd, b[3] / H]; pres[s + k, h] = True; crop_list.append(cr[h]); owner.append((s + k, h))
            if crop_list:
                x = _preprocess(np.stack(crop_list), dev)
                with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16):
                    cls = backbone.forward_features(x)["x_norm_clstoken"].float().cpu().numpy()
                for (t, h), v in zip(owner, cls): feats[t, h] = v.astype(np.float16)
        np.savez(out.with_suffix(".tmp.npz"), frame_ids=frame_ids, timestamps_sec=src["timestamps_sec"], global_features=src["global_features"], hand_features=feats, hand_boxes=boxes_out, hand_presence=pres)
        os.replace(out.with_suffix(".tmp.npz"), out); done += 1
        print(json.dumps({"video": vid, "frames": int(T), "presence": [float(pres[:, 0].mean()), float(pres[:, 1].mean())], "done": done, "of": len(episodes), "elapsed_min": round((time.time() - t_start) / 60, 1)}), flush=True)
    print("shard complete", a.shard_index)

if __name__ == "__main__":
    main()
