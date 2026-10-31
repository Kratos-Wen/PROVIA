"""Fine-tunes DINOv2-S/14 on hand-object crops of the IMPACT-ego training fold, supervised by the per-hand (verb, noun)
step labels; model selection by validation accuracy.
"""
import argparse, json, random, time, threading, queue
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
if torch.cuda.is_available():
    torch.zeros(1).cuda()  # decord must load after the CUDA runtime
import decord
from provia.data.io import read_jsonl
from provia.model.intents import CompositionalIntentSpace
from provia.training.backbone import _load_backbone, _preprocess
from provia.training.common import _partitions

def build_items(manifest, cache, fold):
    train, _ = _partitions(tuple(read_jsonl(manifest)), fold)
    labels = CompositionalIntentSpace.from_segments(s for e in train for s in e.segments).labels
    v2i = {v: i for i, v in enumerate(labels["verb"])}; n2i = {n: i for i, n in enumerate(labels["noun"])}
    vids = {"train": [], "val": []}
    for line in open(manifest):
        e = json.loads(line); part = {r["fold"]: r["partition"] for r in e["attributes"]["cross_validation_roles"]}.get(fold)
        part = {"validation": "val"}.get(part, part)
        if part not in vids: continue
        z = np.load(f"{cache}/{e['video_id']}_hand_features.npz"); ts = z["timestamps_sec"]; fid = z["frame_ids"]; pres = z["hand_presence"]; boxes = z["hand_boxes"]
        yv = np.full((len(ts), 2), -100, np.int64); yn = yv.copy()
        for s in e["segments"]:
            m = (ts >= s["start_sec"]) & (ts <= s["end_sec"]); h = 0 if s["hand"] == "left" else 1
            yv[m, h] = v2i.get(s["verb"] or "", 0); yn[m, h] = n2i.get(s["noun"] or "", 0)
        keep = (yv >= 0) & pres
        items = [(int(fid[t]), h, boxes[t, h].tolist(), int(yv[t, h]), int(yn[t, h])) for t, h in zip(*np.where(keep))]
        vids[part].append((e["video_id"], e["video_path"], items))
    return vids, len(labels["verb"]), len(labels["noun"])

def crop(frame, box, scale, size, jitter=0.0):
    H, W = frame.shape[:2]; x0, y0, x1, y1 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; s = max(x1 - x0, y1 - y0) * scale * (1 + jitter * random.uniform(-1, 1)); s = max(s, 48)
    if jitter: cx += s * jitter * random.uniform(-0.5, 0.5); cy += s * jitter * random.uniform(-0.5, 0.5)
    X0, Y0, X1, Y1 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2)), int(min(W, cx + s / 2)), int(min(H, cy + s / 2))
    c = frame[Y0:Y1, X0:X1]
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA) if c.size else np.zeros((size, size, 3), np.uint8)

def video_crops(video, items, scale, size, train, max_per_video=None):
    reader = decord.VideoReader(video, ctx=decord.cpu(0), num_threads=8)
    items = sorted(items); 
    if max_per_video and len(items) > max_per_video: items = sorted(random.sample(items, max_per_video))
    fids = sorted({it[0] for it in items}); out = []
    for s in range(0, len(fids), 64):
        chunk = fids[s:s + 64]; frames = reader.get_batch(chunk).asnumpy(); lookup = dict(zip(chunk, frames))
        for it in items:
            if it[0] in lookup:
                f = lookup[it[0]]; c = crop(f, it[2], scale, size, jitter=0.15 if train else 0.0)
                if train and random.random() < 0.5: c = c[:, ::-1]
                out.append((c, it[3], it[4]))
        if len(out) >= 2048: yield out; out = []
    if out: yield out

class Producer(threading.Thread):
    """Decodes videos on `workers` threads and feeds crop batches through one queue (None = done)."""
    def __init__(s, videos, scale, size, train, subsample, workers=3):
        super().__init__(daemon=True); s.q = queue.Queue(maxsize=8); s.videos = list(videos); s.scale = scale; s.size = size; s.train = train; s.sub = subsample; s.workers = workers
    def _worker(s, vs):
        for vid, path, items in vs:
            for batch in video_crops(path, items, s.scale, s.size, s.train, s.sub): s.q.put(batch)
    def run(s):
        vs = list(s.videos)
        if s.train: random.shuffle(vs)
        threads = [threading.Thread(target=s._worker, args=(vs[i::s.workers],), daemon=True) for i in range(s.workers)]
        for t in threads: t.start()
        for t in threads: t.join()
        s.q.put(None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--output", required=True); ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--head-lr", type=float, default=1e-3); ap.add_argument("--scale", type=float, default=1.6); ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--train-per-video", type=int, default=2500); ap.add_argument("--val-per-video", type=int, default=1500); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repo", type=Path, default=Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"); ap.add_argument("--checkpoint", type=Path, default=Path.home() / ".cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed); out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    vids, NV, NN = build_items(a.manifest, a.cache, 0)
    print("train videos", len(vids["train"]), "val", len(vids["val"]), "train crops", sum(len(v[2]) for v in vids["train"]), flush=True)
    dev = torch.device(a.device); backbone = _load_backbone(a.repo, a.checkpoint, dev, backbone="dinov2_vits14"); backbone.train()
    for p in backbone.parameters(): p.requires_grad_(True)
    if hasattr(backbone, "mask_token"): backbone.mask_token.requires_grad_(False)
    heads = nn.ModuleDict({"verb": nn.Linear(384, NV), "noun": nn.Linear(384, NN)}).to(dev)
    opt = torch.optim.AdamW([{"params": [p for p in backbone.parameters() if p.requires_grad], "lr": a.lr}, {"params": heads.parameters(), "lr": a.head_lr}], weight_decay=0.05)
    est_steps = a.epochs * max(1, min(sum(len(v[2]) for v in vids["train"]), len(vids["train"]) * a.train_per_video) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr, a.head_lr], total_steps=est_steps + 50, pct_start=0.1)
    def forward(x):
        with torch.autocast(dev.type, dtype=torch.bfloat16): cls = backbone.forward_features(x)["x_norm_clstoken"]
        return heads["verb"](cls.float()), heads["noun"](cls.float())
    @torch.no_grad()
    def evaluate():
        backbone.eval(); cv = cn = ct = 0; prod = Producer(vids["val"], a.scale, a.size, False, a.val_per_video); prod.start()
        while True:
            batch = prod.q.get()
            if batch is None: break
            for s in range(0, len(batch), 256):
                sub = batch[s:s + 256]; x = _preprocess(np.stack([b[0] for b in sub]), dev); yv = torch.tensor([b[1] for b in sub], device=dev); yn = torch.tensor([b[2] for b in sub], device=dev)
                lv, ln = forward(x); cv += (lv.argmax(-1) == yv).sum().item(); cn += (ln.argmax(-1) == yn).sum().item(); ct += len(sub)
        backbone.train(); return {"val_verb": cv / ct, "val_noun": cn / ct, "val_crops": ct}
    best = None; hist = []; step = 0
    for ep in range(a.epochs):
        t0 = time.time(); prod = Producer(vids["train"], a.scale, a.size, True, a.train_per_video); prod.start(); buf = []; tot = nb = 0
        while True:
            batch = prod.q.get()
            if batch is None: break
            buf.extend(batch); random.shuffle(buf)
            while len(buf) >= a.batch:
                sub, buf = buf[:a.batch], buf[a.batch:]
                x = _preprocess(np.stack([b[0] for b in sub]), dev); yv = torch.tensor([b[1] for b in sub], device=dev); yn = torch.tensor([b[2] for b in sub], device=dev)
                lv, ln = forward(x); loss = F.cross_entropy(lv, yv, label_smoothing=0.1) + F.cross_entropy(ln, yn, label_smoothing=0.1)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(heads.parameters()), 1.0); opt.step()
                if step < est_steps + 49: sched.step()
                step += 1; tot += loss.item(); nb += 1
                if nb % 200 == 0: print(json.dumps({"epoch": ep, "step": nb, "loss": round(tot / nb, 4), "min": round((time.time() - t0) / 60, 1)}), flush=True)
        r = evaluate(); r.update(epoch=ep, train_loss=tot / max(1, nb), minutes=round((time.time() - t0) / 60, 1)); hist.append(r); print(json.dumps(r), flush=True)
        if best is None or r["val_verb"] + r["val_noun"] > best[0]:
            best = (r["val_verb"] + r["val_noun"], r); torch.save(backbone.state_dict(), out / "dinov2_vits14_hand_object_finetuned.pth"); torch.save(heads.state_dict(), out / "heads.pt")
    json.dump({"best": best[1], "history": hist, "args": {k: str(v) for k, v in vars(a).items()}}, open(out / "result.json", "w"), indent=2); print("BEST", json.dumps(best[1]))

if __name__ == "__main__":
    main()
