#!/usr/bin/env python3
import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Iterator, Optional, Tuple

import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, accuracy_score


# ----------------------------
# basic utils
# ----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_json(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(p: str, obj: Dict[str, Any]):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ----------------------------
# manifest helpers (FIXED for your multimodal manifest)
# ----------------------------
def get_manifest_items(mf: Any) -> List[Dict[str, Any]]:
    """
    Supports:
      - multimodal manifest: {"paired_items":[...], ...}
      - sharded cache manifests: {"items":[...]}
      - alt: {"shards":[...]} or {"data":[...]}
      - already a list
    """
    if isinstance(mf, list):
        return mf
    if not isinstance(mf, dict):
        raise RuntimeError(f"Unexpected manifest type: {type(mf)}")

    for k in ("paired_items", "items", "shards", "data"):
        if k in mf and isinstance(mf[k], list):
            return mf[k]

    raise KeyError(f"Manifest has no items list. Top keys={list(mf.keys())}")

def get_section_dict(it: Dict[str, Any]) -> Dict[str, Any]:
    """
    In your multimodal manifest, section cache is under key 'section'.
    Some other code uses 'sec'. Support both.
    """
    if "section" in it and isinstance(it["section"], dict):
        return it["section"]
    if "sec" in it and isinstance(it["sec"], dict):
        return it["sec"]
    return {}

def get_tab_dict(it: Dict[str, Any]) -> Dict[str, Any]:
    if "tab" in it and isinstance(it["tab"], dict):
        return it["tab"]
    return {}

def iter_shards(
    manifest_path: str,
    img_key: str,
    sec1d_key: str,
) -> Iterator[Tuple[str, str, str, str, str, str]]:
    """
    Yields:
      base, img_path, sec1d_path, y_path, valid_path, sha_path
    using:
      it["section"][img_key], it["section"][sec1d_key], it["tab"]["y"], it["tab"]["valid"], it["tab"]["sha256"]
    """
    mf = load_json(manifest_path)
    items = get_manifest_items(mf)

    for it in items:
        if not isinstance(it, dict):
            continue

        base = it.get("base", "unknown_base")
        tab = get_tab_dict(it)
        sec = get_section_dict(it)

        if not tab or not sec:
            print(f"[WARN] Missing tab/section dict for base={base}")
            continue

        img_path = sec.get(img_key)
        sec1d_path = sec.get(sec1d_key)
        y_path = tab.get("y")
        v_path = tab.get("valid")
        sha_path = tab.get("sha256")

        if not img_path or not sec1d_path:
            print(f"[WARN] Missing section modality keys for base={base}")
            continue
        if not y_path or not v_path or not sha_path:
            print(f"[WARN] Missing tab y/valid/sha256 for base={base}")
            continue

        # existence checks
        for p in (img_path, sec1d_path, y_path, v_path, sha_path):
            if not os.path.exists(p):
                print(f"[WARN] File missing for base={base}: {p}")
                img_path = None
                break
        if img_path is None:
            continue

        yield base, img_path, sec1d_path, y_path, v_path, sha_path

def infer_sec1d_dim(train_manifest: str, sec1d_key: str) -> int:
    mf = load_json(train_manifest)
    items = get_manifest_items(mf)
    for it in items:
        sec = get_section_dict(it)
        p = sec.get(sec1d_key)
        if p and os.path.exists(p):
            arr = np.load(p, mmap_mode="r")
            if arr.ndim != 2:
                raise RuntimeError(f"Expected sec1d array ndim=2, got {arr.ndim} for {p}")
            return int(arr.shape[1])
    raise RuntimeError(f"Could not infer sec1d_dim. key={sec1d_key} not found or files missing.")


# ----------------------------
# preprocessing (safe)
# ----------------------------
def _prep_img(batch: np.ndarray) -> np.ndarray:
    """
    Your section image cache is typically (N, 17, 25) float16.
    Convert -> (N, 1, 17, 25) float32 in [0,1].
    """
    b = np.asarray(batch)
    if b.ndim == 3:
        b = b[:, None, :, :]
    b = b.astype(np.float32, copy=False)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    mx = float(np.max(b)) if b.size else 0.0
    if mx > 1.5:
        b = b / 255.0
    b = np.clip(b, 0.0, 1.0)
    return b

def _prep_sec1d(batch: np.ndarray) -> np.ndarray:
    """
    Your section 1D cache is typically (N, 224) float16.
    Convert -> float32 and clip.
    """
    b = np.asarray(batch).astype(np.float32, copy=False)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    b = np.clip(b, -1e3, 1e3)
    return b


# ----------------------------
# models
# ----------------------------
class ImgEncoder(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, emb_dim)

    def forward(self, x):
        x = self.net(x).squeeze(-1).squeeze(-1)
        return self.fc(x)

class Sec1DEncoder(nn.Module):
    def __init__(self, in_dim=224, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, emb_dim),
        )

    def forward(self, x):
        return self.net(x)

class SectionFusionNet(nn.Module):
    def __init__(self, img_emb=128, sec1d_in=224, sec1d_emb=128):
        super().__init__()
        self.img_enc = ImgEncoder(img_emb)
        self.sec_enc = Sec1DEncoder(sec1d_in, sec1d_emb)
        self.head = nn.Sequential(
            nn.LayerNorm(img_emb + sec1d_emb),
            nn.Linear(img_emb + sec1d_emb, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, x_img, x_sec1d):
        e_img = self.img_enc(x_img)
        e_sec = self.sec_enc(x_sec1d)
        e = torch.cat([e_img, e_sec], dim=1)
        logit = self.head(e).squeeze(1)
        return logit, e_img, e_sec


# ----------------------------
# quick sanity eval (fast)
# ----------------------------
@torch.no_grad()
def eval_model(model, manifest_path, img_key, sec1d_key, device, max_batches=20, batch_size=1024):
    model.eval()
    ys, ps = [], []

    # evaluate only first shard for speed (enough to sanity-check training)
    for base, img_path, sec1d_path, y_path, v_path, _sha_path in iter_shards(manifest_path, img_key, sec1d_key):
        Ximg = np.load(img_path, mmap_mode="r")
        X1d  = np.load(sec1d_path, mmap_mode="r")
        y    = np.load(y_path, mmap_mode="r").astype(np.uint8)
        v    = np.load(v_path, mmap_mode="r").astype(np.uint8)

        idx = np.where(v == 1)[0]
        if idx.size == 0:
            continue

        idx = idx[: min(idx.size, max_batches * batch_size)]
        done = 0
        for s in range(0, idx.size, batch_size):
            j = idx[s:s+batch_size]
            bi = torch.from_numpy(_prep_img(Ximg[j])).to(device)
            b1 = torch.from_numpy(_prep_sec1d(X1d[j])).to(device)

            logit, _, _ = model(bi, b1)
            prob = torch.sigmoid(logit).detach().cpu().numpy()

            ys.append(y[j])
            ps.append(prob)

            done += 1
            if done >= max_batches:
                break
        break

    if not ys:
        return None

    y_all = np.concatenate(ys).astype(np.uint8)
    p_all = np.concatenate(ps).astype(np.float32)
    auc = float(roc_auc_score(y_all, p_all)) if len(np.unique(y_all)) > 1 else None
    acc = float(accuracy_score(y_all, (p_all >= 0.5).astype(np.uint8)))
    return {"auc": auc, "acc": acc, "n": int(y_all.size)}


# ----------------------------
# train
# ----------------------------
def train(args):
    ensure_dir(args.outdir)
    device = torch.device(args.device)

    sec1d_dim = infer_sec1d_dim(args.train_manifest, args.sec1d_key)
    print("Detected sec1d_dim:", sec1d_dim)

    model = SectionFusionNet(
        img_emb=args.img_emb_dim,
        sec1d_in=sec1d_dim,
        sec1d_emb=args.sec1d_emb_dim
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    save_json(os.path.join(args.outdir, "run_config.json"), {
        "cmd": "train",
        "dataset_tag": args.dataset_tag,
        "train_manifest": args.train_manifest,
        "test_manifest": args.test_manifest,
        "img_key": args.img_key,
        "sec1d_key": args.sec1d_key,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "amp": bool(args.amp),
        "img_emb_dim": args.img_emb_dim,
        "sec1d_emb_dim": args.sec1d_emb_dim,
        "shuffle_shards": bool(args.shuffle_shards),
        "max_rows_per_shard": args.max_rows_per_shard,
        "max_shards_per_epoch": args.max_shards_per_epoch,
        "created_at_utc": datetime.utcnow().isoformat() + "Z"
    })

    best_auc = -1.0
    best_path = os.path.join(args.outdir, "checkpoint_best.pt")
    last_path = os.path.join(args.outdir, "checkpoint_last.pt")
    log_path  = os.path.join(args.outdir, "train_log.jsonl")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        total_n = 0

        shards = list(iter_shards(args.train_manifest, args.img_key, args.sec1d_key))

        if args.shuffle_shards:
            rng = np.random.default_rng(args.seed + epoch)
            rng.shuffle(shards)

        shard_budget = args.max_shards_per_epoch if args.max_shards_per_epoch > 0 else None

        for base, img_path, sec1d_path, y_path, v_path, _sha_path in shards:
            Ximg = np.load(img_path, mmap_mode="r")
            X1d  = np.load(sec1d_path, mmap_mode="r")
            y    = np.load(y_path, mmap_mode="r").astype(np.uint8)
            v    = np.load(v_path, mmap_mode="r").astype(np.uint8)

            idx = np.where(v == 1)[0]
            if idx.size == 0:
                continue

            rng = np.random.default_rng(args.seed + epoch)
            rng.shuffle(idx)

            if args.max_rows_per_shard > 0:
                idx = idx[:args.max_rows_per_shard]

            for s in range(0, idx.size, args.batch_size):
                j = idx[s:s+args.batch_size]
                bi = torch.from_numpy(_prep_img(Ximg[j])).to(device)
                b1 = torch.from_numpy(_prep_sec1d(X1d[j])).to(device)
                by = torch.from_numpy(y[j].astype(np.float32)).to(device)

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    logit, _, _ = model(bi, b1)
                    loss = loss_fn(logit, by)

                if not torch.isfinite(loss):
                    # skip a broken batch instead of destroying training
                    continue

                scaler.scale(loss).backward()
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()

                total_loss += float(loss.detach().cpu().item()) * int(j.size)
                total_n += int(j.size)

            if shard_budget is not None:
                shard_budget -= 1
                if shard_budget <= 0:
                    break

        train_loss = total_loss / max(1, total_n)
        dt = time.time() - t0

        ev = eval_model(model, args.test_manifest, args.img_key, args.sec1d_key, device,
                        max_batches=args.eval_batches, batch_size=max(256, args.batch_size))
        auc = ev["auc"] if ev else None
        acc = ev["acc"] if ev else None

        rec = {"epoch": epoch, "train_loss": train_loss, "eval_auc": auc, "eval_acc": acc, "eval_n": (ev["n"] if ev else 0), "time_sec": dt}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        print(f"[epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} eval_auc={auc} eval_acc={acc} time={dt:.1f}s")

        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "sec1d_dim": sec1d_dim, "img_emb_dim": args.img_emb_dim, "sec1d_emb_dim": args.sec1d_emb_dim},
                   last_path)

        if auc is not None and auc > best_auc:
            best_auc = auc
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "sec1d_dim": sec1d_dim,
                "img_emb_dim": args.img_emb_dim,
                "sec1d_emb_dim": args.sec1d_emb_dim,
                "created_at_utc": datetime.utcnow().isoformat() + "Z",
            }, best_path)
            print("Saved best:", best_path)

    print("Done. Best eval AUC:", best_auc)
    print("Best checkpoint:", best_path)
    print("Logs:", log_path)


# ----------------------------
# export embeddings
# ----------------------------
@torch.no_grad()
def export_embeddings(args):
    ensure_dir(args.outdir)
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    sec1d_dim = int(ckpt["sec1d_dim"])
    img_emb_dim = int(ckpt["img_emb_dim"])
    sec1d_emb_dim = int(ckpt["sec1d_emb_dim"])

    model = SectionFusionNet(img_emb=img_emb_dim, sec1d_in=sec1d_dim, sec1d_emb=sec1d_emb_dim).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    for split_name, manifest_path in [("train", args.train_manifest), ("test", args.test_manifest)]:
        split_dir = os.path.join(args.outdir, split_name)
        ensure_dir(split_dir)

        for base, img_path, sec1d_path, y_path, v_path, sha_path in iter_shards(manifest_path, args.img_key, args.sec1d_key):
            Ximg = np.load(img_path, mmap_mode="r")
            X1d  = np.load(sec1d_path, mmap_mode="r")
            v    = np.load(v_path, mmap_mode="r").astype(np.uint8)
            y    = np.load(y_path, mmap_mode="r").astype(np.uint8)
            sha  = np.load(sha_path, mmap_mode="r")

            n = int(X1d.shape[0])
            prefix = f"{args.dataset_tag}__{base}"

            emb_img_path = os.path.join(split_dir, f"{prefix}__emb_section_img{img_emb_dim}.npy")
            emb_1d_path  = os.path.join(split_dir, f"{prefix}__emb_section_1d{sec1d_emb_dim}.npy")
            emb_cat_path = os.path.join(split_dir, f"{prefix}__emb_section_cat{img_emb_dim+sec1d_emb_dim}.npy")
            meta_path    = os.path.join(split_dir, f"{prefix}__emb_meta.json")

            Eimg = np.lib.format.open_memmap(emb_img_path, mode="w+", dtype=np.float32, shape=(n, img_emb_dim))
            E1d  = np.lib.format.open_memmap(emb_1d_path,  mode="w+", dtype=np.float32, shape=(n, sec1d_emb_dim))
            Ecat = np.lib.format.open_memmap(emb_cat_path, mode="w+", dtype=np.float32, shape=(n, img_emb_dim + sec1d_emb_dim))

            bs = args.batch_size
            for s in range(0, n, bs):
                j0 = s
                j1 = min(n, s + bs)

                vv = v[j0:j1]
                if vv.mean() == 0:
                    Eimg[j0:j1] = 0
                    E1d[j0:j1]  = 0
                    Ecat[j0:j1] = 0
                    continue

                bi = torch.from_numpy(_prep_img(Ximg[j0:j1])).to(device)
                b1 = torch.from_numpy(_prep_sec1d(X1d[j0:j1])).to(device)

                _logit, e_img, e_1d = model(bi, b1)
                e_img = e_img.detach().cpu().numpy().astype(np.float32)
                e_1d  = e_1d.detach().cpu().numpy().astype(np.float32)
                e_cat = np.concatenate([e_img, e_1d], axis=1)

                mask = (vv == 1)
                out_img = np.zeros((j1 - j0, img_emb_dim), dtype=np.float32)
                out_1d  = np.zeros((j1 - j0, sec1d_emb_dim), dtype=np.float32)
                out_cat = np.zeros((j1 - j0, img_emb_dim + sec1d_emb_dim), dtype=np.float32)

                out_img[mask] = e_img[mask]
                out_1d[mask]  = e_1d[mask]
                out_cat[mask] = e_cat[mask]

                Eimg[j0:j1] = out_img
                E1d[j0:j1]  = out_1d
                Ecat[j0:j1] = out_cat

            # alignment sanity
            assert sha.shape[0] == n and y.shape[0] == n and v.shape[0] == n

            save_json(meta_path, {
                "dataset_tag": args.dataset_tag,
                "split": split_name,
                "base": base,
                "img_emb_dim": img_emb_dim,
                "sec1d_emb_dim": sec1d_emb_dim,
                "cat_dim": img_emb_dim + sec1d_emb_dim,
                "created_at_utc": datetime.utcnow().isoformat() + "Z",
                "inputs": {"img": img_path, "sec1d": sec1d_path, "y": y_path, "valid": v_path, "sha256": sha_path},
                "outputs": {"emb_img": emb_img_path, "emb_1d": emb_1d_path, "emb_cat": emb_cat_path},
            })

            print(f"[{split_name}] exported: {base} n={n}")

    print("Done exporting embeddings to:", args.outdir)


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--train_manifest", required=True)
    tr.add_argument("--test_manifest", required=True)
    tr.add_argument("--dataset_tag", required=True)
    tr.add_argument("--outdir", required=True)

    tr.add_argument("--img_key", default="X_section_img_v1")
    tr.add_argument("--sec1d_key", default="X_section_1d_emberv3_section224")

    tr.add_argument("--device", default="cuda")
    tr.add_argument("--epochs", type=int, default=3)
    tr.add_argument("--batch_size", type=int, default=512)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--weight_decay", type=float, default=1e-2)
    tr.add_argument("--grad_clip", type=float, default=1.0)
    tr.add_argument("--amp", action="store_true")  # keep OFF unless stable
    tr.add_argument("--img_emb_dim", type=int, default=128)
    tr.add_argument("--sec1d_emb_dim", type=int, default=128)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--shuffle_shards", action="store_true")
    tr.add_argument("--max_rows_per_shard", type=int, default=0)
    tr.add_argument("--max_shards_per_epoch", type=int, default=0)
    tr.add_argument("--eval_batches", type=int, default=20)

    ex = sub.add_parser("export")
    ex.add_argument("--train_manifest", required=True)
    ex.add_argument("--test_manifest", required=True)
    ex.add_argument("--checkpoint", required=True)
    ex.add_argument("--dataset_tag", required=True)
    ex.add_argument("--outdir", required=True)

    ex.add_argument("--img_key", default="X_section_img_v1")
    ex.add_argument("--sec1d_key", default="X_section_1d_emberv3_section224")
    ex.add_argument("--device", default="cuda")
    ex.add_argument("--batch_size", type=int, default=1024)

    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
    else:
        export_embeddings(args)

if __name__ == "__main__":
    main()