#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score


# -------------------------
# Utils
# -------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_json(path: str, obj: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def npy_mmap(path: str):
    return np.load(path, mmap_mode="r")

def infer_img_tensor(x_img_np: np.ndarray) -> torch.Tensor:
    """
    Convert numpy image array to torch float tensor [B, C, H, W] in [0,1].
    Accepts shapes:
      - [B,H,W] (grayscale)
      - [B,1,H,W]
      - [B,H,W,1]
      - [B,H,W,3]
      - [B,C,H,W]
    If it’s not an image (ndim==2), treat it as vector and return [B, D].
    """
    if x_img_np.ndim == 2:
        # [B, D] embedding-like
        t = torch.from_numpy(np.asarray(x_img_np)).float()
        return t

    if x_img_np.ndim == 3:
        # [B,H,W] -> [B,1,H,W]
        t = torch.from_numpy(np.asarray(x_img_np))
        t = t.unsqueeze(1)
    elif x_img_np.ndim == 4:
        t = torch.from_numpy(np.asarray(x_img_np))
        # [B,H,W,C] -> [B,C,H,W]
        if t.shape[-1] in (1, 3) and t.shape[1] not in (1, 3):
            t = t.permute(0, 3, 1, 2)
    else:
        raise RuntimeError(f"Unexpected X_img ndim: {x_img_np.ndim}")

    if t.dtype == torch.uint8:
        t = t.float() / 255.0
    else:
        t = t.float()
        # if already 0..255 float, normalize roughly
        if t.max() > 1.5:
            t = t / 255.0
    return t

def infer_1d_tensor(x_1d_np: np.ndarray) -> torch.Tensor:
    """
    Convert numpy 1D section array to torch float tensor.
    Accepts:
      - [B,L] or [B,D]
      - [B,C,L]
    """
    t = torch.from_numpy(np.asarray(x_1d_np)).float()
    return t


# -------------------------
# Models
# -------------------------
class TabMLP(nn.Module):
    def __init__(self, in_dim: int = 2568, emb_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, emb_dim),
            nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)

class VecMLP(nn.Module):
    """For cases where X_img is actually [B,D] not an image."""
    def __init__(self, in_dim: int, emb_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, emb_dim),
            nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)

class SmallCNN2D(nn.Module):
    def __init__(self, in_ch: int = 1, emb_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, emb_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class Sec1DEncoder(nn.Module):
    """
    If input is [B,L] -> MLP
    If input is [B,C,L] -> Conv1D + pooling -> emb
    """
    def __init__(self, emb_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.dropout = dropout
        # will be created lazily based on first batch shape
        self._mode = None
        self.mlp = None
        self.conv = None
        self.fc = None

    def _build_mlp(self, in_dim: int):
        self._mode = "mlp"
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, self.emb_dim),
            nn.GELU(),
        )

    def _build_conv(self, in_ch: int):
        self._mode = "conv"
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 64, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, self.emb_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self._mode is None:
            if x.ndim == 2:
                self._build_mlp(x.shape[1])
            elif x.ndim == 3:
                self._build_conv(x.shape[1])
            else:
                raise RuntimeError(f"Unexpected sec1d ndim: {x.ndim}")
            self.to(x.device)

        if self._mode == "mlp":
            return self.mlp(x)
        else:
            return self.fc(self.conv(x))

class FusionNet(nn.Module):
    def __init__(self, tab_dim: int = 2568, tab_emb: int = 256, img_emb: int = 256, sec1d_emb: int = 128, dropout: float = 0.2):
        super().__init__()
        self.tab = TabMLP(tab_dim, tab_emb, dropout=dropout)

        # image encoder created lazily (because image may be vector OR image with unknown channels)
        self.img_is_vector = False
        self.img_vec = None
        self.img_cnn = None
        self.img_emb = img_emb

        self.sec1d = Sec1DEncoder(sec1d_emb, dropout=dropout)

        fusion_in = tab_emb + img_emb + sec1d_emb
        self.head = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def _init_img_encoder(self, x_img: torch.Tensor):
        # x_img either [B,D] or [B,C,H,W]
        if x_img.ndim == 2:
            self.img_is_vector = True
            self.img_vec = VecMLP(in_dim=x_img.shape[1], emb_dim=self.img_emb)
        elif x_img.ndim == 4:
            self.img_is_vector = False
            self.img_cnn = SmallCNN2D(in_ch=x_img.shape[1], emb_dim=self.img_emb)
        else:
            raise RuntimeError(f"Unexpected x_img ndim for image encoder: {x_img.ndim}")
        self.to(x_img.device)

    def forward(self, x_tab, x_img, x_1d):
        t = self.tab(x_tab)

        if self.img_vec is None and self.img_cnn is None:
            self._init_img_encoder(x_img)

        if self.img_is_vector:
            i = self.img_vec(x_img)
        else:
            i = self.img_cnn(x_img)

        s = self.sec1d(x_1d)

        z = torch.cat([t, i, s], dim=1)
        logit = self.head(z).squeeze(1)
        return logit

class TabOnlyNet(nn.Module):
    def __init__(self, tab_dim: int = 2568, dropout: float = 0.2):
        super().__init__()
        self.tab = TabMLP(tab_dim, emb_dim=256, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
    def forward(self, x_tab):
        z = self.tab(x_tab)
        return self.head(z).squeeze(1)


# -------------------------
# Data access (sharded)
# -------------------------
@dataclass
class ShardPaths:
    base: str
    X_tab: str
    y: str
    valid: str
    sha256: str
    X_img: Optional[str]
    X_1d: Optional[str]

def parse_multimodal_manifest(path: str, img_key: str, sec1d_key: str) -> List[ShardPaths]:
    m = load_json(path)
    items = m["paired_items"]
    out: List[ShardPaths] = []
    for it in items:
        base = it["base"]
        tab = it["tab"]
        sec = it["section"]
        out.append(ShardPaths(
            base=base,
            X_tab=tab["X"],
            y=tab["y"],
            valid=tab["valid"],
            sha256=tab["sha256"],
            X_img=sec.get(img_key, None),
            X_1d=sec.get(sec1d_key, None),
        ))
    return out

def iter_batches_indices(n: int, batch_size: int, shuffle: bool, rng: np.random.Generator):
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start:start+batch_size]

@torch.no_grad()
def evaluate(model, shards: List[ShardPaths], device: torch.device, model_kind: str,
             img_key: str, sec1d_key: str, batch_size: int, max_rows_total: int = 0, amp: bool = True) -> Dict[str, Any]:
    model.eval()
    y_true_all = []
    y_prob_all = []

    seen = 0
    for sh in shards:
        y = npy_mmap(sh.y)
        v = npy_mmap(sh.valid)
        X_tab = npy_mmap(sh.X_tab)

        X_img = None
        X_1d = None
        if model_kind == "fusion":
            if not sh.X_img or not sh.X_1d:
                raise RuntimeError(f"Missing X_img or X_1d in shard {sh.base}. Check keys: {img_key}, {sec1d_key}")
            X_img = npy_mmap(sh.X_img)
            X_1d  = npy_mmap(sh.X_1d)

        n = y.shape[0]
        # valid mask (should be all 1)
        valid_mask = (np.asarray(v) == 1)
        idx_all = np.where(valid_mask)[0]

        # optional cap
        if max_rows_total and seen >= max_rows_total:
            break

        # batch loop
        for ixs in iter_batches_indices(len(idx_all), batch_size, shuffle=False, rng=np.random.default_rng(0)):
            rows = idx_all[ixs]
            if max_rows_total and seen >= max_rows_total:
                break

            xb_tab = torch.from_numpy(np.asarray(X_tab[rows])).float().to(device, non_blocking=True)
            yb = torch.from_numpy(np.asarray(y[rows])).float().to(device, non_blocking=True)

            if model_kind == "tab_only":
                with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                    logit = model(xb_tab)
            else:
                xb_img = infer_img_tensor(np.asarray(X_img[rows])).to(device, non_blocking=True)
                xb_1d  = infer_1d_tensor(np.asarray(X_1d[rows])).to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                    logit = model(xb_tab, xb_img, xb_1d)

            prob = torch.sigmoid(logit).detach().cpu().numpy()
            y_true_all.append(yb.detach().cpu().numpy())
            y_prob_all.append(prob)

            seen += rows.shape[0]

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=np.float32)
    y_prob = np.concatenate(y_prob_all) if y_prob_all else np.array([], dtype=np.float32)

    if y_true.size == 0:
        return {"n_eval": 0}

    y_pred = (y_prob >= 0.5).astype(np.int32)
    acc = float((y_pred == y_true.astype(np.int32)).mean())
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = None

    return {"n_eval": int(y_true.size), "acc": acc, "roc_auc": auc}

def train_one_epoch(model, opt, shards: List[ShardPaths], device: torch.device, model_kind: str,
                    img_key: str, sec1d_key: str, batch_size: int,
                    shuffle: bool, amp: bool, scaler: Optional[torch.cuda.amp.GradScaler],
                    max_rows_total: int, seed: int) -> Dict[str, Any]:
    model.train()
    rng = np.random.default_rng(seed)

    total_loss = 0.0
    total_seen = 0
    total_correct = 0

    bce = nn.BCEWithLogitsLoss()

    for sh in shards:
        y = npy_mmap(sh.y)
        v = npy_mmap(sh.valid)
        X_tab = npy_mmap(sh.X_tab)

        X_img = None
        X_1d = None
        if model_kind == "fusion":
            if not sh.X_img or not sh.X_1d:
                raise RuntimeError(f"Missing X_img or X_1d in shard {sh.base}. Check keys: {img_key}, {sec1d_key}")
            X_img = npy_mmap(sh.X_img)
            X_1d  = npy_mmap(sh.X_1d)

        valid_mask = (np.asarray(v) == 1)
        idx_all = np.where(valid_mask)[0]
        n_valid = idx_all.shape[0]

        # optional cap
        if max_rows_total and total_seen >= max_rows_total:
            break

        for ixs in iter_batches_indices(n_valid, batch_size, shuffle=shuffle, rng=rng):
            rows = idx_all[ixs]
            if max_rows_total and total_seen >= max_rows_total:
                break

            xb_tab = torch.from_numpy(np.asarray(X_tab[rows])).float().to(device, non_blocking=True)
            yb = torch.from_numpy(np.asarray(y[rows])).float().to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            if model_kind == "tab_only":
                with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                    logit = model(xb_tab)
                    loss = bce(logit, yb)
            else:
                xb_img = infer_img_tensor(np.asarray(X_img[rows])).to(device, non_blocking=True)
                xb_1d  = infer_1d_tensor(np.asarray(X_1d[rows])).to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                    logit = model(xb_tab, xb_img, xb_1d)
                    loss = bce(logit, yb)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            with torch.no_grad():
                prob = torch.sigmoid(logit)
                pred = (prob >= 0.5).float()
                correct = (pred == yb).sum().item()

            total_loss += float(loss.item()) * rows.shape[0]
            total_correct += int(correct)
            total_seen += int(rows.shape[0])

    avg_loss = total_loss / max(1, total_seen)
    acc = total_correct / max(1, total_seen)
    return {"n_train": total_seen, "loss": float(avg_loss), "acc": float(acc)}


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--test_manifest", required=True)

    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dataset_tag", required=True)

    ap.add_argument("--model", choices=["tab_only", "fusion"], default="fusion")

    ap.add_argument("--img_key", default="X_section_img_v1")
    ap.add_argument("--sec1d_key", default="X_section_1d_emberv3_section224")

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--max_shards_train", type=int, default=0, help="debug: use first N train shards")
    ap.add_argument("--max_shards_test", type=int, default=0, help="debug: use first N test shards")
    ap.add_argument("--max_rows_train", type=int, default=0, help="debug: cap total train rows per epoch")
    ap.add_argument("--max_rows_eval", type=int, default=0, help="debug: cap total eval rows")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    seed_everything(args.seed)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    train_shards = parse_multimodal_manifest(args.train_manifest, args.img_key, args.sec1d_key)
    test_shards  = parse_multimodal_manifest(args.test_manifest,  args.img_key, args.sec1d_key)

    if args.max_shards_train:
        train_shards = train_shards[:args.max_shards_train]
    if args.max_shards_test:
        test_shards = test_shards[:args.max_shards_test]

    # Build model
    if args.model == "tab_only":
        model = TabOnlyNet(tab_dim=2568, dropout=args.dropout)
    else:
        model = FusionNet(tab_dim=2568, dropout=args.dropout)

    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    # Save run config
    save_json(os.path.join(args.outdir, "run_config.json"), vars(args))

    best_auc = -1.0
    log_path = os.path.join(args.outdir, "train_log.jsonl")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr = train_one_epoch(
            model=model,
            opt=opt,
            shards=train_shards,
            device=device,
            model_kind=args.model,
            img_key=args.img_key,
            sec1d_key=args.sec1d_key,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            amp=args.amp,
            scaler=scaler if (args.amp and device.type == "cuda") else None,
            max_rows_total=args.max_rows_train,
            seed=args.seed + epoch,
        )

        ev = evaluate(
            model=model,
            shards=test_shards,
            device=device,
            model_kind=args.model,
            img_key=args.img_key,
            sec1d_key=args.sec1d_key,
            batch_size=args.batch_size,
            max_rows_total=args.max_rows_eval,
            amp=args.amp,
        )

        dt = time.time() - t0

        row = {
            "dataset_tag": args.dataset_tag,
            "epoch": epoch,
            "train": tr,
            "test": ev,
            "time_sec": dt,
        }

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        print(f"[epoch {epoch}/{args.epochs}] "
              f"train_loss={tr.get('loss'):.4f} train_acc={tr.get('acc'):.4f} "
              f"test_acc={ev.get('acc')} test_auc={ev.get('roc_auc')} time={dt:.1f}s")

        # checkpoint
        torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict()},
                   os.path.join(args.outdir, "checkpoint_last.pt"))

        auc = ev.get("roc_auc")
        if auc is not None and auc > best_auc:
            best_auc = auc
            torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict()},
                       os.path.join(args.outdir, "checkpoint_best.pt"))

    save_json(os.path.join(args.outdir, "summary.json"), {"best_test_auc": best_auc, "log": log_path})
    print("Done. Best test AUC:", best_auc)
    print("Logs:", log_path)
    print("Best checkpoint:", os.path.join(args.outdir, "checkpoint_best.pt"))


if __name__ == "__main__":
    main()
