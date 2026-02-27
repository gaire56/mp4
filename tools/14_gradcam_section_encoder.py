#!/usr/bin/env python3
import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


# -------- model (must match tools/09_train_and_export_section_embeddings.py) --------
class ImgEncoder(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(64, emb_dim)

    def forward(self, x):
        x = self.net(x).squeeze(-1).squeeze(-1)  # (B,64)
        return self.fc(x)

class Sec1DEncoder(nn.Module):
    def __init__(self, in_dim=224, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
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
            nn.Linear(img_emb + sec1d_emb, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x_img, x_sec1d):
        e_img = self.img_enc(x_img)
        e_sec = self.sec_enc(x_sec1d)
        e = torch.cat([e_img, e_sec], dim=1)
        logit = self.head(e).squeeze(1)
        return logit


# -------- helpers --------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_json(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def npy_mmap(path: str):
    return np.load(path, mmap_mode="r")

def prep_img(x: np.ndarray) -> np.ndarray:
    # expects (H,W) or (1,H,W) or (H,W,1)
    x = np.asarray(x)
    if x.ndim == 2:
        x = x[None, :, :]  # (1,H,W)
    elif x.ndim == 3 and x.shape[-1] == 1:
        x = np.transpose(x, (2, 0, 1))  # (1,H,W)

    x = x.astype(np.float32, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.max() > 1.5:
        x = x / 255.0
    x = np.clip(x, 0.0, 1.0)
    return x

def prep_sec1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).astype(np.float32, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, -1e3, 1e3)
    return x

def section_img_layout_info() -> Dict[str, Any]:
    # must match tools/02_build_section_modalities.py SectionImageFeaturizer layout
    sec_names = [".text",".rdata",".data",".rsrc",".reloc",".idata",".edata",".pdata",".tls",".bss","OTHER"]
    props = ["CNT_CODE","CNT_INITIALIZED_DATA","CNT_UNINITIALIZED_DATA","MEM_EXECUTE","MEM_READ","MEM_WRITE","MEM_DISCARDABLE","MEM_SHARED"]
    cols = [
        "log1p(size)","log1p(vsize)","entropy","size_ratio","vsize_ratio","is_named",
        *[f"name:{n}" for n in sec_names],
        *[f"prop:{p}" for p in props],
    ]
    return {
        "rows": [
            "section_row_0..15 (up to 16 sections, padded)",
            "overlay_row_16 (last row)"
        ],
        "cols": cols,
        "expected_cols": 24,
        "expected_rows": 17,
    }

def pick_examples(manifest_path: str, img_key: str, sec1d_key: str, n_per_class: int, seed: int):
    """
    Picks examples from TEST manifest:
      returns list of dicts: {base, idx, label, sha256, img_path, sec1d_path, y_path, valid_path, sha_path}
    """
    rng = np.random.default_rng(seed)
    mf = load_json(manifest_path)
    items = mf["paired_items"]

    picked_0 = []
    picked_1 = []

    for it in items:
        base = it["base"]
        tab = it["tab"]
        sec = it["section"]

        img_path = sec[img_key]
        sec1d_path = sec[sec1d_key]
        y_path = tab["y"]
        v_path = tab["valid"]
        sha_path = tab["sha256"]

        y = npy_mmap(y_path).astype(np.uint8)
        v = npy_mmap(v_path).astype(np.uint8)
        valid_idx = np.where(v == 1)[0]
        if valid_idx.size == 0:
            continue

        # shuffle candidates
        cand = valid_idx.copy()
        rng.shuffle(cand)

        for idx in cand:
            lab = int(y[idx])
            if lab == 0 and len(picked_0) < n_per_class:
                picked_0.append((base, int(idx), lab, img_path, sec1d_path, y_path, v_path, sha_path))
            elif lab == 1 and len(picked_1) < n_per_class:
                picked_1.append((base, int(idx), lab, img_path, sec1d_path, y_path, v_path, sha_path))
            if len(picked_0) >= n_per_class and len(picked_1) >= n_per_class:
                break

        if len(picked_0) >= n_per_class and len(picked_1) >= n_per_class:
            break

    out = []
    for t in (picked_0 + picked_1):
        base, idx, lab, img_path, sec1d_path, y_path, v_path, sha_path = t
        sha = npy_mmap(sha_path)[idx].tobytes().decode("utf-8", errors="ignore").strip("\x00")
        out.append({
            "base": base,
            "idx": idx,
            "label": lab,
            "sha256": sha,
            "img_path": img_path,
            "sec1d_path": sec1d_path,
            "y_path": y_path,
            "valid_path": v_path,
            "sha_path": sha_path,
        })

    return out


# -------- Grad-CAM core --------
class GradCAM:
    def __init__(self, model: SectionFusionNet):
        self.model = model
        self.activations = None
        self.gradients = None
        self.hook_handles = []

        # Hook the feature map BEFORE AdaptiveAvgPool2d
        # img_enc.net = [Conv,ReLU,Pool, Conv,ReLU,Pool, Conv,ReLU, AdaptiveAvgPool]
        target_layer = self.model.img_enc.net[7]  # ReLU after last conv -> output feature map (B,64,H,W)

        def fwd_hook(module, inp, out):
            self.activations = out

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.hook_handles.append(target_layer.register_forward_hook(fwd_hook))
        self.hook_handles.append(target_layer.register_full_backward_hook(bwd_hook))

    def close(self):
        for h in self.hook_handles:
            h.remove()

    def __call__(self, x_img: torch.Tensor, x_1d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: prob, cam_up (H,W) aligned to input H,W
        """
        self.model.zero_grad(set_to_none=True)

        logit = self.model(x_img, x_1d)  # (B,)
        prob = torch.sigmoid(logit)

        # target: maximize predicted class score (logit)
        score = logit.sum()
        score.backward()

        acts = self.activations          # (B,C,h,w)
        grads = self.gradients           # (B,C,h,w)

        # weights: GAP over spatial dims
        w = grads.mean(dim=(2,3), keepdim=True)  # (B,C,1,1)
        cam = (w * acts).sum(dim=1, keepdim=True)  # (B,1,h,w)
        cam = F.relu(cam)

        # normalize each sample
        cam_min = cam.amin(dim=(2,3), keepdim=True)
        cam_max = cam.amax(dim=(2,3), keepdim=True)
        cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # upsample to input size
        H, W = x_img.shape[-2], x_img.shape[-1]
        cam_up = F.interpolate(cam_norm, size=(H,W), mode="bilinear", align_corners=False)  # (B,1,H,W)
        return prob.detach(), cam_up.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--img_key", default="X_section_img_v1")
    ap.add_argument("--sec1d_key", default="X_section_1d_emberv3_section224")

    ap.add_argument("--n_per_class", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    sec1d_dim = int(ckpt.get("sec1d_dim", 224))
    img_emb_dim = int(ckpt.get("img_emb_dim", 128))
    sec1d_emb_dim = int(ckpt.get("sec1d_emb_dim", 128))

    model = SectionFusionNet(img_emb=img_emb_dim, sec1d_in=sec1d_dim, sec1d_emb=sec1d_emb_dim).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    cammer = GradCAM(model)

    examples = pick_examples(args.test_manifest, args.img_key, args.sec1d_key, args.n_per_class, args.seed)
    layout = section_img_layout_info()

    meta_all = {
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "checkpoint": args.checkpoint,
        "device": str(device),
        "img_key": args.img_key,
        "sec1d_key": args.sec1d_key,
        "layout": layout,
        "n_examples": len(examples),
        "examples": [],
    }

    for ex in examples:
        base = ex["base"]
        idx = ex["idx"]
        lab = ex["label"]
        sha = ex["sha256"]

        Ximg = npy_mmap(ex["img_path"])
        X1d  = npy_mmap(ex["sec1d_path"])

        img = prep_img(Ximg[idx])             # (1,H,W)
        sec1d = prep_sec1d(X1d[idx])          # (D,)
        H, W = img.shape[1], img.shape[2]

        x_img = torch.from_numpy(img[None, ...]).to(device)      # (B=1,1,H,W)
        x_1d  = torch.from_numpy(sec1d[None, ...]).to(device)    # (B=1,D)

        prob, cam_up = cammer(x_img, x_1d)
        p = float(prob.cpu().numpy()[0])
        cam = cam_up.cpu().numpy()[0,0]  # (H,W)

        # Save arrays
        out_prefix = f"{base}__idx{idx}__y{lab}__{sha[:12]}"
        npz_path = os.path.join(args.outdir, f"{out_prefix}__gradcam.npz")
        np.savez_compressed(npz_path, img=img[0], cam=cam, prob=p, label=lab, sha256=sha, base=base, idx=idx)

        # Plot heatmap
        png_path = os.path.join(args.outdir, f"{out_prefix}__gradcam.png")
        plt.figure(figsize=(8, 5))
        plt.imshow(cam, aspect="auto")
        plt.title(f"Grad-CAM (section image) | y={lab} prob={p:.4f} | {base} | {sha[:12]}")
        plt.xlabel("feature columns (24)")
        plt.ylabel("rows (sections+overlay)")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()

        meta_all["examples"].append({
            "base": base,
            "idx": idx,
            "label": lab,
            "sha256": sha,
            "prob": p,
            "npz": npz_path,
            "png": png_path,
        })

        print("Saved:", png_path)

    cammer.close()

    meta_path = os.path.join(args.outdir, "gradcam_run_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_all, f, indent=2)

    print("Wrote:", meta_path)


if __name__ == "__main__":
    main()
