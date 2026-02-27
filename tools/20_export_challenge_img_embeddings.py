#!/usr/bin/env python3
import argparse, json, os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

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
        x = self.net(x).squeeze(-1).squeeze(-1)
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
        return logit, e_img, e_sec

def prep_img(batch: np.ndarray) -> np.ndarray:
    b = np.asarray(batch)
    if b.ndim == 3:  # (N,H,W)
        b = b[:, None, :, :]
    b = b.astype(np.float32, copy=False)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    if b.max() > 1.5:
        b = b / 255.0
    b = np.clip(b, 0.0, 1.0)
    return b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section_manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset_tag", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=1024)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    out_split_dir = os.path.join(args.outdir, "challenge_malicious")
    ensure_dir(out_split_dir)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    sec1d_dim = int(ckpt.get("sec1d_dim", 224))
    img_emb_dim = int(ckpt.get("img_emb_dim", 128))
    sec1d_emb_dim = int(ckpt.get("sec1d_emb_dim", 128))

    model = SectionFusionNet(img_emb=img_emb_dim, sec1d_in=sec1d_dim, sec1d_emb=sec1d_emb_dim).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    mf = load_json(args.section_manifest)
    items = mf["items"]

    for it in items:
        base = it["base"]
        Ximg_path = it["X_section_img_v1"]
        v_path = it["valid"]
        n = int(np.load(v_path, mmap_mode="r").shape[0])

        out_prefix = f"{args.dataset_tag}__{base}"
        emb_path = os.path.join(out_split_dir, f"{out_prefix}__emb_section_img{img_emb_dim}.npy")
        meta_path = os.path.join(out_split_dir, f"{out_prefix}__emb_meta.json")

        E = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float32, shape=(n, img_emb_dim))

        Ximg = np.load(Ximg_path, mmap_mode="r")
        v = np.load(v_path, mmap_mode="r").astype(np.uint8)

        with torch.no_grad():
            for s in range(0, n, args.batch_size):
                j = slice(s, min(n, s + args.batch_size))
                vv = v[j]
                if vv.mean() == 0:
                    E[j] = 0
                    continue

                bi = torch.from_numpy(prep_img(Ximg[j])).to(device)
                # dummy sec1d (not needed for e_img)
                dummy = torch.zeros((bi.shape[0], sec1d_dim), dtype=torch.float32, device=device)

                _, e_img, _ = model(bi, dummy)
                e_img = e_img.detach().cpu().numpy().astype(np.float32)

                out = np.zeros((len(vv), img_emb_dim), dtype=np.float32)
                mask = (vv == 1)
                out[mask] = e_img[mask]
                E[j] = out

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "dataset_tag": args.dataset_tag,
                "base": base,
                "split": "CHALLENGE_MALICIOUS",
                "img_emb_dim": img_emb_dim,
                "created_at_utc": datetime.utcnow().isoformat() + "Z",
                "source": {"X_section_img_v1": Ximg_path, "valid": v_path},
                "output": {"emb_section_img": emb_path}
            }, f, indent=2)

        print(f"[CHALLENGE EMB] {base}: wrote {emb_path}")

    out_manifest = os.path.join(args.outdir, f"{args.dataset_tag}__challenge_malicious_imgemb_manifest.json")
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump({
            "dataset_tag": args.dataset_tag,
            "split": "CHALLENGE_MALICIOUS",
            "emb_kind": "emb_section_img",
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
            "items": items
        }, f, indent=2)

    print("Wrote:", out_manifest)

if __name__ == "__main__":
    main()