#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, Any, List, Tuple

import numpy as np


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def index_items_by_base(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    items = manifest.get("items", [])
    out = {}
    for it in items:
        base = it.get("base")
        if not base:
            # fallback: derive from meta filename if needed
            meta = it.get("meta", "")
            base = os.path.basename(meta).replace("__meta.json", "")
        out[base] = it
    return out


def load_sha(path: str) -> np.ndarray:
    # sha arrays are typically dtype='S64' memmaps saved as .npy
    return np.load(path, mmap_mode="r")


def sha_check(tab_sha_path: str, sec_sha_path: str, sample_n: int = 256) -> Tuple[bool, str]:
    a = load_sha(tab_sha_path)
    b = load_sha(sec_sha_path)
    if a.shape != b.shape:
        return False, f"shape mismatch {a.shape} vs {b.shape}"
    n = min(sample_n, a.shape[0])
    if n <= 0:
        return True, "empty"
    ok = (a[:n] == b[:n]).all()
    return bool(ok), f"checked_first_n={n}"


def build_split(
    split: str,
    tab_manifest_path: str,
    sec_manifest_path: str,
    outdir: str,
    dataset_tag: str,
    do_sha_check: bool,
    sha_check_n: int,
) -> str:
    tab_m = read_json(tab_manifest_path)
    sec_m = read_json(sec_manifest_path)

    tab_by_base = index_items_by_base(tab_m)
    sec_by_base = index_items_by_base(sec_m)

    bases = sorted(set(tab_by_base.keys()) & set(sec_by_base.keys()))
    combined_items = []
    skipped = []

    for base in bases:
        t = tab_by_base[base]
        s = sec_by_base[base]

        # expected keys from your sharded writer:
        # tab: "sha256", "X", "y", "valid"
        tab_sha = t.get("sha256") or t.get("sha") or ""
        sec_sha = s.get("sha256") or s.get("sha") or ""

        if do_sha_check and tab_sha and sec_sha:
            ok, msg = sha_check(tab_sha, sec_sha, sample_n=sha_check_n)
            if not ok:
                skipped.append({"base": base, "reason": f"sha_check_failed: {msg}", "tab_sha": tab_sha, "sec_sha": sec_sha})
                continue

        combined_items.append({
            "base": base,
            "tab": t,
            "section": s
        })

    out = {
        "dataset_tag": dataset_tag,
        "split": split,
        "n_paired": len(combined_items),
        "n_skipped": len(skipped),
        "paired_items": combined_items,
        "skipped": skipped,
    }

    out_path = os.path.join(outdir, f"{dataset_tag}__multimodal_manifest__{split.lower()}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[{split}] paired={len(combined_items)} skipped={len(skipped)}")
    print("Wrote:", out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_tag", required=True)
    ap.add_argument("--tab_dir", required=True, help="outputs/01_feature_vectors/... root (where manifest_train.json exists)")
    ap.add_argument("--section_dir", required=True, help="outputs/02_section_modalities/... root (where manifest_train.json exists)")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--sha_check", action="store_true", help="Verify sha256 alignment per shard (fast sample check)")
    ap.add_argument("--sha_check_n", type=int, default=256, help="How many rows to compare per shard")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    tab_train = os.path.join(args.tab_dir, "manifest_train.json")
    tab_test  = os.path.join(args.tab_dir, "manifest_test.json")
    sec_train = os.path.join(args.section_dir, "manifest_train.json")
    sec_test  = os.path.join(args.section_dir, "manifest_test.json")

    for p in [tab_train, tab_test, sec_train, sec_test]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing manifest: {p}")

    build_split("TRAIN", tab_train, sec_train, args.outdir, args.dataset_tag, args.sha_check, args.sha_check_n)
    build_split("TEST",  tab_test,  sec_test,  args.outdir, args.dataset_tag, args.sha_check, args.sha_check_n)


if __name__ == "__main__":
    main()
