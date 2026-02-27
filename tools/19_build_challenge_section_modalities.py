#!/usr/bin/env python3
import argparse
import json
import os
import time
import subprocess
from datetime import datetime
from glob import glob
from typing import Dict, Any, List, Tuple

import numpy as np
from numpy.lib.format import open_memmap
import importlib.util


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def wc_l(path: str) -> int:
    try:
        out = subprocess.check_output(["wc", "-l", path], text=True).strip()
        return int(out.split()[0])
    except Exception:
        n = 0
        with open(path, "rb") as f:
            for _ in f:
                n += 1
        return n

def list_files(challenge_dir: str, pattern: str) -> List[str]:
    files = sorted(glob(os.path.join(challenge_dir, pattern)))
    files = [p for p in files if p.endswith(".jsonl") and os.path.isfile(p)]
    if not files:
        raise RuntimeError(f"No files found: {challenge_dir}/{pattern}")
    return files

def dynamic_import_vectorizer(tools_dir: str):
    mod_path = os.path.join(tools_dir, "01_build_feature_vectors_emberv3.py")
    if not os.path.exists(mod_path):
        raise FileNotFoundError(f"Cannot find: {mod_path}")
    spec = importlib.util.spec_from_file_location("emberv3_vec", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if not hasattr(mod, "EmberV3Vectorizer"):
        raise RuntimeError("EmberV3Vectorizer not found in tools/01_build_feature_vectors_emberv3.py")
    return mod.EmberV3Vectorizer

def section_to_image(raw: Dict[str, Any]) -> np.ndarray:
    """
    Build section-structure image: shape (17, 24)
    - 16 rows for sections (padded/truncated) + 1 row for overlay
    - 24 columns = 6 numeric + 10 name one-hot + 8 props one-hot
    """
    sec = raw.get("section", {}) or {}
    sections = sec.get("sections", []) or []
    overlay = sec.get("overlay", {}) or {}

    # 10 common section names + OTHER (10 total)
    name_vocab = [".text",".rdata",".data",".rsrc",".reloc",".idata",".edata",".pdata",".tls","OTHER"]
    name_to_i = {n: i for i, n in enumerate(name_vocab)}

    # 8 properties (flags)
    props_vocab = [
        "CNT_CODE",
        "CNT_INITIALIZED_DATA",
        "CNT_UNINITIALIZED_DATA",
        "MEM_EXECUTE",
        "MEM_READ",
        "MEM_WRITE",
        "MEM_DISCARDABLE",
        "MEM_SHARED",
    ]
    prop_to_i = {p: i for i, p in enumerate(props_vocab)}

    H, W = 17, 24
    img = np.zeros((H, W), dtype=np.float32)

    def encode_row(row_i: int, name: str, size: float, vsize: float, entropy: float,
                   size_ratio: float, vsize_ratio: float, props: List[str]):
        # numeric 6
        img[row_i, 0] = np.log1p(max(0.0, float(size)))
        img[row_i, 1] = np.log1p(max(0.0, float(vsize)))
        img[row_i, 2] = float(entropy)
        img[row_i, 3] = float(size_ratio)
        img[row_i, 4] = float(vsize_ratio)
        img[row_i, 5] = 1.0 if (name and name.strip()) else 0.0

        # name one-hot (10)
        nm = (name or "").strip()
        nm_i = name_to_i.get(nm, name_to_i["OTHER"])
        img[row_i, 6 + nm_i] = 1.0

        # props one-hot (8)
        for p in props or []:
            if p in prop_to_i:
                img[row_i, 16 + prop_to_i[p]] = 1.0

    # section rows (0..15)
    for i in range(min(16, len(sections))):
        s = sections[i] or {}
        encode_row(
            row_i=i,
            name=str(s.get("name","") or ""),
            size=float(s.get("size",0) or 0),
            vsize=float(s.get("vsize",0) or 0),
            entropy=float(s.get("entropy",0) or 0),
            size_ratio=float(s.get("size_ratio",0) or 0),
            vsize_ratio=float(s.get("vsize_ratio",0) or 0),
            props=list(s.get("props",[]) or []),
        )

    # overlay row (row 16): only numeric features, no name/props
    ov_size = float(overlay.get("size",0) or 0)
    ov_entropy = float(overlay.get("entropy",0) or 0)
    ov_size_ratio = float(overlay.get("size_ratio",0) or 0)
    img[16, 0] = np.log1p(max(0.0, ov_size))
    img[16, 1] = 0.0
    img[16, 2] = ov_entropy
    img[16, 3] = ov_size_ratio
    img[16, 4] = 0.0
    img[16, 5] = 0.0
    # remaining columns stay 0
    return img

def process_file(fp: str, out_split_dir: str, dataset_tag: str, vec, max_rows: int = 0, dtype: str = "float16"):
    base = os.path.splitext(os.path.basename(fp))[0]
    prefix = f"{dataset_tag}__{base}"

    n_lines = wc_l(fp)
    n = min(n_lines, max_rows) if (max_rows and max_rows < n_lines) else n_lines

    img_path = os.path.join(out_split_dir, f"{prefix}__X_section_img_v1.npy")
    sec1d_path = os.path.join(out_split_dir, f"{prefix}__X_section_1d_emberv3_section224.npy")
    y_path = os.path.join(out_split_dir, f"{prefix}__y.npy")
    v_path = os.path.join(out_split_dir, f"{prefix}__valid.npy")
    s_path = os.path.join(out_split_dir, f"{prefix}__sha256.npy")
    m_path = os.path.join(out_split_dir, f"{prefix}__meta.json")

    img_dtype = np.float16 if dtype == "float16" else np.float32

    X_img = open_memmap(img_path, mode="w+", dtype=img_dtype, shape=(n, 17, 24))
    X_1d  = open_memmap(sec1d_path, mode="w+", dtype=np.float32, shape=(n, 224))
    y = open_memmap(y_path, mode="w+", dtype=np.uint8, shape=(n,))
    v = open_memmap(v_path, mode="w+", dtype=np.uint8, shape=(n,))
    sha = open_memmap(s_path, mode="w+", dtype="S64", shape=(n,))

    errors = 0
    written = 0
    t0 = time.time()

    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            if written >= n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                lab = int(obj.get("label", 1))
                sid = str(obj.get("sha256","")).encode("utf-8")[:64]

                X_img[written] = section_to_image(obj).astype(img_dtype, copy=False)
                X_1d[written]  = vec.section(obj).astype(np.float32, copy=False)  # 224-d
                y[written] = lab
                v[written] = 1
                sha[written] = sid
            except Exception:
                errors += 1
                X_img[written] = 0
                X_1d[written] = 0
                y[written] = 1
                v[written] = 0
                sha[written] = b""
            written += 1

    n_valid = int(v[:written].sum())
    dt = time.time() - t0

    meta = {
        "dataset_tag": dataset_tag,
        "split": "CHALLENGE_MALICIOUS",
        "source_jsonl": fp,
        "base": base,
        "n_lines_in_file": n_lines,
        "n_rows_written": written,
        "n_valid": n_valid,
        "errors": errors,
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "outputs": {
            "X_section_img_v1": img_path,
            "X_section_1d_emberv3_section224": sec1d_path,
            "y": y_path,
            "valid": v_path,
            "sha256": s_path,
        }
    }
    with open(m_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[CHALLENGE MODALITIES] {base}: wrote={written} valid={n_valid} errors={errors} time={dt:.1f}s")
    return {
        "base": base,
        "meta": m_path,
        **meta["outputs"],
        "n_rows_written": written,
        "n_valid": n_valid,
        "errors": errors,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenge_dir", required=True)
    ap.add_argument("--pattern", default="*_challenge_malicious.jsonl")
    ap.add_argument("--dataset_tag", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--warnings_file", required=True)
    ap.add_argument("--fix_exports_count", action="store_true")
    ap.add_argument("--dtype", choices=["float16","float32"], default="float16")
    ap.add_argument("--max_files", type=int, default=0)
    ap.add_argument("--max_rows_per_file", type=int, default=0)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    out_split_dir = os.path.join(args.outdir, "challenge_malicious")
    ensure_dir(out_split_dir)

    tools_dir = os.path.dirname(__file__)
    EmberV3Vectorizer = dynamic_import_vectorizer(tools_dir)
    vec = EmberV3Vectorizer(strict_exports_bug=(not args.fix_exports_count), warnings_file=args.warnings_file)

    files = list_files(args.challenge_dir, args.pattern)
    if args.max_files and args.max_files > 0:
        files = files[:args.max_files]

    manifest = {
        "dataset_tag": args.dataset_tag,
        "split": "CHALLENGE_MALICIOUS",
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "items": []
    }

    for fp in files:
        it = process_file(fp, out_split_dir, args.dataset_tag, vec, max_rows=args.max_rows_per_file, dtype=args.dtype)
        manifest["items"].append(it)

    manifest_path = os.path.join(args.outdir, f"{args.dataset_tag}__challenge_malicious_section_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Wrote manifest:", manifest_path)

if __name__ == "__main__":
    main()