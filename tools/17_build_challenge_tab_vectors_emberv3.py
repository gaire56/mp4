#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import subprocess
from datetime import datetime
from glob import glob
from typing import Dict, Any, List

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
        # fallback
        n = 0
        with open(path, "rb") as f:
            for _ in f:
                n += 1
        return n

def dynamic_import_vectorizer(tools_dir: str):
    """
    Import EmberV3Vectorizer and FEATURE_DIM directly from tools/01_build_feature_vectors_emberv3.py
    so the vectorization is 100% consistent with your pipeline.
    """
    mod_path = os.path.join(tools_dir, "01_build_feature_vectors_emberv3.py")
    if not os.path.exists(mod_path):
        raise FileNotFoundError(f"Cannot find: {mod_path}")

    spec = importlib.util.spec_from_file_location("emberv3_vec", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    if not hasattr(mod, "EmberV3Vectorizer"):
        raise RuntimeError("EmberV3Vectorizer not found in tools/01_build_feature_vectors_emberv3.py")

    FEATURE_DIM = getattr(mod, "FEATURE_DIM", 2568)
    return mod.EmberV3Vectorizer, int(FEATURE_DIM)

def list_challenge_files(challenge_dir: str, pattern: str) -> List[str]:
    files = sorted(glob(os.path.join(challenge_dir, pattern)))
    files = [p for p in files if p.endswith(".jsonl") and os.path.isfile(p)]
    if not files:
        raise RuntimeError(f"No files found: {challenge_dir}/{pattern}")
    return files

def process_file(fp: str, out_split_dir: str, dataset_tag: str, vectorizer, feature_dim: int,
                 max_rows: int = 0) -> Dict[str, Any]:
    base = os.path.splitext(os.path.basename(fp))[0]
    prefix = f"{dataset_tag}__{base}"

    n_lines = wc_l(fp)
    n = min(n_lines, max_rows) if (max_rows and max_rows < n_lines) else n_lines

    X_path = os.path.join(out_split_dir, f"{prefix}__X_tab_emberv3.npy")
    y_path = os.path.join(out_split_dir, f"{prefix}__y.npy")
    v_path = os.path.join(out_split_dir, f"{prefix}__valid.npy")
    s_path = os.path.join(out_split_dir, f"{prefix}__sha256.npy")
    m_path = os.path.join(out_split_dir, f"{prefix}__meta.json")

    X = open_memmap(X_path, mode="w+", dtype=np.float32, shape=(n, feature_dim))
    y = open_memmap(y_path, mode="w+", dtype=np.uint8, shape=(n,))
    v = open_memmap(v_path, mode="w+", dtype=np.uint8, shape=(n,))
    sha = open_memmap(s_path, mode="w+", dtype="S64", shape=(n,))

    errors = 0
    written = 0
    pos = 0
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
                lab = int(obj.get("label", 1))  # challenge expected label=1
                sid = str(obj.get("sha256", "")).encode("utf-8")[:64]
                vec = vectorizer.transform(obj)

                X[written, :] = vec
                y[written] = lab
                v[written] = 1
                sha[written] = sid

                pos += lab
            except Exception:
                errors += 1
                X[written, :] = 0
                y[written] = 1  # keep malicious as default for challenge
                v[written] = 0
                sha[written] = b""
            written += 1

    n_valid = int(v[:written].sum())
    pos_rate = float(pos / max(1, written))
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
        "pos_rate_written": pos_rate,
        "feature_dim": feature_dim,
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "cache_files": {"X": X_path, "y": y_path, "valid": v_path, "sha256": s_path},
    }

    with open(m_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[CHALLENGE] {base}: wrote={written} valid={n_valid} errors={errors} time={dt:.1f}s")
    return {
        "base": base,
        "meta": m_path,
        "X": X_path,
        "y": y_path,
        "valid": v_path,
        "sha256": s_path,
        "n_rows_written": written,
        "n_valid": n_valid,
        "errors": errors,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenge_dir", required=True, help="Folder containing *_challenge_malicious.jsonl")
    ap.add_argument("--pattern", default="*_challenge_malicious.jsonl")
    ap.add_argument("--dataset_tag", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--warnings_file", required=True, help="resources/pefile_warnings.txt")
    ap.add_argument("--fix_exports_count", action="store_true")
    ap.add_argument("--max_files", type=int, default=0)
    ap.add_argument("--max_rows_per_file", type=int, default=0)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    out_split_dir = os.path.join(args.outdir, "challenge_malicious")
    ensure_dir(out_split_dir)

    tools_dir = os.path.join(os.path.dirname(__file__))  # tools/
    EmberV3Vectorizer, FEATURE_DIM = dynamic_import_vectorizer(tools_dir)

    vec = EmberV3Vectorizer(
        strict_exports_bug=(not args.fix_exports_count),
        warnings_file=args.warnings_file
    )

    files = list_challenge_files(args.challenge_dir, args.pattern)
    if args.max_files and args.max_files > 0:
        files = files[:args.max_files]

    manifest = {
        "dataset_tag": args.dataset_tag,
        "split": "CHALLENGE_MALICIOUS",
        "feature_dim": FEATURE_DIM,
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "items": []
    }

    for fp in files:
        it = process_file(fp, out_split_dir, args.dataset_tag, vec, FEATURE_DIM, max_rows=args.max_rows_per_file)
        manifest["items"].append(it)

    manifest_path = os.path.join(args.outdir, f"{args.dataset_tag}__challenge_malicious_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Wrote manifest:", manifest_path)

if __name__ == "__main__":
    main()
