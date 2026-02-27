#!/usr/bin/env python3
"""
Build section-based multimodal caches from EMBER2024_CORE PE JSONL.

Outputs (per jsonl shard):
  - X_section_img_v1.npy : (N, 17, 24) float16  (16 section rows + 1 overlay row)
  - X_section_1d_emberv3_section224.npy : (N, 224) float16  (EMBERv3-compatible section vector)
  - y.npy, sha256.npy, valid.npy, meta.json

This script is designed to align 1:1 with the raw jsonl order.
"""

import argparse
import json
import os
import sys
import time
import subprocess
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.lib.format import open_memmap
from sklearn.feature_extraction import FeatureHasher

# ----------------------------
# Config
# ----------------------------
SECTION_IMG_ROWS = 17  # 16 sections + 1 overlay
SECTION_IMG_COLS = 25
SECTION_1D_DIM = 224


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def wc_l(path: str) -> int:
    out = subprocess.check_output(["wc", "-l", path], text=True).strip()
    return int(out.split()[0])


def list_jsonl(folder_or_file: str, pattern: str, recursive: bool = False) -> List[str]:
    if os.path.isfile(folder_or_file):
        return [folder_or_file]
    if not os.path.isdir(folder_or_file):
        raise FileNotFoundError(f"Not found: {folder_or_file}")

    if recursive:
        files = glob(os.path.join(folder_or_file, "**", pattern), recursive=True)
    else:
        files = glob(os.path.join(folder_or_file, pattern))

    files = sorted([p for p in files if p.endswith(".jsonl") and os.path.isfile(p)])
    if not files:
        raise RuntimeError(f"No .jsonl files found in {folder_or_file} with pattern {pattern}")
    return files


# ----------------------------
# Section image featurizer (2D)
# ----------------------------
@dataclass
class SectionImageSpec:
    max_sections: int = 16
    rows: int = SECTION_IMG_ROWS
    cols: int = SECTION_IMG_COLS

    # Common section names (one-hot). 'OTHER' handles unknown names.
    sec_names: Tuple[str, ...] = (
        ".text", ".rdata", ".data", ".rsrc", ".reloc", ".idata", ".edata", ".pdata", ".tls", ".bss"
    )

    # Common props/characteristics (multi-hot)
    props: Tuple[str, ...] = (
        "CNT_CODE",
        "CNT_INITIALIZED_DATA",
        "CNT_UNINITIALIZED_DATA",
        "MEM_EXECUTE",
        "MEM_READ",
        "MEM_WRITE",
        "MEM_DISCARDABLE",
        "MEM_SHARED",
    )


class SectionImageFeaturizer:
    """
    Creates a fixed-size matrix from the variable-length PE section list.

    Column layout (24 cols total):
      0..4:  log1p(size), log1p(vsize), entropy, size_ratio, vsize_ratio
      5:     is_named
      6..15: section name one-hot (10 names)
      16:    OTHER name
      17..24: props multi-hot (8 props)  [ends at 23]
    """

    def __init__(self, spec: SectionImageSpec = SectionImageSpec()):
        self.spec = spec
        self.name_to_idx = {n: i for i, n in enumerate(self.spec.sec_names)}
        self.prop_to_idx = {p: i for i, p in enumerate(self.spec.props)}

        expected = 6 + len(self.spec.sec_names) + 1 + len(self.spec.props)
        if expected != self.spec.cols:
            raise ValueError(
                f"SECTION_IMG_COLS mismatch: layout={expected} but SECTION_IMG_COLS={self.spec.cols}. "
                f"Adjust SECTION_IMG_COLS or the name/prop lists."
            )

    def _row_from_section(self, sec: Dict[str, Any]) -> np.ndarray:
        name = str(sec.get("name", "") or "").lower()
        props = sec.get("props", []) or []
        props = [str(p) for p in props]

        size = float(sec.get("size", 0) or 0)
        vsize = float(sec.get("vsize", 0) or 0)
        entropy = float(sec.get("entropy", 0) or 0)
        size_ratio = float(sec.get("size_ratio", 0) or 0)
        vsize_ratio = float(sec.get("vsize_ratio", 0) or 0)

        row = np.zeros(self.spec.cols, dtype=np.float32)
        row[0] = np.log1p(size)
        row[1] = np.log1p(vsize)
        row[2] = entropy
        row[3] = size_ratio
        row[4] = vsize_ratio
        row[5] = 1.0 if name else 0.0

        # section name one-hot
        base = 6
        if name in self.name_to_idx:
            row[base + self.name_to_idx[name]] = 1.0
        else:
            row[base + len(self.spec.sec_names)] = 1.0  # OTHER

        # props multi-hot
        prop_base = base + len(self.spec.sec_names) + 1
        for p in props:
            if p in self.prop_to_idx:
                row[prop_base + self.prop_to_idx[p]] = 1.0

        return row

    def _row_from_overlay(self, overlay: Dict[str, Any]) -> np.ndarray:
        # Overlay doesn't have name/props; keep numeric fields only.
        size = float(overlay.get("size", 0) or 0)
        entropy = float(overlay.get("entropy", 0) or 0)
        size_ratio = float(overlay.get("size_ratio", 0) or 0)

        row = np.zeros(self.spec.cols, dtype=np.float32)
        row[0] = np.log1p(size)
        row[2] = entropy
        row[3] = size_ratio

        # Mark overlay row as OTHER in name slot (helps CNN distinguish row-type)
        base = 6
        row[base + len(self.spec.sec_names)] = 1.0
        return row

    def transform(self, obj: Dict[str, Any]) -> np.ndarray:
        sec_block = obj.get("section", {}) or {}
        sections = sec_block.get("sections", []) or []
        overlay = sec_block.get("overlay", {}) or {}

        mat = np.zeros((self.spec.rows, self.spec.cols), dtype=np.float32)

        for i, sec in enumerate(sections[: self.spec.max_sections]):
            if isinstance(sec, dict):
                mat[i, :] = self._row_from_section(sec)

        # last row reserved for overlay
        if isinstance(overlay, dict) and overlay:
            mat[self.spec.rows - 1, :] = self._row_from_overlay(overlay)

        return mat


# ----------------------------
# EMBERv3-compatible Section(224) (1D)
# ----------------------------
class EmberV3Section224:
    """Replicates the 'section' feature group used in thrember/EMBERv3 (224 dims)."""

    def __init__(self):
        self._hs_sizes = FeatureHasher(50, input_type="pair")
        self._hs_vsize = FeatureHasher(50, input_type="pair")
        self._hs_entropy = FeatureHasher(50, input_type="pair")
        self._hs_chars = FeatureHasher(50, input_type="string")
        self._hs_entry = FeatureHasher(10, input_type="string")

    def transform(self, obj: Dict[str, Any]) -> np.ndarray:
        s = obj.get("section", {}) or {}
        if not s:
            return np.zeros(SECTION_1D_DIM, dtype=np.float32)

        sections = s.get("sections", []) or []
        overlay = s.get("overlay", {}) or {"size": 0, "size_ratio": 0, "entropy": 0}
        entry = s.get("entry", "") or ""

        n_sections = len(sections)
        n_zero = sum(1 for sec in sections if isinstance(sec, dict) and (sec.get("size", 0) or 0) == 0)
        n_empty_name = sum(1 for sec in sections if isinstance(sec, dict) and (sec.get("name", "") or "") == "")
        n_rx = sum(
            1
            for sec in sections
            if isinstance(sec, dict)
            and ("MEM_READ" in (sec.get("props", []) or []) and "MEM_EXECUTE" in (sec.get("props", []) or []))
        )
        n_w = sum(1 for sec in sections if isinstance(sec, dict) and ("MEM_WRITE" in (sec.get("props", []) or [])))

        entropies = [float(sec.get("entropy", 0)) for sec in sections if isinstance(sec, dict)] + [
            float(overlay.get("entropy", 0) if isinstance(overlay, dict) else 0),
            0.0,
        ]
        size_ratios = [float(sec.get("size_ratio", 0)) for sec in sections if isinstance(sec, dict)] + [
            float(overlay.get("size_ratio", 0) if isinstance(overlay, dict) else 0),
            0.0,
        ]
        vsize_ratios = [float(sec.get("vsize_ratio", 0)) for sec in sections if isinstance(sec, dict)] + [0.0]

        general = np.array(
            [
                n_sections,
                n_zero,
                n_empty_name,
                n_rx,
                n_w,
                max(entropies) if entropies else 0.0,
                min(entropies) if entropies else 0.0,
                max(size_ratios) if size_ratios else 0.0,
                min(size_ratios) if size_ratios else 0.0,
                max(vsize_ratios) if vsize_ratios else 0.0,
                min(vsize_ratios) if vsize_ratios else 0.0,
            ],
            dtype=np.float32,
        )

        section_sizes = [(sec.get("name", ""), float(sec.get("size", 0))) for sec in sections if isinstance(sec, dict)]
        section_vsize = [(sec.get("name", ""), float(sec.get("vsize", 0))) for sec in sections if isinstance(sec, dict)]
        section_entropy = [(sec.get("name", ""), float(sec.get("entropy", 0))) for sec in sections if isinstance(sec, dict)]
        characteristics = [
            f"{sec.get('name','')}:{p}"
            for sec in sections
            if isinstance(sec, dict)
            for p in (sec.get("props", []) or [])
        ]

        hs_sizes = self._hs_sizes.transform([section_sizes]).toarray()[0]
        hs_vsize = self._hs_vsize.transform([section_vsize]).toarray()[0]
        hs_entropy = self._hs_entropy.transform([section_entropy]).toarray()[0]
        hs_chars = self._hs_chars.transform([characteristics]).toarray()[0]
        hs_entry = self._hs_entry.transform([[entry]]).toarray()[0]

        out = np.hstack(
            [
                general,
                hs_sizes,
                hs_vsize,
                hs_entropy,
                hs_chars,
                hs_entry,
                float(overlay.get("size", 0) if isinstance(overlay, dict) else 0),
                float(overlay.get("size_ratio", 0) if isinstance(overlay, dict) else 0),
                float(overlay.get("entropy", 0) if isinstance(overlay, dict) else 0),
            ]
        ).astype(np.float32)

        if out.size != SECTION_1D_DIM:
            out = np.pad(out[:SECTION_1D_DIM], (0, max(0, SECTION_1D_DIM - out.size)), constant_values=0).astype(np.float32)
        return out


# ----------------------------
# Writer (sharded)
# ----------------------------
def write_sharded(
    split: str,
    files: List[str],
    outdir: str,
    dataset_tag: str,
    label_key: str,
    id_key: str,
    max_samples_per_file: int = 0,
    dtype: str = "float16",
    debug_first_error: bool = False,
) -> None:
    split_dir = os.path.join(outdir, split.lower())
    ensure_dir(split_dir)

    img_featurizer = SectionImageFeaturizer()
    sec224 = EmberV3Section224()

    manifest = {
        "dataset_tag": dataset_tag,
        "split": split,
        "section_img": {"shape": [SECTION_IMG_ROWS, SECTION_IMG_COLS], "dtype": dtype},
        "section_1d": {"shape": [SECTION_1D_DIM], "dtype": dtype},
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "items": [],
    }

    np_dtype = np.float16 if dtype == "float16" else np.float32

    for fp in files:
        base = os.path.splitext(os.path.basename(fp))[0]
        prefix = f"{dataset_tag}__{base}"

        n_lines = wc_l(fp)
        n = min(n_lines, max_samples_per_file) if max_samples_per_file > 0 else n_lines

        Ximg_path = os.path.join(split_dir, f"{prefix}__X_section_img_v1.npy")
        X1d_path = os.path.join(split_dir, f"{prefix}__X_section_1d_emberv3_section224.npy")
        y_path = os.path.join(split_dir, f"{prefix}__y.npy")
        s_path = os.path.join(split_dir, f"{prefix}__sha256.npy")
        v_path = os.path.join(split_dir, f"{prefix}__valid.npy")
        m_path = os.path.join(split_dir, f"{prefix}__meta.json")

        Ximg = open_memmap(Ximg_path, mode="w+", dtype=np_dtype, shape=(n, SECTION_IMG_ROWS, SECTION_IMG_COLS))
        X1d = open_memmap(X1d_path, mode="w+", dtype=np_dtype, shape=(n, SECTION_1D_DIM))
        y = open_memmap(y_path, mode="w+", dtype=np.uint8, shape=(n,))
        sha = open_memmap(s_path, mode="w+", dtype="S64", shape=(n,))
        valid = open_memmap(v_path, mode="w+", dtype=np.uint8, shape=(n,))

        errors = 0
        written = 0
        n_valid = 0
        pos = 0
        first_err_info = None
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
                    lab = int(obj.get(label_key, 0))
                    sid = str(obj.get(id_key, "")).encode("utf-8")[:64]

                    img = img_featurizer.transform(obj)
                    vec224 = sec224.transform(obj)

                    Ximg[written] = img.astype(np_dtype, copy=False)
                    X1d[written] = vec224.astype(np_dtype, copy=False)
                    y[written] = lab
                    sha[written] = sid
                    valid[written] = 1
                    n_valid += 1
                    pos += lab
                except Exception as e:
                    errors += 1
                    if first_err_info is None:
                        first_err_info = {
                            "error": repr(e),
                            "line_idx": written,
                            "file": fp,
                            "raw_line_prefix": line[:200],
                        }
                    Ximg[written] = 0
                    X1d[written] = 0
                    y[written] = 0
                    sha[written] = b""
                    valid[written] = 0
                written += 1

        pos_rate = (pos / n_valid) if n_valid else None
        dt = time.time() - t0

        meta = {
            "dataset_tag": dataset_tag,
            "split": split,
            "source_jsonl": fp,
            "n_lines_in_file": n_lines,
            "n_rows_written": written,
            "n_valid": n_valid,
            "pos_rate_valid": pos_rate,
            "errors": errors,
            "section_img": {"rows": SECTION_IMG_ROWS, "cols": SECTION_IMG_COLS, "dtype": dtype},
            "section_1d": {"dim": SECTION_1D_DIM, "dtype": dtype},
            "label_key": label_key,
            "id_key": id_key,
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
            "cache_files": {
                "X_section_img_v1": Ximg_path,
                "X_section_1d_emberv3_section224": X1d_path,
                "y": y_path,
                "sha256": s_path,
                "valid": v_path,
            },
            "first_error": first_err_info,
        }
        with open(m_path, "w", encoding="utf-8") as fmeta:
            json.dump(meta, fmeta, indent=2)

        print(f"[{split}] {base}: wrote={written} valid={n_valid} errors={errors} pos_rate={pos_rate} time={dt:.1f}s")

        if errors and debug_first_error and first_err_info is not None:
            print("--- FIRST ERROR (debug) ---")
            print(json.dumps(first_err_info, indent=2))

        manifest["items"].append(
            {
                "source_jsonl": fp,
                "base": base,
                "meta": m_path,
                "X_section_img_v1": Ximg_path,
                "X_section_1d_emberv3_section224": X1d_path,
                "y": y_path,
                "sha256": s_path,
                "valid": v_path,
                "n_rows_written": written,
                "n_valid": n_valid,
                "pos_rate_valid": pos_rate,
                "errors": errors,
            }
        )

    manifest_path = os.path.join(outdir, f"manifest_{split.lower()}_section_modalities.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Wrote manifest:", manifest_path)


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--train", required=True, help="Train folder or single jsonl")
    ap.add_argument("--test", required=True, help="Test folder or single jsonl")
    ap.add_argument("--dataset_tag", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--train_glob", default="*_train.jsonl")
    ap.add_argument("--test_glob", default="*_test.jsonl")
    ap.add_argument("--recursive", action="store_true")

    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--max_files_train", type=int, default=0)
    ap.add_argument("--max_files_test", type=int, default=0)
    ap.add_argument("--max_samples_per_file", type=int, default=0)
    ap.add_argument("--debug_first_error", action="store_true")

    args = ap.parse_args()

    ensure_dir(args.outdir)

    train_files = list_jsonl(args.train, args.train_glob, recursive=args.recursive)
    test_files = list_jsonl(args.test, args.test_glob, recursive=args.recursive)

    if args.max_files_train:
        train_files = train_files[: args.max_files_train]
    if args.max_files_test:
        test_files = test_files[: args.max_files_test]

    print("Train files:", len(train_files), "| Test files:", len(test_files))
    print("Section img:", (SECTION_IMG_ROWS, SECTION_IMG_COLS), "| Section 1D:", SECTION_1D_DIM)
    print("dtype:", args.dtype)

    write_sharded(
        split="TRAIN",
        files=train_files,
        outdir=args.outdir,
        dataset_tag=args.dataset_tag,
        label_key="label",
        id_key="sha256",
        max_samples_per_file=args.max_samples_per_file,
        dtype=args.dtype,
        debug_first_error=args.debug_first_error,
    )

    write_sharded(
        split="TEST",
        files=test_files,
        outdir=args.outdir,
        dataset_tag=args.dataset_tag,
        label_key="label",
        id_key="sha256",
        max_samples_per_file=args.max_samples_per_file,
        dtype=args.dtype,
        debug_first_error=args.debug_first_error,
    )


if __name__ == "__main__":
    main()
