#!/usr/bin/env python3
import argparse, json, os
import numpy as np

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--img_key", default="X_section_img_v1")
    ap.add_argument("--sec1d_key", default="X_section_1d_emberv3_section224")
    ap.add_argument("--n", type=int, default=4096, help="How many rows to scan per shard")
    ap.add_argument("--max_shards", type=int, default=3, help="How many shards to scan (0=all)")
    args = ap.parse_args()

    # Load the manifest and handle missing 'paired_items' key
    mf = load_json(args.manifest)
    
    # Try to access 'paired_items', or use an empty list as a fallback
    items = mf.get("paired_items", [])
    
    # If 'paired_items' is missing, print available keys and exit
    if not items:
        print("Error: 'paired_items' key not found in the manifest.")
        print(f"Available keys in manifest: {list(mf.keys())}")
        exit(1)

    if args.max_shards and args.max_shards > 0:
        items = items[:args.max_shards]

    print("Scanning shards:", len(items))
    for it in items:
        print("Item keys:", it.keys())  # Debugging: print keys for each item
        base = it.get("base")
        
        # Check for the presence of the 'section' key instead of 'sec'
        section = it.get("section")
        if section is None:
            print(f"Warning: 'section' key not found for base {base}. Skipping this item.")
            continue  # Skip this item if 'section' is missing
        
        # Safely access paths using .get(), fallback if keys are missing
        y_path = section.get("y") or it.get("tab", {}).get("y")
        v_path = section.get("valid") or it.get("tab", {}).get("valid")

        img_path = section.get(args.img_key)
        sec1d_path = section.get(args.sec1d_key)

        print("\n==", base, "==")
        print("img_path:", img_path)
        print("sec1d_path:", sec1d_path)

        # Check if file paths exist before loading
        if y_path and os.path.exists(y_path):
            y = np.load(y_path, mmap_mode="r")
        else:
            print(f"Warning: 'y' file not found at {y_path}. Skipping.")
            continue

        if v_path and os.path.exists(v_path):
            v = np.load(v_path, mmap_mode="r").astype(np.uint8)
        else:
            print(f"Warning: 'valid' file not found at {v_path}. Skipping.")
            continue

        # Identify valid indices
        idx = np.where(v == 1)[0]
        if idx.size == 0:
            print("No valid rows.")
            continue
        idx = idx[: min(args.n, idx.size)]

        # IMG stats
        if img_path and os.path.exists(img_path):
            Ximg = np.load(img_path, mmap_mode="r")
            b = Ximg[idx]
            b = b.astype(np.float32, copy=False)
            nan = np.isnan(b).sum()
            inf = np.isinf(b).sum()
            mn = float(np.nanmin(b))
            mx = float(np.nanmax(b))
            mean = float(np.nanmean(b))
            std = float(np.nanstd(b))
            zeros = float((b == 0).mean())
            print("IMG shape:", Ximg.shape, "dtype:", Ximg.dtype)
            print(f"IMG stats: min={mn:.6g} max={mx:.6g} mean={mean:.6g} std={std:.6g} zeros%={zeros*100:.2f} nan={nan} inf={inf}")
        else:
            print("IMG not found (or key mismatch).")

        # 1D stats
        if sec1d_path and os.path.exists(sec1d_path):
            X1 = np.load(sec1d_path, mmap_mode="r")
            b = X1[idx].astype(np.float32, copy=False)
            nan = np.isnan(b).sum()
            inf = np.isinf(b).sum()
            mn = float(np.nanmin(b))
            mx = float(np.nanmax(b))
            mean = float(np.nanmean(b))
            std = float(np.nanstd(b))
            zeros = float((b == 0).mean())
            print("SEC1D shape:", X1.shape, "dtype:", X1.dtype)
            print(f"SEC1D stats: min={mn:.6g} max={mx:.6g} mean={mean:.6g} std={std:.6g} zeros%={zeros*100:.2f} nan={nan} inf={inf}")
        else:
            print("SEC1D not found (or key mismatch).")

        # label balance for that shard subset
        yb = y[idx].astype(np.uint8)
        pos = int(yb.sum())
        print(f"Label subset: n={len(yb)} pos={pos} pos_rate={pos/len(yb):.4f}")

if __name__ == "__main__":
    main()
