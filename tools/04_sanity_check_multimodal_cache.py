#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def load_npy(path):
    return np.load(path, mmap_mode="r")

def pick_key(d, candidates):
    for k in candidates:
        if k in d and isinstance(d[k], str) and d[k].endswith(".npy"):
            return k
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--test_manifest", required=True)
    ap.add_argument("--print_n", type=int, default=1, help="how many shards to inspect per split")
    args = ap.parse_args()

    for split_name, mp in [("TRAIN", args.train_manifest), ("TEST", args.test_manifest)]:
        m = load_json(mp)
        items = m["paired_items"]
        print(f"\n== {split_name} ==")
        print("paired_items:", len(items))

        for i, it in enumerate(items[: args.print_n]):
            base = it["base"]
            tab = it["tab"]
            sec = it["section"]

            # Tab paths (fixed names from your pipeline)
            tab_X = tab.get("X")
            tab_y = tab.get("y")
            tab_sha = tab.get("sha256")
            tab_valid = tab.get("valid")

            # Section paths: try common keys (your 02 script may name differently)
            sec_sha = sec.get("sha256")
            sec_y = sec.get("y")
            sec_valid = sec.get("valid")

            # Guess modality arrays inside sec item
            sec_img_key = pick_key(sec, ["X_img", "X_section_img", "sec_img", "image", "img", "X_image"])
            sec_1d_key  = pick_key(sec, ["X_1d", "X_section_1d", "sec_1d", "array_1d", "X_array_1d", "X_seq"])

            print(f"\n[{split_name}] shard {i}: {base}")
            print(" tab:", {k: tab.get(k) for k in ["X","y","sha256","valid"] if k in tab})
            print(" sec keys:", sorted(list(sec.keys()))[:20], "...")

            # Load + print shapes
            X_tab = load_npy(tab_X)
            y_tab = load_npy(tab_y)
            sha_tab = load_npy(tab_sha)
            v_tab = load_npy(tab_valid)

            print("  X_tab shape/dtype:", X_tab.shape, X_tab.dtype)
            print("  y_tab shape/dtype:", y_tab.shape, y_tab.dtype, "pos_rate:", float(y_tab[:].mean()))
            print("  valid_tab mean:", float(v_tab[:].mean()))

            if sec_img_key:
                X_img = load_npy(sec[sec_img_key])
                print(f"  X_img({sec_img_key}) shape/dtype:", X_img.shape, X_img.dtype)
            else:
                print("  X_img: (not found by auto-detect)")

            if sec_1d_key:
                X_1d = load_npy(sec[sec_1d_key])
                print(f"  X_1d({sec_1d_key}) shape/dtype:", X_1d.shape, X_1d.dtype)
            else:
                print("  X_1d: (not found by auto-detect)")

            if sec_sha:
                sha_sec = load_npy(sec_sha)
                print("  sha match:", bool((sha_tab == sha_sec).all()))
            if sec_y:
                y_sec = load_npy(sec_y)
                print("  y match:", bool((y_tab == y_sec).all()))
            if sec_valid:
                v_sec = load_npy(sec_valid)
                print("  valid match:", bool((v_tab == v_sec).all()))

if __name__ == "__main__":
    main()
