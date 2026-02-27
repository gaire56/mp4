#!/usr/bin/env python3
import argparse, json, os
from glob import glob
import numpy as np

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def try_import_lightgbm():
    try:
        import lightgbm as lgb
        return lgb
    except Exception as e:
        raise RuntimeError("Install lightgbm first: pip install lightgbm\n" + str(e))

def find_emb_file(emb_split_dir, dataset_tag, base, emb_kind="emb_section_img"):
    pat = os.path.join(emb_split_dir, f"{dataset_tag}__{base}__{emb_kind}*.npy")
    hits = sorted(glob(pat))
    if not hits:
        raise FileNotFoundError(f"Embedding file not found: {pat}")
    return hits[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_tag", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tab_dir", required=True, help="outputs/01_feature_vectors/... root")
    ap.add_argument("--emb_dir", required=True, help="outputs/07_section_embeddings/... root")
    ap.add_argument("--split", choices=["train","test"], default="test")
    ap.add_argument("--base", required=True, help="shard base name, e.g. 2024-09-22_2024-09-28_Win32_test")
    ap.add_argument("--idx", type=int, required=True, help="row index inside shard")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--emb_kind", default="emb_section_img")
    ap.add_argument("--final_summary", default="outputs/10_final_results/EMBER2024_CORE_PE_final_v1/final_summary.json")
    args = ap.parse_args()

    lgb = try_import_lightgbm()
    booster = lgb.Booster(model_file=args.model_path)

    split_dir_tab = os.path.join(args.tab_dir, args.split)
    split_dir_emb = os.path.join(args.emb_dir, args.split)

    # tab files
    X_tab = os.path.join(split_dir_tab, f"{args.dataset_tag}__{args.base}__X_tab_emberv3.npy")
    y_path = os.path.join(split_dir_tab, f"{args.dataset_tag}__{args.base}__y.npy")
    v_path = os.path.join(split_dir_tab, f"{args.dataset_tag}__{args.base}__valid.npy")
    sha_path = os.path.join(split_dir_tab, f"{args.dataset_tag}__{args.base}__sha256.npy")

    if not os.path.exists(X_tab):
        raise FileNotFoundError(X_tab)

    emb_path = find_emb_file(split_dir_emb, args.dataset_tag, args.base, args.emb_kind)

    X = np.load(X_tab, mmap_mode="r")
    E = np.load(emb_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r").astype(np.uint8)
    v = np.load(v_path, mmap_mode="r").astype(np.uint8)
    sha = np.load(sha_path, mmap_mode="r")

    idx = args.idx
    if idx < 0 or idx >= X.shape[0]:
        raise IndexError(f"idx out of range: 0..{X.shape[0]-1}")

    if int(v[idx]) != 1:
        print("Warning: selected row is marked invalid. Output may be meaningless.")

    x = np.concatenate([X[idx].astype(np.float32, copy=False), E[idx].astype(np.float32, copy=False)], axis=0)[None, :]
    prob = float(booster.predict(x, num_iteration=booster.best_iteration)[0])
    pred = 1 if prob >= args.threshold else 0
    true = int(y[idx])
    sha_str = sha[idx].tobytes().decode("utf-8", errors="ignore").strip("\x00")

    print("=== Inference Result ===")
    print("base:", args.base)
    print("idx:", idx)
    print("sha256:", sha_str)
    print("true_label:", true)
    print("prob_malware:", prob)
    print("pred_label(threshold={}): {}".format(args.threshold, pred))

    # helpful pointers
    if os.path.exists(args.final_summary):
        summ = load_json(args.final_summary)
        print("\n--- Pointers ---")
        print("Final summary:", args.final_summary)
        print("SHAP summary:", summ["xai_outputs"]["shap_summary_json"])
        print("Grad-CAM meta:", summ["xai_outputs"]["gradcam_meta_json"])

if __name__ == "__main__":
    main()
