#!/usr/bin/env python3
import argparse
import json
import os
import time
from datetime import datetime
from glob import glob
from typing import Dict, Any, List, Tuple, Optional

import numpy as np


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def np_load_mmap(path: str):
    return np.load(path, mmap_mode="r")

def try_import_lightgbm():
    try:
        import lightgbm as lgb
        return lgb
    except Exception as e:
        raise RuntimeError(
            "LightGBM is not installed. Install it:\n"
            "  pip install lightgbm\n"
            f"Import error: {e}"
        )

def find_emb_file(emb_split_dir: str, dataset_tag: str, base: str, emb_kind: str) -> str:
    # Example: EMBER2024_CORE_PE__2023-...__emb_section_cat256.npy
    pat = os.path.join(emb_split_dir, f"{dataset_tag}__{base}__{emb_kind}*.npy")
    hits = sorted(glob(pat))
    if not hits:
        raise FileNotFoundError(f"Embedding file not found: {pat}")
    return hits[0]

def stratified_split(y: np.ndarray, valid_frac: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0); rng.shuffle(idx1)
    n0v = int(len(idx0) * valid_frac)
    n1v = int(len(idx1) * valid_frac)
    valid_idx = np.concatenate([idx0[:n0v], idx1[:n1v]])
    train_idx = np.concatenate([idx0[n0v:], idx1[n1v:]])
    rng.shuffle(train_idx); rng.shuffle(valid_idx)
    return train_idx, valid_idx

def parse_mm_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    mf = load_json(manifest_path)
    items = []
    for it in mf["paired_items"]:
        base = it["base"]
        tab = it["tab"]
        items.append({
            "base": base,
            "X_tab": tab["X"],
            "y": tab["y"],
            "valid": tab["valid"],
            "sha256": tab["sha256"],
        })
    return items

def sample_concat_from_shards(
    shards: List[Dict[str, Any]],
    emb_dir: str,
    split_name: str,
    dataset_tag: str,
    emb_kind: str,
    out_X_path: str,
    out_y_path: str,
    total_samples: int,
    seed: int = 42,
) -> Tuple[str, str, int, int]:
    """
    Uniform-per-shard sampling into memmap arrays:
      X_out = [X_tab, X_emb]
    """
    rng = np.random.default_rng(seed)
    ensure_dir(os.path.dirname(out_X_path))

    n_shards = len(shards)
    per = max(1, total_samples // n_shards)

    # infer dims from first shard
    X0 = np_load_mmap(shards[0]["X_tab"])
    tab_dim = int(X0.shape[1])

    emb_split_dir = os.path.join(emb_dir, split_name.lower())
    emb0_path = find_emb_file(emb_split_dir, dataset_tag, shards[0]["base"], emb_kind)
    E0 = np_load_mmap(emb0_path)
    emb_dim = int(E0.shape[1])

    plan = []
    planned_total = 0

    for sh in shards:
        v = np_load_mmap(sh["valid"]).astype(np.uint8)
        valid_idx = np.where(v == 1)[0]
        k = min(per, len(valid_idx))
        plan.append((sh, valid_idx, k))
        planned_total += k

    # top up
    if planned_total < total_samples:
        need = total_samples - planned_total
        order = rng.permutation(n_shards)
        for j in order:
            if need <= 0:
                break
            sh, valid_idx, k = plan[j]
            extra_cap = len(valid_idx) - k
            if extra_cap <= 0:
                continue
            extra = min(extra_cap, need)
            plan[j] = (sh, valid_idx, k + extra)
            need -= extra
        planned_total = sum(k for _, _, k in plan)

    X_out = np.lib.format.open_memmap(out_X_path, mode="w+", dtype=np.float32, shape=(planned_total, tab_dim + emb_dim))
    y_out = np.lib.format.open_memmap(out_y_path, mode="w+", dtype=np.uint8, shape=(planned_total,))

    row = 0
    for sh, valid_idx, k in plan:
        if k <= 0:
            continue
        take = rng.choice(valid_idx, size=k, replace=False)
        take.sort()

        Xtab = np_load_mmap(sh["X_tab"])
        y = np_load_mmap(sh["y"]).astype(np.uint8)

        emb_path = find_emb_file(emb_split_dir, dataset_tag, sh["base"], emb_kind)
        E = np_load_mmap(emb_path)

        # concat
        X_out[row:row+k, :tab_dim] = Xtab[take, :].astype(np.float32, copy=False)
        X_out[row:row+k, tab_dim:] = E[take, :].astype(np.float32, copy=False)
        y_out[row:row+k] = y[take]
        row += k

    assert row == planned_total
    return out_X_path, out_y_path, planned_total, (tab_dim + emb_dim)

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--test_manifest", required=True)
    ap.add_argument("--emb_dir", required=True, help="outputs/07_section_embeddings/... root")
    ap.add_argument("--dataset_tag", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--emb_kind", default="emb_section_cat", help="prefix in filename, e.g. emb_section_cat")
    ap.add_argument("--train_samples", type=int, default=500000)
    ap.add_argument("--test_samples", type=int, default=200000)
    ap.add_argument("--valid_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--use_gpu", action="store_true", help="Try LightGBM GPU (OpenCL) then fallback to CPU")
    ap.add_argument("--num_threads", type=int, default=0)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    lgb = try_import_lightgbm()

    train_shards = parse_mm_manifest(args.train_manifest)
    test_shards = parse_mm_manifest(args.test_manifest)

    sample_dir = os.path.join(args.outdir, "sampled_arrays")
    ensure_dir(sample_dir)

    trainX = os.path.join(sample_dir, f"{args.dataset_tag}_train_sample_X_tab_plus_{args.emb_kind}.npy")
    trainy = os.path.join(sample_dir, f"{args.dataset_tag}_train_sample_y.npy")
    testX  = os.path.join(sample_dir, f"{args.dataset_tag}_test_sample_X_tab_plus_{args.emb_kind}.npy")
    testy  = os.path.join(sample_dir, f"{args.dataset_tag}_test_sample_y.npy")

    t0 = time.time()
    trainX, trainy, ntr, feat_dim = sample_concat_from_shards(
        train_shards, args.emb_dir, "train", args.dataset_tag, args.emb_kind,
        trainX, trainy, args.train_samples, seed=args.seed
    )
    testX, testy, nte, _ = sample_concat_from_shards(
        test_shards, args.emb_dir, "test", args.dataset_tag, args.emb_kind,
        testX, testy, args.test_samples, seed=args.seed + 1
    )
    print(f"Sampled fused arrays in {time.time()-t0:.1f}s")
    print("Train sample:", ntr, "Test sample:", nte, "Feature dim:", feat_dim)

    Xtr = np.load(trainX, mmap_mode="r")
    ytr = np.load(trainy, mmap_mode="r")
    Xte = np.load(testX, mmap_mode="r")
    yte = np.load(testy, mmap_mode="r")

    tr_idx, va_idx = stratified_split(ytr, valid_frac=args.valid_frac, seed=args.seed)

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": 0.05,
        "num_leaves": 255,
        "max_depth": -1,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_bin": 255,
        "verbosity": -1,
    }
    if args.num_threads and args.num_threads > 0:
        params["num_threads"] = args.num_threads
    if args.use_gpu:
        params["device_type"] = "gpu"

    dtrain = lgb.Dataset(Xtr[tr_idx], label=ytr[tr_idx], free_raw_data=False)
    dvalid = lgb.Dataset(Xtr[va_idx], label=ytr[va_idx], reference=dtrain, free_raw_data=False)

    print("Training LightGBM with params:", params)

    params_path = os.path.join(args.outdir, "params.json")
    model_path = os.path.join(args.outdir, "lgbm_model.txt")
    metrics_path = os.path.join(args.outdir, "metrics_sample.json")

    with open(params_path, "w", encoding="utf-8") as f:
        json.dump({"params": params, "created_at_utc": datetime.utcnow().isoformat()+"Z"}, f, indent=2)

    try:
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(stopping_rounds=200), lgb.log_evaluation(period=50)],
        )
    except Exception as e:
        if args.use_gpu:
            print("GPU failed, fallback to CPU:", str(e))
            params.pop("device_type", None)
            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=5000,
                valid_sets=[dvalid],
                valid_names=["valid"],
                callbacks=[lgb.early_stopping(stopping_rounds=200), lgb.log_evaluation(period=50)],
            )
        else:
            raise

    booster.save_model(model_path)
    print("Saved model:", model_path)

    from sklearn.metrics import roc_auc_score, accuracy_score
    yprob = booster.predict(Xte, num_iteration=booster.best_iteration)
    auc = float(roc_auc_score(yte, yprob))
    acc = float(accuracy_score(yte, (yprob >= 0.5).astype(np.uint8)))

    out = {
        "train_sample_n": int(ntr),
        "test_sample_n": int(nte),
        "feature_dim": int(feat_dim),
        "best_iteration": int(booster.best_iteration),
        "test_sample_auc": auc,
        "test_sample_acc": acc,
        "saved_model": model_path,
        "created_at_utc": datetime.utcnow().isoformat()+"Z",
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Test(sample) AUC:", auc, "ACC:", acc)
    print("Wrote:", metrics_path)

if __name__ == "__main__":
    main()
