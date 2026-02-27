#!/usr/bin/env python3
import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def np_load_mmap(path: str):
    return np.load(path, mmap_mode="r")

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

def parse_manifest(manifest_path: str) -> List[Dict[str, str]]:
    mf = load_json(manifest_path)

    # Your multimodal manifest structure
    if isinstance(mf, dict) and "paired_items" in mf:
        items_list = mf["paired_items"]

    # Older tab-only manifest structure
    elif isinstance(mf, dict) and "items" in mf:
        items_list = mf["items"]

    # If manifest itself is already a list
    elif isinstance(mf, list):
        items_list = mf

    else:
        raise ValueError(
            f"Unsupported manifest structure. Keys: {list(mf.keys())}"
        )

    items = []
    for it in items_list:
        if "tab" not in it:
            raise ValueError(f"Missing 'tab' in item: {it.keys()}")

        tab = it["tab"]

        items.append({
            "base": it.get("base", ""),
            "X": tab["X"],
            "y": tab["y"],
            "valid": tab["valid"],
        })

    return items





def sample_from_shards(
    shards: List[Dict[str, str]],
    out_X_path: str,
    out_y_path: str,
    total_samples: int,
    seed: int = 42,
) -> Tuple[str, str, int]:
    """
    Uniform-per-shard sampling (approximately) into memmap arrays to avoid huge RAM use.
    Only samples rows where valid==1.
    """
    rng = np.random.default_rng(seed)
    ensure_dir(os.path.dirname(out_X_path))

    # Determine per-shard quota
    n_shards = len(shards)
    per = max(1, total_samples // n_shards)

    # First pass: count how many we will really take
    plan = []
    planned_total = 0
    for sh in shards:
        v = np_load_mmap(sh["valid"]).astype(np.uint8)
        valid_idx = np.where(v == 1)[0]
        k = min(per, len(valid_idx))
        plan.append((sh, valid_idx, k))
        planned_total += k

    # If planned_total < total_samples, top up by taking extra from random shards
    if planned_total < total_samples:
        need = total_samples - planned_total
        order = rng.permutation(n_shards)
        for j in order:
            if need <= 0:
                break
            sh, valid_idx, k = plan[j]
            # remaining capacity
            extra_cap = len(valid_idx) - k
            if extra_cap <= 0:
                continue
            extra = min(extra_cap, need)
            plan[j] = (sh, valid_idx, k + extra)
            need -= extra
        planned_total = sum(k for _, _, k in plan)

    # Allocate memmaps
    # infer feature dim from first shard
    X0 = np_load_mmap(shards[0]["X"])
    feat_dim = int(X0.shape[1])
    X_out = np.lib.format.open_memmap(out_X_path, mode="w+", dtype=np.float32, shape=(planned_total, feat_dim))
    y_out = np.lib.format.open_memmap(out_y_path, mode="w+", dtype=np.uint8, shape=(planned_total,))

    # Fill
    row = 0
    for sh, valid_idx, k in plan:
        if k <= 0:
            continue
        # choose k indices from valid rows
        take = rng.choice(valid_idx, size=k, replace=False)
        take.sort()

        X = np_load_mmap(sh["X"])
        y = np_load_mmap(sh["y"]).astype(np.uint8)

        X_out[row:row+k, :] = X[take, :].astype(np.float32, copy=False)
        y_out[row:row+k] = y[take]
        row += k

    assert row == planned_total
    return out_X_path, out_y_path, planned_total

def try_import_lightgbm():
    try:
        import lightgbm as lgb
        return lgb
    except Exception as e:
        raise RuntimeError(
            "LightGBM is not installed in this environment. Install it first:\n"
            "  pip install lightgbm\n"
            f"Original import error: {e}"
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--test_manifest", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--train_samples", type=int, default=500_000, help="Sample size from TRAIN shards (default 500k)")
    ap.add_argument("--test_samples", type=int, default=200_000, help="Sample size from TEST shards (default 200k)")
    ap.add_argument("--valid_frac", type=float, default=0.1, help="Validation fraction from train sample")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--num_threads", type=int, default=0)
    ap.add_argument("--use_gpu", action="store_true", help="Try LightGBM GPU (OpenCL). Falls back to CPU if fails.")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    lgb = try_import_lightgbm()

    train_shards = parse_manifest(args.train_manifest)
    test_shards = parse_manifest(args.test_manifest)

    # Sample into memmaps (clear naming)
    sample_dir = os.path.join(args.outdir, "sampled_arrays")
    ensure_dir(sample_dir)

    trainX_path = os.path.join(sample_dir, "EMBER2024_CORE_PE_train_sample_X_emberv3.npy")
    trainy_path = os.path.join(sample_dir, "EMBER2024_CORE_PE_train_sample_y.npy")
    testX_path  = os.path.join(sample_dir, "EMBER2024_CORE_PE_test_sample_X_emberv3.npy")
    testy_path  = os.path.join(sample_dir, "EMBER2024_CORE_PE_test_sample_y.npy")

    t0 = time.time()
    trainX_path, trainy_path, n_tr = sample_from_shards(
        train_shards, trainX_path, trainy_path, args.train_samples, seed=args.seed
    )
    testX_path, testy_path, n_te = sample_from_shards(
        test_shards, testX_path, testy_path, args.test_samples, seed=args.seed + 1
    )
    print(f"Sampled train={n_tr} test={n_te} in {time.time()-t0:.1f}s")
    Xtr = np.load(trainX_path, mmap_mode="r")
    ytr = np.load(trainy_path, mmap_mode="r")
    Xte = np.load(testX_path, mmap_mode="r")
    yte = np.load(testy_path, mmap_mode="r")

    train_idx, valid_idx = stratified_split(ytr, valid_frac=args.valid_frac, seed=args.seed)

    # LightGBM params (strong baseline)
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

    # Try GPU if requested (OpenCL). Many CUDA servers won’t have OpenCL configured.
    if args.use_gpu:
        params["device_type"] = "gpu"

    dtrain = lgb.Dataset(Xtr[train_idx], label=ytr[train_idx], free_raw_data=False)
    dvalid = lgb.Dataset(Xtr[valid_idx], label=ytr[valid_idx], reference=dtrain, free_raw_data=False)

    print("Training LightGBM with params:", params)

    model_path = os.path.join(args.outdir, "lgbm_model.txt")
    metrics_path = os.path.join(args.outdir, "metrics.json")
    params_path = os.path.join(args.outdir, "params.json")

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
        # GPU fallback
        if args.use_gpu:
            print("GPU training failed, falling back to CPU. Error:", str(e))
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

    # Evaluate on test sample
    yprob = booster.predict(Xte, num_iteration=booster.best_iteration)
    # compute AUC manually to avoid extra deps
    # (sklearn is fine too, but keep it simple)
    from sklearn.metrics import roc_auc_score, accuracy_score
    auc = float(roc_auc_score(yte, yprob))
    acc = float(accuracy_score(yte, (yprob >= 0.5).astype(np.uint8)))

    out = {
        "train_sample_n": int(n_tr),
        "test_sample_n": int(n_te),
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
