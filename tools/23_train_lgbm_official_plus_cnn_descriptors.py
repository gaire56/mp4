#!/usr/bin/env python3
import argparse
import json
import os
from glob import glob
from typing import Dict, Any, List, Tuple

import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def try_import_lightgbm():
    try:
        import lightgbm as lgb
        return lgb
    except Exception as e:
        raise RuntimeError(f"LightGBM import failed: {e}\nInstall with: pip install lightgbm")


def np_load(path: str):
    return np.load(path, mmap_mode="r")


def find_emb_file(emb_split_dir: str, dataset_tag: str, base: str, emb_kind: str) -> str:
    pat = os.path.join(emb_split_dir, f"{dataset_tag}__{base}__{emb_kind}*.npy")
    hits = sorted(glob(pat))
    if not hits:
        raise FileNotFoundError(f"Embedding file not found: {pat}")
    return hits[0]


def parse_mm_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    """
    Uses your existing multimodal manifest format:
      mf["paired_items"] -> it["tab"]["X"], it["tab"]["y"], it["tab"]["valid"], it["tab"]["sha256"]
    """
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


def stratified_split(y: np.ndarray, valid_frac: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n0v = int(len(idx0) * valid_frac)
    n1v = int(len(idx1) * valid_frac)

    valid_idx = np.concatenate([idx0[:n0v], idx1[:n1v]])
    train_idx = np.concatenate([idx0[n0v:], idx1[n1v:]])

    rng.shuffle(valid_idx)
    rng.shuffle(train_idx)
    return train_idx, valid_idx


def sample_concat_from_shards(
    shards: List[Dict[str, Any]],
    emb_dir: str,
    split_name: str,
    dataset_tag: str,
    emb_kind: str,
    total_samples: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Build sampled fused matrix:
      X_fused = [X_tab || X_emb]

    Returns:
      X_out, y_out, tab_dim
    """
    rng = np.random.default_rng(seed)
    n_shards = len(shards)
    per = max(1, total_samples // n_shards)

    emb_split_dir = os.path.join(emb_dir, split_name.lower())

    # infer dims
    X0 = np_load(shards[0]["X_tab"])
    tab_dim = int(X0.shape[1])

    emb0_path = find_emb_file(emb_split_dir, dataset_tag, shards[0]["base"], emb_kind)
    E0 = np_load(emb0_path)
    emb_dim = int(E0.shape[1])

    plan = []
    planned_total = 0

    for sh in shards:
        v = np_load(sh["valid"]).astype(np.uint8)
        valid_idx = np.where(v == 1)[0]
        k = min(per, len(valid_idx))
        plan.append((sh, valid_idx, k))
        planned_total += k

    # top-up to exact requested sample count if possible
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

    X_out = np.zeros((planned_total, tab_dim + emb_dim), dtype=np.float32)
    y_out = np.zeros((planned_total,), dtype=np.uint8)

    row = 0
    for sh, valid_idx, k in plan:
        if k <= 0:
            continue

        take = rng.choice(valid_idx, size=k, replace=False)
        take.sort()

        Xtab = np_load(sh["X_tab"])
        y = np_load(sh["y"]).astype(np.uint8)

        emb_path = find_emb_file(emb_split_dir, dataset_tag, sh["base"], emb_kind)
        E = np_load(emb_path)

        X_out[row:row+k, :tab_dim] = Xtab[take, :].astype(np.float32, copy=False)
        X_out[row:row+k, tab_dim:] = E[take, :].astype(np.float32, copy=False)
        y_out[row:row+k] = y[take]
        row += k

    return X_out, y_out, tab_dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--test_manifest", required=True)
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--dataset_tag", required=True)
    ap.add_argument("--emb_kind", default="emb_section_img")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--train_samples", type=int, default=500000)
    ap.add_argument("--test_samples", type=int, default=200000)
    ap.add_argument("--valid_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    # "official-like" config defaults based on EMBER2024 example config
    ap.add_argument("--num_iterations", type=int, default=500)
    ap.add_argument("--learning_rate", type=float, default=0.1)
    ap.add_argument("--num_leaves", type=int, default=64)
    ap.add_argument("--feature_fraction", type=float, default=0.9)
    ap.add_argument("--bagging_fraction", type=float, default=0.9)
    ap.add_argument("--bagging_freq", type=int, default=1)
    ap.add_argument("--lambda_l2", type=float, default=1.0)
    ap.add_argument("--min_data_in_leaf", type=int, default=100)
    ap.add_argument("--max_bin", type=int, default=255)
    ap.add_argument("--num_threads", type=int, default=0)
    ap.add_argument("--use_gpu", action="store_true")

    args = ap.parse_args()
    ensure_dir(args.outdir)

    lgb = try_import_lightgbm()

    train_shards = parse_mm_manifest(args.train_manifest)
    test_shards = parse_mm_manifest(args.test_manifest)

    print("Sampling fused training data...")
    Xtr_all, ytr_all, tab_dim = sample_concat_from_shards(
        train_shards, args.emb_dir, "train", args.dataset_tag, args.emb_kind,
        total_samples=args.train_samples, seed=args.seed
    )

    print("Sampling fused test data...")
    Xte, yte, _ = sample_concat_from_shards(
        test_shards, args.emb_dir, "test", args.dataset_tag, args.emb_kind,
        total_samples=args.test_samples, seed=args.seed + 1
    )

    tr_idx, va_idx = stratified_split(ytr_all, valid_frac=args.valid_frac, seed=args.seed)

    # Keep official categorical indices only for original tabular part.
    # Appended CNN descriptor columns are continuous and not categorical.
    categorical_feature = [2, 3, 4, 5, 6, 701, 702]

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",
        "num_iterations": args.num_iterations,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "lambda_l2": args.lambda_l2,
        "min_data_in_leaf": args.min_data_in_leaf,
        "max_bin": args.max_bin,
        "verbosity": -1,
        "is_unbalance": True,
    }

    if args.num_threads and args.num_threads > 0:
        params["num_threads"] = args.num_threads
    if args.use_gpu:
        params["device_type"] = "gpu"

    dtrain = lgb.Dataset(
        Xtr_all[tr_idx],
        label=ytr_all[tr_idx],
        categorical_feature=categorical_feature,
        free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        Xtr_all[va_idx],
        label=ytr_all[va_idx],
        categorical_feature=categorical_feature,
        reference=dtrain,
        free_raw_data=False,
    )

    params_path = os.path.join(args.outdir, "params_official_plus_cnn.json")
    model_path = os.path.join(args.outdir, "lgbm_model_official_plus_cnn.txt")
    metrics_path = os.path.join(args.outdir, "metrics_sample_official_plus_cnn.json")

    with open(params_path, "w", encoding="utf-8") as f:
        json.dump({
            "params": params,
            "categorical_feature": categorical_feature,
            "tab_dim": int(tab_dim),
            "emb_kind": args.emb_kind,
        }, f, indent=2)

    print("Training LightGBM with official-like config + CNN descriptors...")
    try:
        booster = lgb.train(
            params,
            dtrain,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=50),
            ],
        )
    except Exception as e:
        if args.use_gpu:
            print("GPU failed, retrying on CPU:", str(e))
            params.pop("device_type", None)
            booster = lgb.train(
                params,
                dtrain,
                valid_sets=[dvalid],
                valid_names=["valid"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100),
                    lgb.log_evaluation(period=50),
                ],
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
        "train_sample_n": int(len(ytr_all)),
        "test_sample_n": int(len(yte)),
        "tab_dim": int(tab_dim),
        "final_feature_dim": int(Xtr_all.shape[1]),
        "best_iteration": int(booster.best_iteration),
        "test_sample_auc": auc,
        "test_sample_acc": acc,
        "saved_model": model_path,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Test(sample) AUC:", auc, "ACC:", acc)
    print("Wrote:", metrics_path)


if __name__ == "__main__":
    main()