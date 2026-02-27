#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_test_tab_items(manifest_path: str) -> List[Dict[str, str]]:
    mf = load_json(manifest_path)

    # Handle multimodal manifest (your case)
    if isinstance(mf, dict) and "paired_items" in mf:
        items_list = mf["paired_items"]

    # Handle old tab-only manifest
    elif isinstance(mf, dict) and "items" in mf:
        items_list = mf["items"]

    # Handle list-style manifest
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


def try_import_lightgbm():
    try:
        import lightgbm as lgb
        return lgb
    except Exception as e:
        raise RuntimeError(
            "LightGBM not installed. Install it first:\n"
            "  pip install lightgbm\n"
            f"Import error: {e}"
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_manifest", required=True)
    ap.add_argument("--model_path", required=True, help="LightGBM model .txt")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max_shards", type=int, default=0, help="Debug: limit number of shards (0=all)")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    lgb = try_import_lightgbm()
    booster = lgb.Booster(model_file=args.model_path)

    items = parse_test_tab_items(args.test_manifest)
    if args.max_shards and args.max_shards > 0:
        items = items[:args.max_shards]

    y_all = []
    p_all = []

    per_shard = []
    total_rows = 0
    total_valid = 0

    for i, it in enumerate(items):
        X = np.load(it["X"], mmap_mode="r")
        y = np.load(it["y"], mmap_mode="r").astype(np.uint8)
        v = np.load(it["valid"], mmap_mode="r").astype(np.uint8)

        idx = np.where(v == 1)[0]
        n_valid = int(idx.size)

        # predict only valid rows
        probs = booster.predict(X[idx], num_iteration=booster.best_iteration)

        yv = y[idx]
        y_all.append(yv)
        p_all.append(probs)

        # quick per-shard stats
        shard_auc = float(roc_auc_score(yv, probs)) if n_valid > 0 else None
        shard_acc = float(accuracy_score(yv, (probs >= args.threshold).astype(np.uint8))) if n_valid > 0 else None

        per_shard.append({
            "base": it["base"],
            "n_rows": int(X.shape[0]),
            "n_valid": n_valid,
            "auc": shard_auc,
            "acc": shard_acc,
        })

        total_rows += int(X.shape[0])
        total_valid += n_valid
        print(f"[{i+1}/{len(items)}] {it['base']}: valid={n_valid} auc={shard_auc} acc={shard_acc}")

    y_all = np.concatenate(y_all, axis=0)
    p_all = np.concatenate(p_all, axis=0)

    auc = float(roc_auc_score(y_all, p_all))
    preds = (p_all >= args.threshold).astype(np.uint8)
    acc = float(accuracy_score(y_all, preds))
    cm = confusion_matrix(y_all, preds).tolist()
    report = classification_report(y_all, preds, digits=4)

    metrics = {
        "model_path": args.model_path,
        "test_manifest": args.test_manifest,
        "threshold": args.threshold,
        "total_rows_all_test_shards": total_rows,
        "total_valid_rows_evaluated": int(total_valid),
        "full_test_auc": auc,
        "full_test_accuracy": acc,
        "confusion_matrix": cm,
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "per_shard": per_shard,
    }

    metrics_path = os.path.join(args.outdir, "EMBER2024_CORE_PE_tab_lgbm_emberv3_full_test_metrics.json")
    report_path = os.path.join(args.outdir, "EMBER2024_CORE_PE_tab_lgbm_emberv3_full_test_classification_report.txt")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print("\n=== FULL TEST RESULTS ===")
    print("AUC:", auc)
    print("ACC:", acc)
    print("Confusion matrix [ [TN FP], [FN TP] ]:", cm)
    print("Wrote:", metrics_path)
    print("Wrote:", report_path)

if __name__ == "__main__":
    main()
