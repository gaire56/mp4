#!/usr/bin/env python3
import argparse
import json
import os
import csv
import numpy as np
from sklearn.metrics import roc_auc_score


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def try_import_lightgbm():
    try:
        import lightgbm as lgb
        return lgb
    except Exception as e:
        raise RuntimeError("Install lightgbm first: pip install lightgbm\n" + str(e))


def parse_mm_manifest(manifest_path):
    mf = load_json(manifest_path)
    shards = []
    for it in mf["paired_items"]:
        base = it["base"]
        tab = it["tab"]
        shards.append({
            "base": base,
            "X": tab["X"],
            "y": tab["y"],
            "valid": tab["valid"],
        })
    return shards


def metrics_from_cm(tn, fp, fn, tp):
    total = tn + fp + fn + tp
    acc = (tn + tp) / total if total else 0.0

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    tpr = recall

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    return acc, precision, recall, tpr, f1, fpr, fnr


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--test_manifest", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--chunk_size", type=int, default=200000)

    ap.add_argument("--tmin", type=float, default=0.001)
    ap.add_argument("--tmax", type=float, default=0.999)
    ap.add_argument("--n_thresholds", type=int, default=500)

    args = ap.parse_args()

    ensure_dir(args.outdir)

    lgb = try_import_lightgbm()
    booster = lgb.Booster(model_file=args.model_path)

    shards = parse_mm_manifest(args.test_manifest)

    ys = []
    ps = []

    for sh in shards:

        X = np.load(sh["X"], mmap_mode="r")
        y = np.load(sh["y"], mmap_mode="r").astype(np.uint8)
        v = np.load(sh["valid"], mmap_mode="r").astype(np.uint8)

        idx = np.where(v == 1)[0]
        if idx.size == 0:
            continue

        probs_list = []

        for s in range(0, idx.size, args.chunk_size):
            j = idx[s:s + args.chunk_size]
            probs_list.append(
                booster.predict(X[j], num_iteration=booster.best_iteration)
            )

        probs = np.concatenate(probs_list).astype(np.float32)

        ys.append(y[idx])
        ps.append(probs)

    y_all = np.concatenate(ys).astype(np.uint8)
    p_all = np.concatenate(ps).astype(np.float32)

    roc_auc = float(roc_auc_score(y_all, p_all))

    print("Full test ROC-AUC:", roc_auc)
    print("Samples:", len(y_all))

    thresholds = np.linspace(args.tmin, args.tmax, args.n_thresholds)

    rows = []

    for t in thresholds:

        pred = (p_all >= t).astype(np.uint8)

        tn = int(((pred == 0) & (y_all == 0)).sum())
        fp = int(((pred == 1) & (y_all == 0)).sum())
        fn = int(((pred == 0) & (y_all == 1)).sum())
        tp = int(((pred == 1) & (y_all == 1)).sum())

        acc, precision, recall, tpr, f1, fpr, fnr = metrics_from_cm(tn, fp, fn, tp)

        rows.append([
            float(t),
            float(acc),
            float(precision),
            float(recall),
            float(tpr),
            float(f1),
            float(fpr),
            float(fnr),
            tn, fp, fn, tp
        ])

    out_csv = os.path.join(
        args.outdir,
        "EMBER2024_CORE_PE__threshold_sweep__tab_only__N500.csv"
    )

    out_json = os.path.join(
        args.outdir,
        "EMBER2024_CORE_PE__threshold_sweep__tab_only__N500.json"
    )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:

        w = csv.writer(f)

        w.writerow([
            "threshold",
            "acc",
            "precision",
            "recall",
            "tpr",
            "f1",
            "fpr",
            "fnr",
            "TN",
            "FP",
            "FN",
            "TP"
        ])

        w.writerows(rows)

    with open(out_json, "w", encoding="utf-8") as f:

        json.dump({
            "model_path": args.model_path,
            "test_manifest": args.test_manifest,
            "full_test_roc_auc": roc_auc,
            "threshold_range": {
                "tmin": args.tmin,
                "tmax": args.tmax,
                "n_thresholds": args.n_thresholds
            },
            "csv": out_csv
        }, f, indent=2)

    print("Wrote:", out_csv)
    print("Wrote:", out_json)


if __name__ == "__main__":
    main()