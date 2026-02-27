#!/usr/bin/env python3
import argparse, json, os
from datetime import datetime
from glob import glob
from typing import Dict, Any, List

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

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
        raise RuntimeError("Install lightgbm first: pip install lightgbm\n" + str(e))

def find_emb_file(emb_split_dir: str, dataset_tag: str, base: str, emb_kind: str) -> str:
    pat = os.path.join(emb_split_dir, f"{dataset_tag}__{base}__{emb_kind}*.npy")
    hits = sorted(glob(pat))
    if not hits:
        raise FileNotFoundError(f"Embedding file not found: {pat}")
    return hits[0]

def parse_mm_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    mf = load_json(manifest_path)
    out = []
    for it in mf["paired_items"]:
        base = it["base"]
        tab = it["tab"]
        out.append({
            "base": base,
            "X_tab": tab["X"],
            "y": tab["y"],
            "valid": tab["valid"],
        })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_manifest", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--dataset_tag", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--emb_kind", default="emb_section_cat")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--chunk_size", type=int, default=200000, help="rows per predict chunk")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    lgb = try_import_lightgbm()
    booster = lgb.Booster(model_file=args.model_path)

    shards = parse_mm_manifest(args.test_manifest)
    emb_split_dir = os.path.join(args.emb_dir, "test")

    y_all = []
    p_all = []
    per_shard = []

    for i, sh in enumerate(shards):
        base = sh["base"]
        Xtab = np_load_mmap(sh["X_tab"])
        y = np_load_mmap(sh["y"]).astype(np.uint8)
        v = np_load_mmap(sh["valid"]).astype(np.uint8)
        idx = np.where(v == 1)[0]
        if idx.size == 0:
            continue

        emb_path = find_emb_file(emb_split_dir, args.dataset_tag, base, args.emb_kind)
        E = np_load_mmap(emb_path)

        # predict in chunks to avoid big concat memory
        probs_list = []
        for s in range(0, idx.size, args.chunk_size):
            j = idx[s:s+args.chunk_size]
            Xf = np.concatenate([Xtab[j].astype(np.float32, copy=False),
                                 E[j].astype(np.float32, copy=False)], axis=1)
            probs = booster.predict(Xf, num_iteration=booster.best_iteration)
            probs_list.append(probs)

        probs = np.concatenate(probs_list, axis=0)
        yv = y[idx]

        shard_auc = float(roc_auc_score(yv, probs)) if len(np.unique(yv)) > 1 else None
        shard_acc = float(accuracy_score(yv, (probs >= args.threshold).astype(np.uint8)))
        per_shard.append({"base": base, "n_valid": int(idx.size), "auc": shard_auc, "acc": shard_acc})
        print(f"[{i+1}/{len(shards)}] {base}: n={int(idx.size)} auc={shard_auc} acc={shard_acc}")

        y_all.append(yv)
        p_all.append(probs)

    y_all = np.concatenate(y_all).astype(np.uint8)
    p_all = np.concatenate(p_all).astype(np.float32)

    auc = float(roc_auc_score(y_all, p_all))
    preds = (p_all >= args.threshold).astype(np.uint8)
    acc = float(accuracy_score(y_all, preds))
    cm = confusion_matrix(y_all, preds).tolist()
    report = classification_report(y_all, preds, digits=4)

    metrics = {
        "model_path": args.model_path,
        "test_manifest": args.test_manifest,
        "emb_dir": args.emb_dir,
        "emb_kind": args.emb_kind,
        "threshold": args.threshold,
        "full_test_auc": auc,
        "full_test_accuracy": acc,
        "confusion_matrix": cm,
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "per_shard": per_shard,
    }

    metrics_path = os.path.join(args.outdir, f"{args.dataset_tag}_tab_plus_{args.emb_kind}_full_test_metrics.json")
    report_path = os.path.join(args.outdir, f"{args.dataset_tag}_tab_plus_{args.emb_kind}_full_test_report.txt")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print("\n=== FULL TEST RESULTS (TAB + EMB) ===")
    print("AUC:", auc)
    print("ACC:", acc)
    print("Confusion matrix [ [TN FP], [FN TP] ]:", cm)
    print("Wrote:", metrics_path)
    print("Wrote:", report_path)

if __name__ == "__main__":
    main()
