#!/usr/bin/env python3
import argparse, json, os, csv
from datetime import datetime
from typing import Dict, Any, List

import numpy as np

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_json(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def try_import_lightgbm():
    try:
        import lightgbm as lgb
        return lgb
    except Exception as e:
        raise RuntimeError("Install lightgbm first: pip install lightgbm\n" + str(e))

def percentile_list(x: np.ndarray, ps=(1,5,50,95,99)) -> Dict[str, float]:
    return {f"p{p}": float(np.percentile(x, p)) for p in ps}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenge_manifest", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.35, 0.5])
    ap.add_argument("--chunk_size", type=int, default=200000)
    ap.add_argument("--save_lowest_k", type=int, default=200, help="save lowest-prob malware examples")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    lgb = try_import_lightgbm()
    booster = lgb.Booster(model_file=args.model_path)

    mf = load_json(args.challenge_manifest)
    items = mf["items"]

    per_file_rows = []
    lowest_global = []  # (prob, base, idx, sha256)

    total_valid = 0
    total_detect = {t: 0 for t in args.thresholds}

    for it in items:
        base = it["base"]
        X = np.load(it["X"], mmap_mode="r")
        y = np.load(it["y"], mmap_mode="r").astype(np.uint8)
        v = np.load(it["valid"], mmap_mode="r").astype(np.uint8)
        sha = np.load(it["sha256"], mmap_mode="r")

        idx = np.where(v == 1)[0]
        if idx.size == 0:
            continue

        # predict in chunks
        probs_list = []
        for s in range(0, idx.size, args.chunk_size):
            j = idx[s:s+args.chunk_size]
            probs_list.append(booster.predict(X[j], num_iteration=booster.best_iteration))
        probs = np.concatenate(probs_list, axis=0).astype(np.float32)

        # detection rate == recall on malicious (labels are 1)
        det = {}
        for t in args.thresholds:
            det[t] = float((probs >= t).mean())

        stats = {
            "min": float(probs.min()),
            "max": float(probs.max()),
            "mean": float(probs.mean()),
            "std": float(probs.std()),
            **percentile_list(probs),
        }

        # update totals
        n_valid = int(idx.size)
        total_valid += n_valid
        for t in args.thresholds:
            total_detect[t] += int((probs >= t).sum())

        per_file_rows.append({
            "base": base,
            "n_valid": n_valid,
            **{f"detect_rate@{t}": det[t] for t in args.thresholds},
            **stats
        })

        # collect lowest k examples from this file
        k = min(args.save_lowest_k, probs.size)
        if k > 0:
            loc = np.argpartition(probs, k-1)[:k]
            for li in loc:
                global_row_idx = int(idx[li])
                sha_str = sha[global_row_idx].tobytes().decode("utf-8", errors="ignore").strip("\x00")
                lowest_global.append((float(probs[li]), base, global_row_idx, sha_str))

        print(f"[CHALLENGE] {base}: n={n_valid} " +
              " ".join([f"det@{t}={det[t]:.4f}" for t in args.thresholds]) +
              f" mean={stats['mean']:.4f} min={stats['min']:.4f}")

    # overall summary
    overall = {
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "model_path": args.model_path,
        "challenge_manifest": args.challenge_manifest,
        "total_valid": int(total_valid),
        "thresholds": args.thresholds,
        "overall_detect_rate": {str(t): (total_detect[t] / max(1, total_valid)) for t in args.thresholds},
    }

    # write per-file CSV
    csv_path = os.path.join(args.outdir, "challenge_malicious_summary_by_file.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(per_file_rows[0].keys()) if per_file_rows else ["base","n_valid"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_file_rows)

    # write lowest-k global CSV
    lowest_global.sort(key=lambda x: x[0])  # ascending prob
    lowest_global = lowest_global[:args.save_lowest_k]
    low_path = os.path.join(args.outdir, "challenge_malicious_lowest_prob_examples.csv")
    with open(low_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prob_malware","base","row_idx","sha256"])
        w.writerows(lowest_global)

    # write overall JSON
    json_path = os.path.join(args.outdir, "challenge_malicious_overall_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    print("\n=== OVERALL CHALLENGE MALICIOUS RESULTS (TAB LGBM) ===")
    print("total_valid:", total_valid)
    for t in args.thresholds:
        print(f"detect_rate@{t}:", overall["overall_detect_rate"][str(t)])
    print("Wrote:", csv_path)
    print("Wrote:", low_path)
    print("Wrote:", json_path)

if __name__ == "__main__":
    main()
