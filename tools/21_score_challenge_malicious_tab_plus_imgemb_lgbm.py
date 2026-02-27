#!/usr/bin/env python3
import argparse, json, os, csv
from glob import glob
from datetime import datetime
import numpy as np

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def try_import_lightgbm():
    try:
        import lightgbm as lgb
        return lgb
    except Exception as e:
        raise RuntimeError("Install lightgbm: pip install lightgbm\n" + str(e))

def find_emb(emb_dir: str, dataset_tag: str, base: str) -> str:
    pat = os.path.join(emb_dir, "challenge_malicious", f"{dataset_tag}__{base}__emb_section_img*.npy")
    hits = sorted(glob(pat))
    if not hits:
        raise FileNotFoundError(f"Missing embedding for base={base}: {pat}")
    return hits[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenge_tab_manifest", required=True)
    ap.add_argument("--challenge_imgemb_dir", required=True)
    ap.add_argument("--dataset_tag", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.35, 0.5])
    ap.add_argument("--chunk_size", type=int, default=200000)
    ap.add_argument("--save_lowest_k", type=int, default=200)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    lgb = try_import_lightgbm()
    booster = lgb.Booster(model_file=args.model_path)

    mf = load_json(args.challenge_tab_manifest)
    items = mf["items"]

    per_file = []
    lowest = []
    total_valid = 0
    total_detect = {t: 0 for t in args.thresholds}

    for it in items:
        base = it["base"]
        Xtab = np.load(it["X"], mmap_mode="r")
        y = np.load(it["y"], mmap_mode="r").astype(np.uint8)
        v = np.load(it["valid"], mmap_mode="r").astype(np.uint8)
        sha = np.load(it["sha256"], mmap_mode="r")

        idx = np.where(v == 1)[0]
        if idx.size == 0:
            continue

        emb_path = find_emb(args.challenge_imgemb_dir, args.dataset_tag, base)
        Eimg = np.load(emb_path, mmap_mode="r")

        probs_list = []
        for s in range(0, idx.size, args.chunk_size):
            j = idx[s:s+args.chunk_size]
            Xf = np.concatenate([Xtab[j].astype(np.float32, copy=False),
                                 Eimg[j].astype(np.float32, copy=False)], axis=1)
            probs_list.append(booster.predict(Xf, num_iteration=booster.best_iteration))
        probs = np.concatenate(probs_list, axis=0).astype(np.float32)

        n_valid = int(idx.size)
        total_valid += n_valid
        det_rates = {}
        for t in args.thresholds:
            hit = int((probs >= t).sum())
            total_detect[t] += hit
            det_rates[t] = hit / n_valid

        stats = {
            "min": float(probs.min()),
            "max": float(probs.max()),
            "mean": float(probs.mean()),
            "std": float(probs.std()),
            "p5": float(np.percentile(probs, 5)),
            "p50": float(np.percentile(probs, 50)),
            "p95": float(np.percentile(probs, 95)),
        }

        per_file.append({
            "base": base,
            "n_valid": n_valid,
            **{f"detect_rate@{t}": det_rates[t] for t in args.thresholds},
            **stats
        })

        # store lowest-prob malware (hard misses)
        k = min(args.save_lowest_k, probs.size)
        if k > 0:
            loc = np.argpartition(probs, k-1)[:k]
            for li in loc:
                row_idx = int(idx[li])
                sha_str = sha[row_idx].tobytes().decode("utf-8", errors="ignore").strip("\x00")
                lowest.append((float(probs[li]), base, row_idx, sha_str))

        print(f"[CHALLENGE TAB+IMG] {base}: n={n_valid} " +
              " ".join([f"det@{t}={det_rates[t]:.4f}" for t in args.thresholds]) +
              f" mean={stats['mean']:.4f} min={stats['min']:.4f}")

    overall = {
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "model_path": args.model_path,
        "challenge_tab_manifest": args.challenge_tab_manifest,
        "challenge_imgemb_dir": args.challenge_imgemb_dir,
        "total_valid": int(total_valid),
        "overall_detect_rate": {str(t): (total_detect[t] / max(1, total_valid)) for t in args.thresholds},
        "thresholds": args.thresholds
    }

    csv_path = os.path.join(args.outdir, "challenge_malicious_tab_plus_imgemb_summary_by_file.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_file[0].keys()) if per_file else ["base","n_valid"])
        w.writeheader()
        w.writerows(per_file)

    lowest.sort(key=lambda x: x[0])
    lowest = lowest[:args.save_lowest_k]
    low_path = os.path.join(args.outdir, "challenge_malicious_tab_plus_imgemb_lowest_prob_examples.csv")
    with open(low_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prob_malware","base","row_idx","sha256"])
        w.writerows(lowest)

    json_path = os.path.join(args.outdir, "challenge_malicious_tab_plus_imgemb_overall_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    print("\n=== OVERALL CHALLENGE MALICIOUS RESULTS (TAB+IMG LGBM) ===")
    print("total_valid:", total_valid)
    for t in args.thresholds:
        print(f"detect_rate@{t}:", overall["overall_detect_rate"][str(t)])
    print("Wrote:", csv_path)
    print("Wrote:", low_path)
    print("Wrote:", json_path)

if __name__ == "__main__":
    main()