#!/usr/bin/env python3
import argparse, json, os, csv
from glob import glob
import numpy as np
from sklearn.metrics import roc_auc_score

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def try_import_lightgbm():
    try:
        import lightgbm as lgb
        return lgb
    except Exception as e:
        raise RuntimeError("Install lightgbm first: pip install lightgbm\n" + str(e))

def find_emb_file(emb_split_dir, dataset_tag, base, emb_kind):
    pat = os.path.join(emb_split_dir, f"{dataset_tag}__{base}__{emb_kind}*.npy")
    hits = sorted(glob(pat))
    if not hits:
        raise FileNotFoundError(f"Embedding file not found: {pat}")
    return hits[0]

def parse_mm_manifest(test_manifest):
    mf = load_json(test_manifest)
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

def metrics_from_cm(tn, fp, fn, tp):
    acc = (tn + tp) / max(1, (tn+fp+fn+tp))
    prec = tp / max(1, (tp + fp))
    rec  = tp / max(1, (tp + fn))   # TPR
    f1 = 2*prec*rec / max(1e-12, (prec + rec))
    fpr = fp / max(1, (fp + tn))
    fnr = fn / max(1, (fn + tp))
    return acc, prec, rec, f1, fpr, fnr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_manifest", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--dataset_tag", required=True)
    ap.add_argument("--emb_kind", required=True)   # emb_section_img or emb_section_cat etc.
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--chunk_size", type=int, default=200000)
    ap.add_argument("--tmin", type=float, default=0.05)
    ap.add_argument("--tmax", type=float, default=0.95)
    ap.add_argument("--tstep", type=float, default=0.01)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    lgb = try_import_lightgbm()
    booster = lgb.Booster(model_file=args.model_path)

    shards = parse_mm_manifest(args.test_manifest)
    emb_split_dir = os.path.join(args.emb_dir, "test")

    ys, ps = [], []

    # collect probs on full test
    for sh in shards:
        base = sh["base"]
        Xtab = np.load(sh["X_tab"], mmap_mode="r")
        y = np.load(sh["y"], mmap_mode="r").astype(np.uint8)
        v = np.load(sh["valid"], mmap_mode="r").astype(np.uint8)
        idx = np.where(v == 1)[0]
        if idx.size == 0:
            continue

        emb_path = find_emb_file(emb_split_dir, args.dataset_tag, base, args.emb_kind)
        E = np.load(emb_path, mmap_mode="r")

        probs_list = []
        for s in range(0, idx.size, args.chunk_size):
            j = idx[s:s+args.chunk_size]
            Xf = np.concatenate([Xtab[j].astype(np.float32, copy=False),
                                 E[j].astype(np.float32, copy=False)], axis=1)
            probs_list.append(booster.predict(Xf, num_iteration=booster.best_iteration))

        probs = np.concatenate(probs_list, axis=0).astype(np.float32)
        ys.append(y[idx])
        ps.append(probs)

    y_all = np.concatenate(ys).astype(np.uint8)
    p_all = np.concatenate(ps).astype(np.float32)

    auc = float(roc_auc_score(y_all, p_all))
    print("Full test AUC:", auc, "N:", int(len(y_all)))

    # sweep thresholds
    thresholds = np.arange(args.tmin, args.tmax + 1e-12, args.tstep)
    rows = []
    best_f1 = (-1.0, None)
    best_fn = (10**18, None)

    for t in thresholds:
        pred = (p_all >= t).astype(np.uint8)

        tn = int(((pred == 0) & (y_all == 0)).sum())
        fp = int(((pred == 1) & (y_all == 0)).sum())
        fn = int(((pred == 0) & (y_all == 1)).sum())
        tp = int(((pred == 1) & (y_all == 1)).sum())

        acc, prec, rec, f1, fpr, fnr = metrics_from_cm(tn, fp, fn, tp)
        rows.append([t, acc, prec, rec, f1, fpr, fnr, tn, fp, fn, tp])

        if f1 > best_f1[0]:
            best_f1 = (f1, t)
        if fn < best_fn[0]:
            best_fn = (fn, t)

    out_csv = os.path.join(args.outdir, f"{args.dataset_tag}__threshold_sweep__{args.emb_kind}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["threshold","acc","precision","recall","f1","fpr","fnr","TN","FP","FN","TP"])
        w.writerows(rows)

    summary = {
        "dataset_tag": args.dataset_tag,
        "model_path": args.model_path,
        "emb_kind": args.emb_kind,
        "full_test_auc": auc,
        "best_f1": {"threshold": best_f1[1], "f1": best_f1[0]},
        "min_fn": {"threshold": best_fn[1], "fn": best_fn[0]},
        "csv": out_csv,
    }
    out_json = os.path.join(args.outdir, f"{args.dataset_tag}__threshold_sweep__{args.emb_kind}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Wrote:", out_csv)
    print("Wrote:", out_json)
    print("Best F1 threshold:", best_f1[1], "F1:", best_f1[0])
    print("Min FN threshold:", best_fn[1], "FN:", best_fn[0])

if __name__ == "__main__":
    main()
