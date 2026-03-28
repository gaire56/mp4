#!/usr/bin/env python3
import argparse, json, os
import numpy as np

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_json(p):
    with open(p,"r",encoding="utf-8") as f:
        return json.load(f)

def try_import_lightgbm():
    try:
        import lightgbm as lgb
        return lgb
    except Exception as e:
        raise RuntimeError("Install lightgbm: pip install lightgbm\n"+str(e))

def parse_mm_manifest(manifest_path):
    mf = load_json(manifest_path)
    shards=[]
    for it in mf["paired_items"]:
        tab=it["tab"]
        shards.append({"base": it["base"], "X": tab["X"], "y": tab["y"], "valid": tab["valid"]})
    return shards

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_manifest", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--chunk_size", type=int, default=200000)
    args = ap.parse_args()
    ensure_dir(args.outdir)

    lgb = try_import_lightgbm()
    booster = lgb.Booster(model_file=args.model_path)

    shards = parse_mm_manifest(args.test_manifest)
    ys, ps = [], []

    for sh in shards:
        X = np.load(sh["X"], mmap_mode="r")
        y = np.load(sh["y"], mmap_mode="r").astype(np.uint8)
        v = np.load(sh["valid"], mmap_mode="r").astype(np.uint8)
        idx = np.where(v==1)[0]
        if idx.size==0:
            continue
        probs_list=[]
        for s in range(0, idx.size, args.chunk_size):
            j = idx[s:s+args.chunk_size]
            probs_list.append(booster.predict(X[j], num_iteration=booster.best_iteration))
        probs = np.concatenate(probs_list).astype(np.float32)
        ys.append(y[idx]); ps.append(probs)

    y_all = np.concatenate(ys).astype(np.uint8)
    p_all = np.concatenate(ps).astype(np.float32)

    y_path = os.path.join(args.outdir, "full_test_y_true.npy")
    p_path = os.path.join(args.outdir, "full_test_y_prob.npy")
    np.save(y_path, y_all)
    np.save(p_path, p_all)

    print("Wrote:", y_path)
    print("Wrote:", p_path)
    print("N:", len(y_all), "pos_rate:", float(y_all.mean()))

if __name__ == "__main__":
    main()