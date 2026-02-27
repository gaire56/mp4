#!/usr/bin/env python3
import argparse, json, os
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

def try_import_shap():
    try:
        import shap
        return shap
    except Exception as e:
        raise RuntimeError("Install shap: pip install shap\n" + str(e))

def parse_test_items(test_manifest):
    mf = load_json(test_manifest)
    out = []
    for it in mf["paired_items"]:
        base = it["base"]
        tab = it["tab"]
        out.append({"base": base, "X": tab["X"], "y": tab["y"], "valid": tab["valid"]})
    return out

def find_emb_file(emb_split_dir, dataset_tag, base, emb_kind):
    # e.g. EMBER2024_CORE_PE__...__emb_section_img128.npy
    import glob
    pat = os.path.join(emb_split_dir, f"{dataset_tag}__{base}__{emb_kind}*.npy")
    hits = sorted(glob.glob(pat))
    if not hits:
        raise FileNotFoundError(f"Embedding file not found: {pat}")
    return hits[0]

def build_feature_names(tab_dim: int, emb_dim: int):
    names = [f"tab_{i}" for i in range(tab_dim)]
    names += [f"emb_img_{i}" for i in range(emb_dim)]
    return names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_manifest", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--dataset_tag", required=True)
    ap.add_argument("--emb_kind", default="emb_section_img")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--n_background", type=int, default=2000, help="background samples for SHAP")
    ap.add_argument("--n_explain", type=int, default=5000, help="samples to explain")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk_size", type=int, default=200000)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    lgb = try_import_lightgbm()
    shap = try_import_shap()

    booster = lgb.Booster(model_file=args.model_path)
    items = parse_test_items(args.test_manifest)
    emb_split_dir = os.path.join(args.emb_dir, "test")

    rng = np.random.default_rng(args.seed)

    # Collect a pool of samples from shards (valid rows only)
    X_pool = []
    y_pool = []

    for it in items:
        base = it["base"]
        Xtab = np.load(it["X"], mmap_mode="r")
        y = np.load(it["y"], mmap_mode="r").astype(np.uint8)
        v = np.load(it["valid"], mmap_mode="r").astype(np.uint8)
        idx = np.where(v == 1)[0]
        if idx.size == 0:
            continue

        emb_path = find_emb_file(emb_split_dir, args.dataset_tag, base, args.emb_kind)
        E = np.load(emb_path, mmap_mode="r")

        # take a small random subset from this shard
        take_n = min(2000, idx.size)
        take = rng.choice(idx, size=take_n, replace=False)
        take.sort()

        Xf = np.concatenate([Xtab[take].astype(np.float32, copy=False),
                             E[take].astype(np.float32, copy=False)], axis=1)
        X_pool.append(Xf)
        y_pool.append(y[take])

        # stop when enough
        if sum(x.shape[0] for x in X_pool) >= max(args.n_background, args.n_explain):
            break

    X_pool = np.concatenate(X_pool, axis=0)
    y_pool = np.concatenate(y_pool, axis=0)

    # sample background + explain sets
    n_total = X_pool.shape[0]
    n_bg = min(args.n_background, n_total)
    n_ex = min(args.n_explain, n_total)

    perm = rng.permutation(n_total)
    bg_idx = perm[:n_bg]
    ex_idx = perm[:n_ex]

    X_bg = X_pool[bg_idx]
    X_ex = X_pool[ex_idx]
    y_ex = y_pool[ex_idx]

    tab_dim = 2568
    emb_dim = X_bg.shape[1] - tab_dim
    feat_names = build_feature_names(tab_dim, emb_dim)

    # SHAP for tree model
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_ex)  # binary -> array
    # shap_values shape: (n_ex, n_features)

    shap_abs_mean = np.mean(np.abs(shap_values), axis=0)
    top_idx = np.argsort(-shap_abs_mean)[:50]

    top_features = []
    for i in top_idx:
        top_features.append({
            "feature": feat_names[int(i)],
            "mean_abs_shap": float(shap_abs_mean[int(i)]),
        })

    # save summary json
    out = {
        "dataset_tag": args.dataset_tag,
        "model_path": args.model_path,
        "emb_kind": args.emb_kind,
        "n_background": int(n_bg),
        "n_explain": int(n_ex),
        "tab_dim": tab_dim,
        "emb_dim": int(emb_dim),
        "top50_mean_abs_shap": top_features,
    }

    out_json = os.path.join(args.outdir, f"{args.dataset_tag}__shap_summary__tab_plus_{args.emb_kind}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Save plots
    import matplotlib.pyplot as plt

    # bar plot (top 25)
    top25 = top_features[:25]
    names = [d["feature"] for d in top25][::-1]
    vals = [d["mean_abs_shap"] for d in top25][::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(names, vals)
    plt.title("Top 25 SHAP mean(|value|) — LGBM (tab + img-emb)")
    plt.tight_layout()
    bar_path = os.path.join(args.outdir, f"{args.dataset_tag}__shap_top25_bar__tab_plus_{args.emb_kind}.png")
    plt.savefig(bar_path, dpi=200)
    plt.close()

    # SHAP beeswarm (may be heavy but OK at n_explain=5000)
    # Create SHAP Explanation object for nicer plots
    try:
        exp = shap.Explanation(values=shap_values, data=X_ex, feature_names=feat_names)
        plt.figure()
        shap.plots.beeswarm(exp, max_display=25, show=False)
        bee_path = os.path.join(args.outdir, f"{args.dataset_tag}__shap_beeswarm_top25__tab_plus_{args.emb_kind}.png")
        plt.tight_layout()
        plt.savefig(bee_path, dpi=200)
        plt.close()
    except Exception:
        bee_path = None

    print("Wrote:", out_json)
    print("Wrote:", bar_path)
    if bee_path:
        print("Wrote:", bee_path)
    else:
        print("Beeswarm plot skipped (plotting issue).")

if __name__ == "__main__":
    main()
