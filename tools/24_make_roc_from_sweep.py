#!/usr/bin/env python3
import argparse
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_csv", required=True, help="Threshold sweep CSV (must include TN,FP,FN,TP,threshold)")
    ap.add_argument("--sweep_json", required=True, help="Sweep JSON containing full_test_roc_auc")
    ap.add_argument("--outdir", required=True, help="Where to write ROC PNGs and ROC points CSV")
    ap.add_argument("--prefix", default="ROC_curve", help="Output file prefix")
    ap.add_argument("--title", default="ROC Curve (from threshold sweep)")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # Load sweep
    df = pd.read_csv(args.sweep_csv)
    required = {"threshold", "TN", "FP", "FN", "TP"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in sweep CSV: {missing}. Found: {list(df.columns)}")

    tn = df["TN"].astype(float)
    fp = df["FP"].astype(float)
    fn = df["FN"].astype(float)
    tp = df["TP"].astype(float)

    # Compute ROC points
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)

    # Load AUC from JSON (exact value from your evaluation)
    with open(args.sweep_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    auc = float(meta.get("full_test_roc_auc"))

    # Build points table
    pts = pd.DataFrame({
        "threshold": df["threshold"].astype(float),
        "fpr": fpr,
        "tpr": tpr
    }).replace([np.inf, -np.inf], np.nan).dropna()

    # Add endpoints for a complete ROC plot
    pts = pd.concat([pts,
                     pd.DataFrame([{"threshold": 1.0, "fpr": 0.0, "tpr": 0.0},
                                   {"threshold": 0.0, "fpr": 1.0, "tpr": 1.0}])],
                    ignore_index=True)

    # Sort by FPR then enforce monotone TPR (safe for stepped sweep curves)
    pts = pts.sort_values(["fpr", "tpr"]).reset_index(drop=True)
    pts["tpr_monotone"] = np.maximum.accumulate(pts["tpr"].to_numpy())

    # Save ROC points
    roc_pts_csv = os.path.join(args.outdir, f"{args.prefix}_points.csv")
    pts_out = pts[["threshold", "fpr", "tpr_monotone"]].rename(columns={"tpr_monotone": "tpr"})
    pts_out.to_csv(roc_pts_csv, index=False)

    # Plot (linear)
    out_linear = os.path.join(args.outdir, f"{args.prefix}_linear.png")
    plt.figure(figsize=(6, 6))
    plt.plot(pts["fpr"], pts["tpr_monotone"], lw=2, label=f"AUC={auc:.6f}")
    plt.plot([0, 1], [0, 1], ls="--", lw=1, label="Random baseline")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(args.title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_linear, dpi=200)
    plt.close()

    # Plot (log-FPR)
    out_log = os.path.join(args.outdir, f"{args.prefix}_logfpr.png")
    eps = 1e-6
    plt.figure(figsize=(6, 6))
    plt.plot(np.maximum(pts["fpr"], eps), pts["tpr_monotone"], lw=2, label=f"AUC={auc:.6f}")
    plt.xscale("log")
    plt.xlabel("False Positive Rate (FPR, log scale)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(args.title + " — log FPR")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_log, dpi=200)
    plt.close()

    # Print summary
    print("=== ROC from threshold sweep ===")
    print("Sweep CSV:", args.sweep_csv)
    print("Sweep JSON:", args.sweep_json)
    print("AUC (from JSON):", auc)
    print("Rows in sweep:", len(df))
    print("FPR range:", float(pts['fpr'].min()), "to", float(pts['fpr'].max()))
    print("TPR range:", float(pts['tpr_monotone'].min()), "to", float(pts['tpr_monotone'].max()))
    print("Wrote:", out_linear)
    print("Wrote:", out_log)
    print("Wrote:", roc_pts_csv)


if __name__ == "__main__":
    main()