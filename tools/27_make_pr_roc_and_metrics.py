#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y_true", required=True)
    ap.add_argument("--y_prob", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--name", default="model")
    ap.add_argument("--fpr_targets", nargs="+", type=float, default=[0.001, 0.01, 0.05])
    args = ap.parse_args()
    ensure_dir(args.outdir)

    y = np.load(args.y_true).astype(np.uint8)
    p = np.load(args.y_prob).astype(np.float32)

    # ROC
    fpr, tpr, thr = roc_curve(y, p)
    roc_auc = float(roc_auc_score(y, p))

    # PR
    prec, rec, thr_pr = precision_recall_curve(y, p)
    pr_auc = float(average_precision_score(y, p))  # AP (PR-AUC)

    
    # Save ROC plot (log-scale x-axis, EMBER-style)
    roc_png = os.path.join(args.outdir, f"ROC_{args.name}.png")
    
    target_fpr = 0.01
    mask = fpr <= target_fpr
    target_tpr = float(tpr[mask].max()) if mask.any() else 0.0
    
    plt.figure(figsize=(5.2, 4.2))
    plt.plot(fpr, tpr, color="black", lw=1.5)
    
    # dashed guides
    plt.plot([target_fpr, target_fpr], [0.65, target_tpr], "r--", lw=1)
    plt.plot([1e-4, target_fpr], [target_tpr, target_tpr], "r--", lw=1)
    
    plt.xscale("log")
    plt.xlim(1e-4, 1.0)
    plt.ylim(0.65, 1.02)
    
    plt.xlabel("False Positive Rate (log scale)")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for EMBER2024 Tab+ImgEmb LightGBM Model")
    
    plt.grid(True, which="both", linestyle="-", linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.savefig(roc_png, dpi=300)
    plt.close()
        
    # Save PR plot
    pr_png = os.path.join(args.outdir, f"PR_{args.name}.png")
    plt.figure(figsize=(6,6))
    plt.plot(rec, prec, lw=2, label=f"PR-AUC(AP)={pr_auc:.6f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve — {args.name}")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(pr_png, dpi=200)
    plt.close()

    # TPR at selected FPR targets
    tpr_at = {}
    for ft in args.fpr_targets:
        # first index where fpr <= ft? We want max TPR under fpr<=ft
        mask = fpr <= ft
        tpr_at[str(ft)] = float(tpr[mask].max()) if mask.any() else 0.0

    # Save arrays for reproducibility
    np.save(os.path.join(args.outdir, f"roc_fpr_{args.name}.npy"), fpr)
    np.save(os.path.join(args.outdir, f"roc_tpr_{args.name}.npy"), tpr)
    np.save(os.path.join(args.outdir, f"pr_precision_{args.name}.npy"), prec)
    np.save(os.path.join(args.outdir, f"pr_recall_{args.name}.npy"), rec)

    # Summary JSON
    summary = {
        "name": args.name,
        "n": int(len(y)),
        "pos_rate": float(y.mean()),
        "roc_auc": roc_auc,
        "pr_auc_ap": pr_auc,
        "tpr_at_fpr": tpr_at,
        "files": {
            "roc_png": roc_png,
            "pr_png": pr_png
        }
    }
    out_json = os.path.join(args.outdir, f"metrics_{args.name}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("ROC-AUC:", roc_auc)
    print("PR-AUC(AP):", pr_auc)
    print("TPR@FPR targets:", tpr_at)
    print("Wrote:", roc_png)
    print("Wrote:", pr_png)
    print("Wrote:", out_json)

if __name__ == "__main__":
    main()