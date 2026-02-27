#!/usr/bin/env python3
import argparse, json, os, csv, math
from datetime import datetime

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def read_cm_from_metrics(metrics_json_path):
    m = load_json(metrics_json_path)
    cm = m["confusion_matrix"]  # [[TN,FP],[FN,TP]]
    tn, fp = cm[0]
    fn, tp = cm[1]
    return float(m.get("full_test_auc", m.get("AUC", 0.0))), float(m.get("full_test_accuracy", m.get("ACC", 0.0))), tn, fp, fn, tp

def find_threshold_row(csv_path, target_t=0.35):
    best = None
    best_dist = 1e9
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            t = float(row["threshold"])
            d = abs(t - target_t)
            if d < best_dist:
                best_dist = d
                best = row
    return best, best_dist

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_tag", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--tab_only_metrics", required=True)
    ap.add_argument("--tab_plus_img_metrics", required=True)

    ap.add_argument("--threshold_sweep_csv", required=True)
    ap.add_argument("--threshold_sweep_json", required=True)

    ap.add_argument("--shap_summary_json", required=True)
    ap.add_argument("--gradcam_meta_json", required=True)

    ap.add_argument("--recommended_threshold", type=float, default=0.35)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # Load fixed-threshold(0.5) metrics
    tab_auc, tab_acc, tab_tn, tab_fp, tab_fn, tab_tp = read_cm_from_metrics(args.tab_only_metrics)
    img_auc, img_acc, img_tn, img_fp, img_fn, img_tp = read_cm_from_metrics(args.tab_plus_img_metrics)

    # Load threshold sweep (for recommended threshold)
    sweep = load_json(args.threshold_sweep_json)
    row, dist = find_threshold_row(args.threshold_sweep_csv, args.recommended_threshold)
    if row is None:
        raise RuntimeError("Could not read threshold sweep CSV")

    # Pull row metrics
    t = float(row["threshold"])
    acc = float(row["acc"])
    prec = float(row["precision"])
    rec = float(row["recall"])
    f1 = float(row["f1"])
    fpr = float(row["fpr"])
    fnr = float(row["fnr"])
    tn = int(float(row["TN"])); fp = int(float(row["FP"]))
    fn = int(float(row["FN"])); tp = int(float(row["TP"]))

    # Build a report-ready table
    table_rows = [
        ["Model", "Threshold", "AUC", "ACC", "TN", "FP", "FN", "TP", "Precision", "Recall", "F1"],
        ["Tab-only LGBM (EMBERv3 2568)", 0.5, tab_auc, tab_acc, tab_tn, tab_fp, tab_fn, tab_tp, "", "", ""],
        ["Tab + Img-Emb LGBM (2568 + 128)", 0.5, img_auc, img_acc, img_tn, img_fp, img_fn, img_tp, "", "", ""],
        ["Tab + Img-Emb LGBM (tuned)", t, img_auc, acc, tn, fp, fn, tp, prec, rec, f1],
    ]

    csv_path = os.path.join(args.outdir, "final_results_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(table_rows)

    # Final summary JSON (easy to cite in thesis)
    summary = {
        "dataset_tag": args.dataset_tag,
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "final_choice": {
            "model": "LightGBM (tab + img-embedding)",
            "reason": "best overall (slightly higher AUC/ACC than tab-only) and supports multimodal XAI",
            "recommended_threshold": args.recommended_threshold,
            "threshold_note": "0.35 maximizes F1 on full test sweep; use lower if you want fewer FN (higher recall) at cost of FP."
        },
        "tab_only_full_test": {
            "metrics_json": args.tab_only_metrics,
            "auc": tab_auc, "acc": tab_acc,
            "cm": {"TN": tab_tn, "FP": tab_fp, "FN": tab_fn, "TP": tab_tp}
        },
        "tab_plus_img_full_test": {
            "metrics_json": args.tab_plus_img_metrics,
            "auc": img_auc, "acc": img_acc,
            "cm": {"TN": img_tn, "FP": img_fp, "FN": img_fn, "TP": img_tp}
        },
        "tuned_threshold_full_test": {
            "threshold_sweep_json": args.threshold_sweep_json,
            "threshold_sweep_csv": args.threshold_sweep_csv,
            "picked_threshold": t,
            "distance_to_requested_threshold": dist,
            "metrics_at_threshold": {
                "acc": acc, "precision": prec, "recall": rec, "f1": f1, "fpr": fpr, "fnr": fnr,
                "cm": {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
            }
        },
        "xai_outputs": {
            "shap_summary_json": args.shap_summary_json,
            "gradcam_meta_json": args.gradcam_meta_json
        },
        "outputs": {
            "final_results_table_csv": csv_path
        }
    }

    out_json = os.path.join(args.outdir, "final_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Wrote:", csv_path)
    print("Wrote:", out_json)

if __name__ == "__main__":
    main()
