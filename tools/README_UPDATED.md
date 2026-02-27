# EMBER2024_CORE PE — Multimodal Malware Detection + XAI

This repository contains a **complete, reproducible malware-detection pipeline** for **EMBER2024_CORE/PE** (JSONL shards), including:
- **Feature caching** (tabular + section modalities)
- **Model training & full-test evaluation**
- **Threshold sweep (deployment-style tuning)**
- **Explainable AI (XAI)** with **SHAP** (LightGBM) and **Grad-CAM** (CNN section encoder)
- **Final packaging** (summary + results table)
- **Optional challenge-malicious evaluation** (malicious-only holdout)

> **Why “section images” instead of raw byteplots?**  
> The dataset JSONL schema reports `has_raw_bytes: False`, so raw `.exe` bytes are not available from these files.  
> Instead, we build a compact **section-structure image (17×24)** from PE **section + overlay** metadata.

---

## Repo location ( environment)

All commands in this README assume we run from:

```bash
cd /workspace/malware
```

Key folders:
- `tools/` — all pipeline scripts
- `resources/` — static resources (e.g., `pefile_warnings.txt`)
- `outputs/` — cached features, models, evaluations, XAI artifacts, and final summaries

---

## Dataset layout (EMBER2024_CORE)

Expected dataset folders:
- Train shards: `/workspace/EMBER2024_CORE/PE/train/` (**104** `*_train.jsonl`)
- Test shards:  `/workspace/EMBER2024_CORE/PE/test/`  (**24** `*_test.jsonl`)

Typical rows per shard:
- Win32: 60,000
- Win64: 20,000

Confirmed total train valid rows (from your run): **4,160,000**

---

## Final classification architecture

### Inputs (per file)
1) **Tabular vector**: `X_tab ∈ R^2568`  
   EMBERv3-compatible layout (matches THREMBER/EMBER2024 baseline feature packing).
2) **Section-structure image**: `X_section_img ∈ R^(17×24)`  
   16 sections + 1 overlay row (compact static structural view).
3) (Optional) **Section 1D vector**: `X_section_1d ∈ R^224`  
   Useful for training the section encoder (and for some ablations).

### Encoders + late fusion
- CNN section encoder → `emb_section_img128 ∈ R^128`
- Late fusion vector: `z = concat(X_tab, emb_section_img128) ∈ R^2696`

### Final classifier
- **LightGBM** binary classifier → `p(malware | z)`

### Operating threshold
- Default threshold: **0.50**
- Best-F1 (full test sweep): **0.35**

---

## Performance summary (from your completed runs)

### Full test set (N = 960,000)

**Tab-only LightGBM (EMBERv3 2568-d)**
- AUC: **0.9979385**
- ACC: **0.9794063**
- Confusion matrix `[[TN, FP], [FN, TP]]`:
  - `[[474204, 5796], [13974, 466026]]`

**Tab + Img-Embedding LightGBM (2568 + 128 = 2696-d)**
- AUC: **0.9979489**
- ACC: **0.9794833**
- Confusion matrix:
  - `[[474108, 5892], [13804, 466196]]`

**Threshold sweep (full test)**
- Best-F1 threshold: **0.35**
- Best F1: **0.9801136**

### Challenge malware (malicious-only holdout, N = 6,315)

Because the challenge set is **label=1 only**, AUC is not applicable.  
We report **detection rate (recall on malicious)**:

- detect_rate@0.35: **0.6553**
- detect_rate@0.50: **0.5907**

---

# Quickstart: run the full pipeline (A → D)

> All outputs are written under `outputs/` with descriptive run folders.

## Part A — Data & feature caches

### A0) Check schema (sanity)
```bash
python3 tools/00_check_dataset_schema.py \
  --train /workspace/EMBER2024_CORE/PE/train/2023-09-24_2023-09-30_Win32_train.jsonl \
  --test  /workspace/EMBER2024_CORE/PE/test/2024-09-22_2024-09-28_Win32_test.jsonl \
  --dataset_tag EMBER2024_CORE_PE \
  --outdir outputs/00_dataset_schema \
  --max_lines 50000
```

### A1) Build tabular feature vectors (EMBERv3 2568-d)
```bash
python3 tools/01_build_feature_vectors_emberv3.py \
  --train /workspace/EMBER2024_CORE/PE/train \
  --test  /workspace/EMBER2024_CORE/PE/test \
  --dataset_tag EMBER2024_CORE_PE \
  --outdir outputs/01_feature_vectors/EMBER2024_CORE_PE_emberv3 \
  --warnings_file resources/pefile_warnings.txt \
  --mode sharded
```

Per-shard outputs:
- `...__X_tab_emberv3.npy` (N×2568)
- `...__y.npy`, `...__sha256.npy`, `...__valid.npy`, `...__meta.json`

### A2) Build section modalities (image + 1D vector)
```bash
python3 tools/02_build_section_modalities.py \
  --train /workspace/EMBER2024_CORE/PE/train \
  --test  /workspace/EMBER2024_CORE/PE/test \
  --dataset_tag EMBER2024_CORE_PE \
  --outdir outputs/02_section_modalities/EMBER2024_CORE_PE_section_modalities_v1
```

Per-shard outputs:
- `...__X_section_img_v1.npy` (N×17×24)
- `...__X_section_1d_emberv3_section224.npy` (N×224)

### A3) Pair tab + section shards into a multimodal manifest
```bash
python3 tools/03_build_multimodal_manifest.py \
  --dataset_tag EMBER2024_CORE_PE \
  --tab_dir outputs/01_feature_vectors/EMBER2024_CORE_PE_emberv3 \
  --section_dir outputs/02_section_modalities/EMBER2024_CORE_PE_section_modalities_v1 \
  --outdir outputs/03_multimodal_manifests/EMBER2024_CORE_PE_mm_v1 \
  --sha_check --sha_check_n 512
```

Outputs:
- `...__multimodal_manifest__train.json` (paired=104)
- `...__multimodal_manifest__test.json` (paired=24)

### A4) Sanity check alignment (sha/label/valid)
```bash
python3 tools/04_sanity_check_multimodal_cache.py \
  --train_manifest outputs/03_multimodal_manifests/EMBER2024_CORE_PE_mm_v1/EMBER2024_CORE_PE__multimodal_manifest__train.json \
  --test_manifest  outputs/03_multimodal_manifests/EMBER2024_CORE_PE_mm_v1/EMBER2024_CORE_PE__multimodal_manifest__test.json \
  --print_n 1
```

(Optional) Section modality stats:
```bash
python3 tools/08_check_section_modalities_stats.py \
  --train_manifest outputs/03_multimodal_manifests/EMBER2024_CORE_PE_mm_v1/EMBER2024_CORE_PE__multimodal_manifest__train.json \
  --test_manifest  outputs/03_multimodal_manifests/EMBER2024_CORE_PE_mm_v1/EMBER2024_CORE_PE__multimodal_manifest__test.json \
  --outdir outputs/02_section_modalities/EMBER2024_CORE_PE_section_modalities_v1/stats
```

---

## Part B — Modeling (baseline + final)

### B1) Train tab-only LightGBM
```bash
python3 tools/06_train_lgbm_tab_emberv3.py \
  --train_manifest outputs/03_multimodal_manifests/EMBER2024_CORE_PE_mm_v1/EMBER2024_CORE_PE__multimodal_manifest__train.json \
  --test_manifest  outputs/03_multimodal_manifests/EMBER2024_CORE_PE_mm_v1/EMBER2024_CORE_PE__multimodal_manifest__test.json \
  --outdir outputs/05_lgbm_runs/EMBER2024_CORE_PE_tab_lgbm_emberv3_v1
```

### B2) Full-test evaluation (tab-only)
```bash
python3 tools/07_evaluate_lgbm_full_test_emberv3.py \
  --test_manifest outputs/03_multimodal_manifests/EMBER2024_CORE_PE_mm_v1/EMBER2024_CORE_PE__multimodal_manifest__test.json \
  --model_path outputs/05_lgbm_runs/EMBER2024_CORE_PE_tab_lgbm_emberv3_v1/lgbm_model.txt \
  --outdir outputs/05_lgbm_runs/EMBER2024_CORE_PE_tab_lgbm_emberv3_v1/full_test_evaluation
```

### B3) Train a CNN section encoder + export embeddings
Train:
```bash
python3 tools/09_train_and_export_section_embeddings.py train \
  --train_manifest outputs/03_multimodal_manifests/EMBER2024_CORE_PE_mm_v1/EMBER2024_CORE_PE__multimodal_manifest__train.json \
  --test_manifest  outputs/03_multimodal_manifests/EMBER2024_CORE_PE_mm_v1/EMBER2024_CORE_PE__multimodal_manifest__test.json \
  --dataset_tag EMBER2024_CORE_PE \
  --outdir outputs/06_section_encoder_runs/EMBER2024_CORE_PE_section_encoder_fusion_v1 \
  --device cuda \
  --epochs 3 \
  --batch_size 512 \
  --lr 3e-4 \
  --grad_clip 1.0 \
  --shuffle_shards
```

Export embeddings:
```bash
python3 tools/09_train_and_export_section_embeddings.py export \
  --train_manifest outputs/03_multimodal_manifests/EMBER2024_CORE_PE_mm_v1/EMBER2024_CORE_PE__multimodal_manifest__train.json \
  --test_manifest  outputs/03_multimodal_manifests/EMBER2024_CORE_PE_mm_v1/EMBER2024_CORE_PE__multimodal_manifest__test.json \
  --dataset_tag EMBER2024_CORE_PE \
  --checkpoint outputs/06_section_encoder_runs/EMBER2024_CORE_PE_section_encoder_fusion_v1/checkpoint_best.pt \
  --outdir outputs/07_section_embeddings/EMBER2024_CORE_PE_section_embeddings_v1 \
  --device cuda \
  --batch_size 1024
```

Embeddings per shard:
- `__emb_section_img128.npy`
- `__emb_section_1d128.npy`
- `__emb_section_cat256.npy`

### B4) Train final LightGBM late fusion (tab + img embedding)
```bash
python3 tools/10_train_lgbm_tab_plus_section_embeddings.py \
  --train_manifest outputs/03_multimodal_manifests/EMBER2024_CORE_PE_mm_v1/EMBER2024_CORE_PE__multimodal_manifest__train.json \
  --test_manifest  outputs/03_multimodal_manifests/EMBER2024_CORE_PE_mm_v1/EMBER2024_CORE_PE__multimodal_manifest__test.json \
  --emb_dir outputs/07_section_embeddings/EMBER2024_CORE_PE_section_embeddings_v1 \
  --dataset_tag EMBER2024_CORE_PE \
  --outdir outputs/08_lgbm_runs/EMBER2024_CORE_PE_tab_plus_imgemb_lgbm_v1 \
  --emb_kind emb_section_img
```

### B5) Full-test evaluation (tab + img embedding)
```bash
python3 tools/11_evaluate_lgbm_full_test_tab_plus_section_embeddings.py \
  --test_manifest outputs/03_multimodal_manifests/EMBER2024_CORE_PE_mm_v1/EMBER2024_CORE_PE__multimodal_manifest__test.json \
  --model_path outputs/08_lgbm_runs/EMBER2024_CORE_PE_tab_plus_imgemb_lgbm_v1/lgbm_model.txt \
  --emb_dir outputs/07_section_embeddings/EMBER2024_CORE_PE_section_embeddings_v1 \
  --dataset_tag EMBER2024_CORE_PE \
  --outdir outputs/08_lgbm_runs/EMBER2024_CORE_PE_tab_plus_imgemb_lgbm_v1/full_test_evaluation \
  --emb_kind emb_section_img \
  --threshold 0.5
```

### B6) Threshold sweep (full test)
```bash
python3 tools/12_threshold_sweep_lgbm_tab_plus_emb.py \
  --test_manifest outputs/03_multimodal_manifests/EMBER2024_CORE_PE_mm_v1/EMBER2024_CORE_PE__multimodal_manifest__test.json \
  --model_path outputs/08_lgbm_runs/EMBER2024_CORE_PE_tab_plus_imgemb_lgbm_v1/lgbm_model.txt \
  --emb_dir outputs/07_section_embeddings/EMBER2024_CORE_PE_section_embeddings_v1 \
  --dataset_tag EMBER2024_CORE_PE \
  --emb_kind emb_section_img \
  --outdir outputs/08_lgbm_runs/EMBER2024_CORE_PE_tab_plus_imgemb_lgbm_v1/threshold_sweep_full_test
```

---

## Part C — Explainability (XAI)

### C1) SHAP (LightGBM final model)
```bash
python3 tools/13_shap_explain_lgbm_tab_plus_imgemb.py \
  --test_manifest outputs/03_multimodal_manifests/EMBER2024_CORE_PE_mm_v1/EMBER2024_CORE_PE__multimodal_manifest__test.json \
  --model_path outputs/08_lgbm_runs/EMBER2024_CORE_PE_tab_plus_imgemb_lgbm_v1/lgbm_model.txt \
  --emb_dir outputs/07_section_embeddings/EMBER2024_CORE_PE_section_embeddings_v1 \
  --dataset_tag EMBER2024_CORE_PE \
  --emb_kind emb_section_img \
  --outdir outputs/09_xai/EMBER2024_CORE_PE_tab_plus_imgemb_lgbm_shap_v1
```

Artifacts:
- SHAP summary JSON (global feature importance + top features)
- Top-25 bar plot PNG
- Beeswarm plot PNG

### C2) Grad-CAM (CNN section encoder)
```bash
python3 tools/14_gradcam_section_encoder.py \
  --test_manifest outputs/03_multimodal_manifests/EMBER2024_CORE_PE_mm_v1/EMBER2024_CORE_PE__multimodal_manifest__test.json \
  --checkpoint outputs/06_section_encoder_runs/EMBER2024_CORE_PE_section_encoder_fusion_v1/checkpoint_best.pt \
  --outdir outputs/09_xai/EMBER2024_CORE_PE_section_encoder_gradcam_v1 \
  --img_key X_section_img_v1 \
  --sec1d_key X_section_1d_emberv3_section224 \
  --n_per_class 4 \
  --device cuda
```

Artifacts:
- `*__gradcam.png` (heatmaps)
- `*__gradcam.npz` (raw CAM + input)
- `gradcam_run_meta.json` (how to interpret rows/cols)

---

## Part D — Final packaging + inference

### D1) Compile final results
```bash
python3 tools/15_compile_final_results.py \
  --dataset_tag EMBER2024_CORE_PE \
  --outdir outputs/10_final_results/EMBER2024_CORE_PE_final_v1 \
  --recommended_threshold 0.35
```

Outputs:
- `final_results_table.csv`
- `final_summary.json`

### D2) Inference CLI (example)
```bash
python3 tools/16_infer_tab_plus_imgemb.py \
  --dataset_tag EMBER2024_CORE_PE \
  --model_path outputs/08_lgbm_runs/EMBER2024_CORE_PE_tab_plus_imgemb_lgbm_v1/lgbm_model.txt \
  --tab_dir outputs/01_feature_vectors/EMBER2024_CORE_PE_emberv3 \
  --emb_dir outputs/07_section_embeddings/EMBER2024_CORE_PE_section_embeddings_v1 \
  --split test \
  --base 2024-09-22_2024-09-28_Win32_test \
  --idx 15566 \
  --threshold 0.35
```

---

# Optional Part E — Challenge malware evaluation (malicious-only)

The challenge files live at:
- `/workspace/EMBER2024_CORE/*_challenge_malicious.jsonl`

### E1) Build tab vectors for challenge
```bash
python3 tools/17_build_challenge_tab_vectors_emberv3.py \
  --challenge_dir /workspace/EMBER2024_CORE \
  --pattern "*_challenge_malicious.jsonl" \
  --dataset_tag EMBER2024_CORE_PE \
  --outdir outputs/11_challenge_malicious/EMBER2024_CORE_PE_challenge_malicious_tab_emberv3_v1 \
  --warnings_file resources/pefile_warnings.txt
```

### E2) Score challenge with tab-only LightGBM
```bash
python3 tools/18_score_challenge_malicious_tab_lgbm.py \
  --challenge_manifest outputs/11_challenge_malicious/EMBER2024_CORE_PE_challenge_malicious_tab_emberv3_v1/EMBER2024_CORE_PE__challenge_malicious_manifest.json \
  --model_path outputs/05_lgbm_runs/EMBER2024_CORE_PE_tab_lgbm_emberv3_v1/lgbm_model.txt \
  --outdir outputs/11_challenge_malicious/EMBER2024_CORE_PE_challenge_malicious_tab_lgbm_eval_v1 \
  --thresholds 0.35 0.5
```

Outputs:
- `challenge_malicious_summary_by_file.csv`
- `challenge_malicious_lowest_prob_examples.csv`
- `challenge_malicious_overall_summary.json`

---

# explanation (what each tool does, and why)

Below is the “why” behind each code.

## 00_check_dataset_schema.py
**Purpose:** Confirm the JSONL structure (keys, label fields, presence/absence of raw bytes), and detect any shape mismatches early.  
**Why needed:** A single schema mismatch (missing keys, wrong label key) can silently break a large pipeline. This script prevents wasted training time.

## 01_build_feature_vectors_emberv3.py
**Purpose:** Convert each JSONL sample into a fixed-size **2568-d** tab vector (EMBERv3-compatible).  
**Why needed:** LightGBM requires a consistent numeric matrix. This also ensures compatibility with baseline feature layouts used in EMBER2024/THREMBER.

## 02_build_section_modalities.py
**Purpose:** Build additional **section-based modalities**:
- a compact **section image** (17×24)
- a **section 1D** vector (224-d)
**Why needed:** This creates the image-like structural view for CNN encoding, giving a second modality beyond tab features.

## 03_build_multimodal_manifest.py
**Purpose:** Pair the tab cache and section cache per shard and write a manifest that lists all file paths.  
**Why needed:** Avoids manual file bookkeeping and guarantees correct alignment across modalities.

## 04_sanity_check_multimodal_cache.py
**Purpose:** Verify alignment (`sha256`, labels, valid masks, shapes) between modalities.  
**Why needed:** Prevents training/eval on mismatched samples (a common silent bug in multimodal systems).

## 05_train_multimodal_fusion.py (experimental)
**Purpose:** End-to-end neural training (tab encoder + CNN + fusion head).  
**Why included:** Used for research exploration. In practice it can be sensitive to AMP/numerical scaling; the final best system uses stable late fusion via embeddings + LightGBM.

## 06_train_lgbm_tab_emberv3.py
**Purpose:** Train a strong tab-only LightGBM baseline using the tab vectors.  
**Why needed:** LightGBM is a known strong performer for engineered tabular PE features and gives a reliable baseline.

## 07_evaluate_lgbm_full_test_emberv3.py
**Purpose:** Run full-test evaluation (all test shards) for the tab-only model and write a full metrics report.  
**Why needed:** “Sample evaluation” can be misleading; this gives a definitive full-test result.

## 08_check_section_modalities_stats.py
**Purpose:** Compute distribution stats for section modalities (sanity: ranges, sparsity, NaN checks).  
**Why needed:** Helps diagnose why a CNN encoder might fail (bad scaling, constant features).

## 09_train_and_export_section_embeddings.py
**Purpose:** Train a CNN encoder on the section modalities, then **export embeddings** for every shard.  
**Why needed:** Embeddings let us do stable late fusion: learn a compact structural signature with CNN, then hand it to LightGBM.

## 10_train_lgbm_tab_plus_section_embeddings.py
**Purpose:** Train LightGBM on `[tab + embeddings]` (late fusion).  
**Why needed:** Combines the strength of LightGBM on tabular features with extra structural signal from CNN embeddings.

## 11_evaluate_lgbm_full_test_tab_plus_section_embeddings.py
**Purpose:** Full-test evaluation for the late-fusion model.  
**Why needed:** Confirms the multimodal model generalizes at scale.

## 12_threshold_sweep_lgbm_tab_plus_emb.py
**Purpose:** Sweep thresholds and compute metrics to select an operating point (e.g., best F1).  
**Why needed:** SOC deployment needs a policy threshold; 0.5 is not always optimal.

## 13_shap_explain_lgbm_tab_plus_imgemb.py
**Purpose:** Produce SHAP explanations for LightGBM (global + local), and save plots/JSON.  
**Why needed:** Provides analyst-friendly evidence: “which features pushed the decision”.

## 14_gradcam_section_encoder.py
**Purpose:** Produce Grad-CAM heatmaps for the CNN section encoder.  
**Why needed:** Visual structural evidence: “where the CNN looked” in the section image.

## 15_compile_final_results.py
**Purpose:** Aggregate key results into one place and write:
- `final_results_table.csv`
- `final_summary.json`
**Why needed:** Makes reporting easy and prevents copying mistakes across papers/slides.

## 16_infer_tab_plus_imgemb.py
**Purpose:** Single-sample inference CLI for the final model (by shard base + row index).  
**Why needed:** Demonstration tool for reviewers and for quick debugging.

## 17_build_challenge_tab_vectors_emberv3.py (optional)
**Purpose:** Build tab vectors for `*_challenge_malicious.jsonl`.  
**Why needed:** Challenge is separate from train/test; we keep it isolated and reproducible.

## 18_score_challenge_malicious_tab_lgbm.py (optional)
**Purpose:** Score challenge malware with a trained model and report detection rates.  
**Why needed:** Challenge is malicious-only, so the correct metric is detection rate at chosen thresholds.

---

## Notes / troubleshooting

- You may see: `Applied workaround for CuDNN issue, install nvrtc.so`  
  This is an environment warning and does not stop training/inference.

- Disk usage is large because caches are stored per-shard:
  - Tab vectors are stored as memmapped `.npy` for scalability.

- If end-to-end fusion training shows NaNs (script 05), prefer late fusion (embeddings + LightGBM), which is the final stable design used here.

---
