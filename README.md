# Predicting Emerging Topics from Outliers: A Prospective Study of Weak Signals in Embedding Space

Reproducibility materials for our EMNLP submission


The repository includes the released feature matrices, machine-learning result workbooks, agreement outputs, SHAP interpretation tables, and scripts used to rerun the supervised-stage experiments.

---

## Repository layout

```text
CONFIG/
  config.example.yaml

DATA/
  README.md
  collection_scripts/
    climatenewsfr_data_collection_googlenews.py
    climatenewsfr_data_collection_X.py
  climatenewsfr_article-url_target_features_all_k.xlsx
  hydronewsfr_article-url_target_features_all_k.xlsx

DOCS/
  TABLE_SCHEMAS.md

RESULTS/
  agreement/
    agreement_hdbscan_th30_d20.xlsx
    agreement_hdbscan_th30_d20_climat.xlsx
  climatenewsfr/
    climatenewsfr_results.xlsx
    climatenewsfr_interpretability_k660_xgboost.xlsx
  hydronewsfr/
    hydronewsfr_results.xlsx
    hydronewsfr_interpretability_k440_xgboost.xlsx

SCRIPTS/
  00_validate_inputs.py
  01_dynamic_topic_reconstruction.py
  02_trajectory_annotation_matrix.py
  03_calculate_agreement.py
  04_create_features_long_tables.py
  05_export_feature_matrix.py
  06_run_ml_experiments.py
  07_shap_oof_interpretation.py

APPENDIX/
  ml_feature_glossary.xlsx
  additional_experiments/

requirements.txt
environment.yml
run_all.sh
README.md
```

---

## Released feature matrices

```text
DATA/hydronewsfr_article-url_target_features_all_k.xlsx
DATA/climatenewsfr_article-url_target_features_all_k.xlsx
```

Each workbook contains a `feature_matrix` sheet. Rows correspond to article-threshold pairs.

Main columns:

| Column | Description |
|---|---|
| `article_url` | Article identifier used in the released matrix. |
| `agreement_k` | Diagonal consensus threshold, with `agreement_k = outlier_k = toa_k`. |
| `label_TOA` | Binary supervised label. `1` indicates a consensus anticipatory outlier; `0` indicates a confident non-anticipatory publication-time outlier. |
| `n_models_present` | Number of embedding-model representations available for the article. |

The remaining columns are the geometric, textual, named-entity, and social/co-sharing features used in the supervised models.

The released matrices include `agreement_k` values from 1 to 8.

---

## Machine-learning results

```text
RESULTS/hydronewsfr/hydronewsfr_results.xlsx
RESULTS/climatenewsfr/climatenewsfr_results.xlsx
```

Each workbook contains:

```text
ml_metrics_with_ablation
fold_ablation
fold_ablation_paired_tests
```

### `ml_metrics_with_ablation`

Cross-validated supervised-learning results across agreement thresholds.

The sheet uses the diagonal rule:

```text
outlier_k = toa_k = agreement_k
toa_max_neg = 0
```

Reported metrics:

```text
Precision
F1
Recall
```

Classifier labels:

| Label | Classifier |
|---|---|
| `xgb` | XGBoost |
| `rf` | Random Forest |
| `logreg` | Logistic Regression |
| `linear_svc` | Linear Support Vector Classifier |
| `dt` | Decision Tree |
| `baseline_all_pos` | Constant-positive baseline |

Ablation labels:

| Label | Description |
|---|---|
| `all_features` | Geometric, textual, and social features. |
| `no_geom` | All features except geometric features. |
| `no_social` | All features except social features. |
| `no_text` | All features except textual features. |
| `only_geom` | Geometric features only. |
| `only_social` | Social features only. |
| `only_text` | Textual features only. |
| `baseline` | Baseline row. |

The baseline is reported once per agreement threshold.

### `fold_ablation`

Fold-level XGBoost ablation results for the selected paper settings:

```text
HYDRONEWSFR:   k = 4 -> (4, 4, 0)
CLIMATENEWSFR: k = 6 -> (6, 6, 0)
```

This sheet contains one row per cross-validation fold and ablation setting.

### `fold_ablation_paired_tests`

Paired fold-level t-tests for the selected XGBoost ablation comparisons.

The sheet reports tests for:

```text
F1
Precision
Recall
```

The paper table uses significance symbols only for F1. Precision and recall tests are included to document whether the F1 changes are driven by one or both components.

Paper-symbol convention:

| Symbol | Meaning |
|---|---|
| `†` | Significant F1 drop relative to `all_features` after Benjamini-Hochberg correction. |
| `‡` | Significant F1 drop relative to `only_geom` after Benjamini-Hochberg correction. |

Detailed column descriptions are in:

```text
DOCS/TABLE_SCHEMAS.md
```

---

## Agreement outputs

```text
RESULTS/agreement/agreement_hdbscan_th30_d20.xlsx
RESULTS/agreement/agreement_hdbscan_th30_d20_climat.xlsx
```

These workbooks contain model-specific trajectory labels and agreement summaries.

The main sheet is:

```text
TOA matrix and agreement
```

It includes article identifiers, publication dates, model-specific trajectory assignments, anticipatory-vote counts, and publication-time outlier-vote counts.

---

## SHAP interpretation outputs

```text
RESULTS/hydronewsfr/hydronewsfr_interpretability_k440_xgboost.xlsx
RESULTS/climatenewsfr/climatenewsfr_interpretability_k660_xgboost.xlsx
```

The filename tag encodes the selected consensus rule:

| Tag | Meaning |
|---|---|
| `k440` | `outlier_k=4`, `toa_k=4`, `toa_max_neg=0` |
| `k660` | `outlier_k=6`, `toa_k=6`, `toa_max_neg=0` |

These workbooks contain the global and local SHAP outputs for the selected XGBoost models.

---

## Data collection scripts

```text
DATA/collection_scripts/climatenewsfr_data_collection_googlenews.py
DATA/collection_scripts/climatenewsfr_data_collection_X.py
```

These scripts document the collection procedure used for the CLIMATENEWSFR corpus. They are parameterized command-line scripts and do not execute at import time.

News articles were collected with Google News RSS/decoding utilities. X-sharing activity was collected through the X API.

Raw article text, raw social-media traces, and private collection files are not redistributed.

---

## Installation

Using a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download fr_core_news_md
```

Using Conda or Mamba:

```bash
mamba env create -f environment.yml
conda activate topic-outlier-reproduction
python -m spacy download fr_core_news_md
```

The French spaCy model is used for named-entity features.

---

## Rerun supervised-stage experiments

HYDRONEWSFR:

```bash
python SCRIPTS/06_run_ml_experiments.py \
  --feature-matrix DATA/hydronewsfr_article-url_target_features_all_k.xlsx \
  --thresholds 1 2 3 4 5 6 7 8 \
  --selected-ablation-k 4 \
  --output RESULTS/hydronewsfr/hydronewsfr_results_rerun.xlsx
```

CLIMATENEWSFR:

```bash
python SCRIPTS/06_run_ml_experiments.py \
  --feature-matrix DATA/climatenewsfr_article-url_target_features_all_k.xlsx \
  --thresholds 1 2 3 4 5 6 7 8 \
  --selected-ablation-k 6 \
  --output RESULTS/climatenewsfr/climatenewsfr_results_rerun.xlsx
```

Rerun values may differ slightly from the archived workbooks because some estimators include stochastic components.

Expected runtime depends on hardware. The supervised-stage ML scripts run from the released feature matrices and are substantially faster than the full trajectory-reconstruction pipeline. The full pipeline requires local article data, embeddings/API outputs, and may take considerably longer.

---

## Rerun SHAP interpretation

HYDRONEWSFR:

```bash
python SCRIPTS/07_shap_oof_interpretation.py \
  --feature-matrix DATA/hydronewsfr_article-url_target_features_all_k.xlsx \
  --agreement-k 4 \
  --dataset-name hydronewsfr \
  --output RESULTS/hydronewsfr/hydronewsfr_interpretability_k440_xgboost_rerun.xlsx
```

CLIMATENEWSFR:

```bash
python SCRIPTS/07_shap_oof_interpretation.py \
  --feature-matrix DATA/climatenewsfr_article-url_target_features_all_k.xlsx \
  --agreement-k 6 \
  --dataset-name climatenewsfr \
  --output RESULTS/climatenewsfr/climatenewsfr_interpretability_k660_xgboost_rerun.xlsx
```

---

## Pipeline scripts

| Script | Purpose |
|---|---|
| `00_validate_inputs.py` | Validate local input files. |
| `01_dynamic_topic_reconstruction.py` | Build cumulative daily topic snapshots. |
| `02_trajectory_annotation_matrix.py` | Build model-specific article trajectory labels. |
| `03_calculate_agreement.py` | Compute agreement-based consensus labels. |
| `04_create_features_long_tables.py` | Create publication-time feature tables. |
| `05_export_feature_matrix.py` | Export the article-threshold feature matrix. |
| `06_run_ml_experiments.py` | Run supervised models, ablations, and selected-setting paired ablation tests. |
| `07_shap_oof_interpretation.py` | Compute XGBoost SHAP interpretation outputs. |

---

## Main experimental settings

The main experiments use cumulative daily snapshots, UMAP with 20 dimensions, HDBSCAN clustering, centroid-based topic alignment with threshold `0.30`, and an ensemble of embedding models.

Selected agreement thresholds:

```text
HYDRONEWSFR:   outlier_k=4, toa_k=4, toa_max_neg=0
CLIMATENEWSFR: outlier_k=6, toa_k=6, toa_max_neg=0
```

The supervised task is evaluated at publication time.
