# Can Topic-Model Outliers Predict Future Topics?
## A prospective study of weak signals in embedding space.

This repository accompanies the EMNLP submission. It contains the scripts and review workbooks needed to reproduce the paper workflow on a new news corpus and to inspect the article-level feature/target tables used in the reported experiments.

The main pipeline reconstructs cumulative dynamic topics, derives trajectory-based labels, calculates cross-model agreement, creates publication-time features, runs supervised models, and exports out-of-fold SHAP interpretation tables.

## Repository structure

```text
CONFIG/
  config.example.yaml

SCRIPTS/
  00_validate_inputs.py
  01_dynamic_topic_reconstruction.py
  02_trajectory_annotation_matrix.py
  03_calculate_agreement.py
  04_create_features_long_tables.py
  05_export_feature_matrix.py
  06_run_ml_experiments.py
  07_shap_oof_interpretation.py

DOCS/
  TABLE_SCHEMAS.md

DATA/
  README.md
  climatenewsfr_article-url_target_features_all_k.xlsx
  hydronewsfr_article-url_target_features_all_k.xlsx

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

APPENDIX/
  ml_feature_glossary.xlsx
  additional_experiments/
    climatenewsfr_toa_correlations_by_model.xlsx
    hydronewsfr_toa_correlations_by_model.xlsx
    hydronewsfr_extended_results.xlsx
    hydronewsfr_extended_chronological_eval_k440.xlsx

requirements.txt
environment.yml
run_all.sh
README.md
```

## What is included

`DATA/` contains the shareable article-level workbooks used for review. These include article URLs, agreement thresholds, target labels, and final model features. They allow reviewers to inspect which URLs enter the supervised experiments and to rerun the modeling and interpretation stages.

`RESULTS/` contains aggregate model results, agreement matrices, and SHAP interpretation workbooks corresponding to the paper experiments.

`APPENDIX/` contains the feature glossary and additional appendix workbooks, including correlation diagnostics and the extended hydrogen chronological robustness output.

Raw article text, raw collection files, and raw social-media traces are not redistributed here because they may be subject to publisher, API, or platform restrictions. The scripts support reproduction on a local corpus with the schema documented in `DOCS/TABLE_SCHEMAS.md`.

## Setup

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

The French spaCy model is recommended for named-entity features. If it is not installed, the feature script falls back to zero named-entity counts.

## Reproducing the full pipeline on a new corpus

Copy the example configuration and edit the input paths:

```bash
cp CONFIG/config.example.yaml CONFIG/my_corpus.yaml
```

At minimum, the article table must contain:

- a stable article identifier or canonical URL;
- a publication date;
- a title;
- a lead paragraph, description, or body-text field.

Social-sharing data are optional. If the configured sharing file is missing, social features are zero-filled.

Run the full script sequence:

```bash
bash run_all.sh CONFIG/my_corpus.yaml
```

The example config assumes local private inputs under `DATA/private/`. These files are intentionally not included in the public repository.

## Script sequence

### 0. Validate inputs

```bash
python SCRIPTS/00_validate_inputs.py --config CONFIG/my_corpus.yaml
```

Checks required columns, date parsing, optional social-sharing overlap, and configured precomputed embeddings.

### 1. Dynamic topic reconstruction

```bash
python SCRIPTS/01_dynamic_topic_reconstruction.py --config CONFIG/my_corpus.yaml
```

For each embedding model, this writes:

```text
RESULTS/<corpus>/models/<model>/results.csv
```

The file contains cumulative snapshot assignments, aligned topic IDs, UMAP coordinates, HDBSCAN outlier indicators, and outlier scores.

### 2. Trajectory annotation matrix

```bash
python SCRIPTS/02_trajectory_annotation_matrix.py \
  --config CONFIG/my_corpus.yaml \
  --output RESULTS/my_corpus/trajectory_matrix.xlsx
```

Builds the model-by-model trajectory matrix used for agreement and label reconstruction.

### 3. Agreement and consensus labels

```bash
python SCRIPTS/03_calculate_agreement.py \
  --matrix RESULTS/my_corpus/trajectory_matrix.xlsx \
  --max-k 11 \
  --toa-max-neg 0 \
  --output RESULTS/my_corpus/agreement_labels.xlsx
```

For each agreement threshold `k`, positives require at least `k` anticipatory votes. Negatives require publication-time outlier support and zero anticipatory votes in the paper setting.

### 4. Publication-time features and long tables

```bash
python SCRIPTS/04_create_features_long_tables.py --config CONFIG/my_corpus.yaml
```

Outputs:

```text
RESULTS/<corpus>/feature_long_model_level.csv
RESULTS/<corpus>/feature_article_level.csv
```

Geometric features are computed at article-model level and aggregated across embedding models. Text and social features are article-level.

### 5. Export supervised feature matrix

```bash
python SCRIPTS/05_export_feature_matrix.py \
  --features RESULTS/my_corpus/feature_article_level.csv \
  --labels RESULTS/my_corpus/agreement_labels.xlsx \
  --id-col media_url \
  --max-k 11 \
  --output RESULTS/my_corpus/article_url_target_features_all_k.xlsx
```

Exports the shareable supervised-learning table with URL, threshold, target, and final features.

### 6. Supervised ML experiments

```bash
python SCRIPTS/06_run_ml_experiments.py \
  --feature-matrix RESULTS/my_corpus/article_url_target_features_all_k.xlsx \
  --thresholds 1 2 3 4 5 6 7 8 \
  --output RESULTS/my_corpus/results.xlsx
```

Evaluates Logistic Regression, Linear SVC, Decision Tree, Random Forest, XGBoost, constant baselines, and feature-family ablations.

### 7. SHAP interpretation with out-of-fold predictions

```bash
python SCRIPTS/07_shap_oof_interpretation.py \
  --feature-matrix RESULTS/my_corpus/article_url_target_features_all_k.xlsx \
  --agreement-k 4 \
  --dataset-name MyCorpus \
  --output RESULTS/my_corpus/interpretability_k440_xgboost.xlsx
```

Use the selected corpus-specific agreement threshold. The hydrogen main setting uses `k=4`; the climate main setting uses `k=6`.

## Rerunning only the supervised stage from the shared workbooks

The shared feature matrices in `DATA/` can be used directly for scripts 6 and 7.

Hydrogen example:

```bash
python SCRIPTS/06_run_ml_experiments.py \
  --feature-matrix DATA/hydronewsfr_article-url_target_features_all_k.xlsx \
  --thresholds 1 2 3 4 5 6 7 8 \
  --output RESULTS/hydronewsfr/hydronewsfr_results_rerun.xlsx

python SCRIPTS/07_shap_oof_interpretation.py \
  --feature-matrix DATA/hydronewsfr_article-url_target_features_all_k.xlsx \
  --agreement-k 4 \
  --dataset-name hydronewsfr \
  --output RESULTS/hydronewsfr/hydronewsfr_interpretability_k440_xgboost_rerun.xlsx
```

Climate example:

```bash
python SCRIPTS/06_run_ml_experiments.py \
  --feature-matrix DATA/climatenewsfr_article-url_target_features_all_k.xlsx \
  --thresholds 1 2 3 4 5 6 7 8 \
  --output RESULTS/climatenewsfr/climatenewsfr_results_rerun.xlsx

python SCRIPTS/07_shap_oof_interpretation.py \
  --feature-matrix DATA/climatenewsfr_article-url_target_features_all_k.xlsx \
  --agreement-k 6 \
  --dataset-name climatenewsfr \
  --output RESULTS/climatenewsfr/climatenewsfr_interpretability_k660_xgboost_rerun.xlsx
```

## Paper settings

The main paper setting uses cumulative daily snapshots, UMAP with 20 dimensions, HDBSCAN clustering, centroid alignment threshold `0.30`, and an ensemble of embedding models. The supervised task is evaluated at publication time (`TA`).

Main agreement thresholds:

- HYDRONEWSFR: `k=4`
- CLIMATENEWSFR: `k=6`

The extended hydrogen chronological robustness output is included under `APPENDIX/additional_experiments/`.

## Expected outputs from a new run

```text
RESULTS/<corpus>/models/<model>/results.csv
RESULTS/<corpus>/trajectory_matrix.xlsx
RESULTS/<corpus>/agreement_labels.xlsx
RESULTS/<corpus>/feature_long_model_level.csv
RESULTS/<corpus>/feature_article_level.csv
RESULTS/<corpus>/article_url_target_features_all_k.xlsx
RESULTS/<corpus>/results.xlsx
RESULTS/<corpus>/interpretability_k*_xgboost.xlsx
```

## Reproducibility notes

`random_state` defaults to `42`. Exact numerical reproduction requires the same raw corpus, cached embeddings, package versions, preprocessing choices, and configuration. API-based embeddings should be provided as precomputed files rather than regenerated during review.
