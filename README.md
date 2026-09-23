# Predicting Emerging Topics from Outliers: A Prospective Study of Weak Signals in Embedding Space

Code for the paper accepted at AACL-IJCNLP 2026 (Findings).

The repository contains the pipeline scripts, configuration, dependencies, data-collection scripts, and documentation needed to run the method on a news corpus.

Data are not included. Article text, social-media traces, embeddings, feature matrices, and result workbooks are not redistributed because they may be subject to publisher, API, or platform restrictions. They can be shared for research purposes upon reasonable request.

## Structure

```text
APPENDIX/
  ml_feature_glossary.xlsx        Definition of every feature used by the models
CONFIG/
  config.example.yaml             Example configuration (paths, clustering, labeling, ML)
DATA/
  README.md
  collection_scripts/
    climatenewsfr_data_collection_googlenews.py
    climatenewsfr_data_collection_X.py
DOCS/
  TABLE_SCHEMAS.md                Column descriptions of the result workbooks
SCRIPTS/
  00_validate_inputs.py
  01_dynamic_topic_reconstruction.py
  02_trajectory_annotation_matrix.py
  03_calculate_agreement.py
  04_create_features_long_tables.py
  05_export_feature_matrix.py
  06_run_ml_experiments.py
  07_shap_oof_interpretation.py
  08_forward_chaining.py
environment.yml
requirements.txt
run_all.sh
```

`DATA/private/` (input data) and `RESULTS/` (outputs) are created locally and are not tracked.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download fr_core_news_md
```

or

```bash
mamba env create -f environment.yml
conda activate topic-outlier-reproduction
python -m spacy download fr_core_news_md
```

The French spaCy model is required for named-entity features. Experiments were run with Python 3.12 and the versions pinned in `requirements.txt`.

## Input data

Paths and column names are defined in `CONFIG/config.example.yaml`. Copy and edit it for your corpus.

### Article table (required)

`input.articles`, default `DATA/private/articles.csv` (`.csv` or `.xlsx`), one row per article.

| Config key | Default column | Content |
|---|---|---|
| `article_id_col` | `media_url` | Unique article identifier |
| `url_col` | `media_url` | Article URL, joined with the share table |
| `date_col` | `publication_date_cleaned` | Publication date |
| `title_col` | `title` | Title |
| `text_col` | `description` | Lead paragraph |

Embedded text is built from `embedding.text_template` (default `"{title} {description}"`).

### Share table (optional)

`input.shares`, default `DATA/private/social_shares.csv`, one row per share. If absent, social features are set to zero.

| Column | Content |
|---|---|
| `media_url`, `article_url`, or `url` | Shared article URL |
| `created_at` | Share timestamp; only shares before the evaluation time are used |
| `author_id` or `user_id` | Sharing account |
| `followers_count`, `tweet_count`, `listed_count` | Optional account metrics (`user_public_metrics_` prefix accepted) |

### Embeddings

Models of type `sentence_transformer` are computed locally. For API models, set `type: precomputed` and place one file per model in `embedding.precomputed_dir` (default `DATA/private/embeddings/<short_name>.npy` or `.csv`), rows aligned with the article table.

## Pipeline

```bash
bash run_all.sh CONFIG/my_config.yaml
```

Outputs go to `output_dir` from the configuration (default `RESULTS/<project_name>/`).

| Script | Purpose | Output |
|---|---|---|
| `00_validate_inputs.py` | Check input files, columns, dates, embeddings | — |
| `01_dynamic_topic_reconstruction.py` | Cumulative daily topic snapshots per embedding model | `models/<short_name>/results.csv` |
| `02_trajectory_annotation_matrix.py` | Model-specific article trajectory labels | `trajectory_matrix.xlsx` |
| `03_calculate_agreement.py` | Consensus labels from inter-model agreement | `agreement_labels.xlsx` |
| `04_create_features_long_tables.py` | Publication-time features | `feature_article_level.csv` |
| `05_export_feature_matrix.py` | Article × threshold feature matrix | `article_url_target_features_all_k.xlsx` |
| `06_run_ml_experiments.py` | Classifiers, ablations, paired ablation tests | `results.xlsx` |
| `07_shap_oof_interpretation.py` | XGBoost SHAP interpretation | `interpretability_k<tag>_xgboost.xlsx` |
| `08_forward_chaining.py` | Chronological forward-chaining evaluation | `forward_chaining_results.xlsx` |

### Supervised experiments

```bash
python SCRIPTS/06_run_ml_experiments.py \
  --feature-matrix RESULTS/<project_name>/article_url_target_features_all_k.xlsx \
  --thresholds 1 2 3 4 5 6 7 8 \
  --selected-ablation-k 4 \
  --output RESULTS/<project_name>/results.xlsx
```

`--selected-ablation-k` adds the fold-level ablation and paired-test sheets for that threshold.

### SHAP interpretation

```bash
python SCRIPTS/07_shap_oof_interpretation.py \
  --feature-matrix RESULTS/<project_name>/article_url_target_features_all_k.xlsx \
  --agreement-k 4 \
  --dataset-name <project_name> \
  --output RESULTS/<project_name>/interpretability_k440_xgboost.xlsx
```

The tag `k440` encodes `outlier_k=4`, `toa_k=4`, `toa_max_neg=0`.

### Chronological evaluation

```bash
python SCRIPTS/08_forward_chaining.py
```

Forward-chaining evaluation of Section 6.3 and Appendix E. Input paths are set in `CORPORA` at the top of the script. Each input workbook requires `media_url`, `TA`, `y`, `true_subclass`, and the feature columns of the feature matrix.

## Outputs

### Feature matrix

`article_url_target_features_all_k.xlsx`, sheet `feature_matrix`, one row per article × threshold.

| Column | Content |
|---|---|
| `article_url` | Article identifier |
| `agreement_k` | Consensus threshold, `agreement_k = outlier_k = toa_k` |
| `label_TOA` | 1: consensus anticipatory outlier; 0: confident non-anticipatory outlier |
| `n_models_present` | Number of embedding models in which the article appears |

Remaining columns are the geometric, textual, named-entity, and social features defined in `APPENDIX/ml_feature_glossary.xlsx`.

### Results

`results.xlsx`:

| Sheet | Content |
|---|---|
| `ml_metrics_with_ablation` | Cross-validated precision, recall, F1 per threshold, classifier, feature set |
| `fold_ablation` | Fold-level XGBoost ablations at the selected threshold |
| `fold_ablation_paired_tests` | Paired t-tests with Benjamini–Hochberg correction |

Classifiers: `xgb`, `rf`, `logreg`, `linear_svc`, `dt`, `baseline_all_pos`.
Feature sets: `all_features`, `no_geom`, `no_social`, `no_text`, `only_geom`, `only_social`, `only_text`.
Column descriptions and the significance convention are in `DOCS/TABLE_SCHEMAS.md`.

`interpretability_k<tag>_xgboost.xlsx`: `xgb_global_shap` (global importance), `xgb_oof_local_shap` (out-of-fold local explanations), `xgb_local_shap` (local explanations, all articles).

## Settings used in the paper

Cumulative daily snapshots; UMAP, 20 dimensions; HDBSCAN; centroid alignment threshold 0.30; ensemble of 11 embedding models. Supervised models: `GroupKFold`, five folds, `random_state=42`, evaluated at publication time.

```text
HYDRONEWSFR:   outlier_k=4, toa_k=4, toa_max_neg=0
CLIMATENEWSFR: outlier_k=6, toa_k=6, toa_max_neg=0
```

## Data collection (ClimateNewsFr)

Articles were retrieved from Google News with the [GNews](https://pypi.org/project/gnews/) library, French query `changement climatique`:

```bash
python DATA/collection_scripts/climatenewsfr_data_collection_googlenews.py \
  --start-date 2025-04-02 \
  --end-date 2025-05-25 \
  --output DATA/private/climatenewsfr/articles.xlsx
```

X shares were collected through the [X API](https://docs.x.com/x-api/introduction). Provide the token through an environment variable:

```bash
export X_BEARER_TOKEN="..."
python DATA/collection_scripts/climatenewsfr_data_collection_X.py \
  --start-time 2025-04-02T00:00:00Z \
  --end-time 2025-05-25T23:59:59Z \
  --output-dir DATA/private/climatenewsfr/x
```
