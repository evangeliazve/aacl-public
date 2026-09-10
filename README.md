# Predicting Emerging Topics from Outliers: A Prospective Study of Weak Signals in Embedding Space

Code for our paper, accepted at AACL-IJCNLP 2026 (Findings).

This repository contains the pipeline scripts, configuration, dependencies, data-collection scripts, and documentation needed to run the method on a news corpus.

**Data are not included.** Raw article text, social-media traces, embeddings, feature matrices, and result workbooks are not redistributed in this repository, because they may be subject to publisher, API, or platform restrictions. To run the pipeline, provide your own local files in the formats described below.


## Repository structure

```text
APPENDIX/
  ml_feature_glossary.xlsx        Description of every feature used by the models

CONFIG/
  config.example.yaml             Example configuration (paths, clustering, labeling, ML)

DATA/
  README.md
  collection_scripts/
    climatenewsfr_data_collection_googlenews.py
    climatenewsfr_data_collection_X.py

DOCS/
  TABLE_SCHEMAS.md                Column descriptions of the ML result workbooks

SCRIPTS/
  00_validate_inputs.py
  01_dynamic_topic_reconstruction.py
  02_trajectory_annotation_matrix.py
  03_calculate_agreement.py
  04_create_features_long_tables.py
  05_export_feature_matrix.py
  06_run_ml_experiments.py
  07_shap_oof_interpretation.py

environment.yml
requirements.txt
run_all.sh
README.md
```

The folders `DATA/private/` (your input data) and `RESULTS/` (pipeline outputs) are created locally and should not be committed.


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

The French spaCy model is used for named-entity features. The reported experiments use Python 3.12 and the package versions pinned in `requirements.txt` / `environment.yml`.


## Input data

Paths and column names are set in `CONFIG/config.example.yaml`. Copy it and edit the copy for your corpus.

### Article table (required)

`input.articles`, default `DATA/private/articles.csv` (`.csv` or `.xlsx`). One row per article.

| Config key | Default column | Description |
|---|---|---|
| `article_id_col` | `media_url` | Unique article identifier. |
| `url_col` | `media_url` | Article URL, used to join with the social-sharing table. |
| `date_col` | `publication_date_cleaned` | Publication date (parseable by pandas). |
| `title_col` | `title` | Article title. |
| `text_col` | `description` | Lead paragraph or short text. |

The text embedded for each article is built from `embedding.text_template` (default `"{title} {description}"`).

### Social-sharing table (optional)

`input.shares`, default `DATA/private/social_shares.csv`. One row per share. If the file is missing, social features are set to zero.

| Column | Description |
|---|---|
| `media_url`, `article_url`, or `url` | Shared article, matching the article table. |
| `created_at` | Share timestamp. Only shares before the article's evaluation time are used. |
| `author_id` or `user_id` | Sharing account. |
| `followers_count`, `tweet_count`, `listed_count` | Optional account metrics (the `user_public_metrics_` prefix is also accepted). |

### Embeddings

Models of type `sentence_transformer` are computed locally. For API-based models, set `type: precomputed` and place one file per model in `embedding.precomputed_dir` (default `DATA/private/embeddings/<short_name>.npy` or `.csv`), with rows aligned to the article table.


## Running the pipeline

Full pipeline with one command:

```bash
bash run_all.sh CONFIG/my_config.yaml
```

Outputs are written to `output_dir` from the configuration (default `RESULTS/<project_name>/`).

Or step by step:

| Script | Purpose | Main output |
|---|---|---|
| `00_validate_inputs.py` | Check input files, columns, dates, and embeddings. | — |
| `01_dynamic_topic_reconstruction.py` | Build cumulative daily topic snapshots per embedding model. | `models/<short_name>/results.csv` |
| `02_trajectory_annotation_matrix.py` | Build model-specific article trajectory labels. | `trajectory_matrix.xlsx` |
| `03_calculate_agreement.py` | Compute agreement-based consensus labels. | `agreement_labels.xlsx` |
| `04_create_features_long_tables.py` | Compute publication-time features. | `feature_article_level.csv` |
| `05_export_feature_matrix.py` | Export the article-threshold feature matrix. | `article_url_target_features_all_k.xlsx` |
| `06_run_ml_experiments.py` | Run classifiers, ablations, and paired ablation tests. | `results.xlsx` |
| `07_shap_oof_interpretation.py` | Compute XGBoost SHAP interpretation. | `interpretability_k<tag>_xgboost.xlsx` |

### Supervised experiments

```bash
python SCRIPTS/06_run_ml_experiments.py \
  --feature-matrix RESULTS/<project_name>/article_url_target_features_all_k.xlsx \
  --thresholds 1 2 3 4 5 6 7 8 \
  --selected-ablation-k 4 \
  --output RESULTS/<project_name>/results.xlsx
```

`--selected-ablation-k` adds the fold-level ablation and paired-test sheets for the chosen threshold. Without it, only `ml_metrics_with_ablation` is produced.

### SHAP interpretation

```bash
python SCRIPTS/07_shap_oof_interpretation.py \
  --feature-matrix RESULTS/<project_name>/article_url_target_features_all_k.xlsx \
  --agreement-k 4 \
  --dataset-name <project_name> \
  --output RESULTS/<project_name>/interpretability_k440_xgboost.xlsx
```

The filename tag encodes the consensus rule: `k440` means `outlier_k=4`, `toa_k=4`, `toa_max_neg=0`.


## Outputs

### Feature matrix

`article_url_target_features_all_k.xlsx`, sheet `feature_matrix`. One row per article-threshold pair.

| Column | Description |
|---|---|
| `article_url` | Article identifier. |
| `agreement_k` | Diagonal consensus threshold, with `agreement_k = outlier_k = toa_k`. |
| `label_TOA` | `1` for a consensus anticipatory outlier; `0` for a confident non-anticipatory publication-time outlier. |
| `n_models_present` | Number of embedding models in which the article is represented. |

The remaining columns are geometric, textual, named-entity, and social/co-sharing features, described in `APPENDIX/ml_feature_glossary.xlsx`.

### ML results

`results.xlsx` contains three sheets:

| Sheet | Content |
|---|---|
| `ml_metrics_with_ablation` | Cross-validated precision, recall, and F1 per threshold, classifier, and feature set. |
| `fold_ablation` | Fold-level XGBoost ablation results for the selected threshold. |
| `fold_ablation_paired_tests` | Paired t-tests between ablations, with Benjamini-Hochberg correction. |

Classifiers: `xgb` (XGBoost), `rf` (Random Forest), `logreg` (Logistic Regression), `linear_svc` (Linear SVC), `dt` (Decision Tree), and `baseline_all_pos` (constant-positive baseline).

Feature sets: `all_features`, `no_geom`, `no_social`, `no_text`, `only_geom`, `only_social`, `only_text`.

Full column descriptions and the significance-symbol convention used in the paper are in `DOCS/TABLE_SCHEMAS.md`.

### SHAP interpretation

`interpretability_k<tag>_xgboost.xlsx` contains `xgb_global_shap` (global feature importance), `xgb_oof_local_shap` (out-of-fold local explanations), and `xgb_local_shap` (local explanations for all articles).


## Main experimental settings

The paper uses cumulative daily snapshots, UMAP with 20 dimensions, HDBSCAN clustering, centroid-based topic alignment with threshold `0.30`, and an ensemble of embedding models. Supervised models use `GroupKFold` with five folds and `random_state=42`, and are evaluated at publication time.

Selected agreement thresholds:

```text
HYDRONEWSFR:   outlier_k=4, toa_k=4, toa_max_neg=0
CLIMATENEWSFR: outlier_k=6, toa_k=6, toa_max_neg=0
```


## Data collection (CLIMATENEWSFR)

The scripts in `DATA/collection_scripts/` document how the CLIMATENEWSFR corpus was collected.

News articles were collected from Google News with the [GNews Python library](https://pypi.org/project/gnews/), using the French query `changement climatique` between `2025-04-02` and `2025-05-25`:

```bash
python DATA/collection_scripts/climatenewsfr_data_collection_googlenews.py \
  --start-date 2025-04-02 \
  --end-date 2025-05-25 \
  --output DATA/private/climatenewsfr/articles.xlsx
```

X sharing activity was collected through the official [X API](https://docs.x.com/x-api/introduction). Pass your own token through an environment variable, never in the code:

```bash
export X_BEARER_TOKEN="..."
python DATA/collection_scripts/climatenewsfr_data_collection_X.py \
  --start-time 2025-04-02T00:00:00Z \
  --end-time 2025-05-25T23:59:59Z \
  --output-dir DATA/private/climatenewsfr/x
```
