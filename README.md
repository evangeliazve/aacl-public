# Can Topic-Model Outliers Predict Future Topics?

This repository contains results and reproducibility materials for the EMNLP submission: **Can Topic-Model Outliers Predict Future Topics? A Prospective Study of Weak Signals in Embedding Space**.

It includes the article-level feature matrices used in the supervised experiments, released result workbooks, agreement tables, SHAP interpretation tables, data-collection scripts, and scripts for rerunning the experiments on other data.

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
  collection_scripts/
    climatenewsfr_data_collection_googlenews.py
    climatenewsfr_data_collection_X.py
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

requirements.txt
environment.yml
run_all.sh
README.md
```

## Released data and result files

### Article-level feature matrices

```text
DATA/hydronewsfr_article-url_target_features_all_k.xlsx
DATA/climatenewsfr_article-url_target_features_all_k.xlsx
```

Each workbook has one sheet, `feature_matrix`. Rows are article-threshold pairs. The main columns are:

- `article_url`: URL used as the article identifier in the released matrix;
- `agreement_k`: released diagonal agreement threshold, with `agreement_k = outlier_k = toa_k`;
- `label_TOA`: binary supervised target, where 1 marks a consensus anticipatory outlier and 0 marks a confident non-anticipatory publication-time outlier;
- `n_models_present`: number of embedding-model representations available for the article;
- geometric, text, named-entity, and social/co-sharing features used by scripts 6 and 7.

The released feature matrices contain `agreement_k` values 1 through 8.


### Data-collection scripts

```text
DATA/collection_scripts/climatenewsfr_data_collection_googlenews.py
DATA/collection_scripts/climatenewsfr_data_collection_X.py
```

These scripts document the collection procedure used to construct the CLIMATENEWSFR corpus from Google News API results and observed X-sharing activity. They are provided to support reuse of the collection pipeline on new local corpora.

### Machine-learning result workbooks

```text
RESULTS/hydronewsfr/hydronewsfr_results.xlsx
RESULTS/climatenewsfr/climatenewsfr_results.xlsx
```

Each workbook contains one sheet:

```text
ml_metrics_with_ablation
```

The sheet reports cross-validated results for the released diagonal agreement settings, where:

```text
outlier_k = toa_k = agreement_k
toa_max_neg = 0
```

The columns include `horizon`, `outlier_k`, `toa_k`, `toa_max_neg`, `cv_n_splits`, `clf_name`, `ablation`, `n_articles`, class counts, and fold means/standard deviations for F1, precision, recall, average precision (`AP_*`), and ROC AUC (`ROC_AUC_*`).

The supervised models are abbreviated as follows: `xgb` denotes XGBoost, `rf` Random Forest, `logreg` Logistic Regression, `linear_svc` a linear Support Vector Machine, and `dt` a Decision Tree classifier.

The ablation labels describe which feature families are used. `all_features` includes the full set of geometric, textual, and social predictors. `no_geom`, `no_social`, and `no_text` remove one feature family at a time. `only_geom`, `only_social`, and `only_text` keep only the corresponding feature family.

The baseline row is `baseline_all_pos`. It is reported once per agreement threshold, not once per ablation; its ablation field is `baseline`.

### Agreement matrices

```text
RESULTS/agreement/agreement_hdbscan_th30_d20.xlsx
RESULTS/agreement/agreement_hdbscan_th30_d20_climat.xlsx
```

These workbooks contain the model-by-model trajectory labels and agreement summaries. The sheet `TOA matrix and agreement` includes the article URL, publication date, first-topic information, one trajectory column per embedding model, and agreement counts such as TOA vote counts and publication-time outlier vote counts.

### SHAP interpretation workbooks

```text
RESULTS/hydronewsfr/hydronewsfr_interpretability_k440_xgboost.xlsx
RESULTS/climatenewsfr/climatenewsfr_interpretability_k660_xgboost.xlsx
```

The filename tag encodes the selected diagonal agreement rule:

- `k440`: `outlier_k=4`, `toa_k=4`, `toa_max_neg=0`;
- `k660`: `outlier_k=6`, `toa_k=6`, `toa_max_neg=0`.

Each workbook contains three sheets:

- `xgb_global_shap`: global feature importance and direction diagnostics;
- `xgb_local_predictions`: out-of-fold predicted probabilities for individual articles;
- `xgb_local_shap_long`: article-feature-level SHAP contributions.

### Appendix files

`APPENDIX/ml_feature_glossary.xlsx` provides the released feature glossary. Additional robustness and correlation tables are in `APPENDIX/additional_experiments/`.

## Release sanity checks

The files in this release have the following expected structure:

| File | Expected structure |
|---|---|
| `DATA/hydronewsfr_article-url_target_features_all_k.xlsx` | `feature_matrix`, 3,338 rows, `agreement_k` 1–8 |
| `DATA/climatenewsfr_article-url_target_features_all_k.xlsx` | `feature_matrix`, 6,501 rows, `agreement_k` 1–8 |
| `RESULTS/hydronewsfr/hydronewsfr_results.xlsx` | `ml_metrics_with_ablation`, 288 rows |
| `RESULTS/climatenewsfr/climatenewsfr_results.xlsx` | `ml_metrics_with_ablation`, 288 rows |
| `RESULTS/hydronewsfr/hydronewsfr_interpretability_k440_xgboost.xlsx` | 3 SHAP sheets |
| `RESULTS/climatenewsfr/climatenewsfr_interpretability_k660_xgboost.xlsx` | 3 SHAP sheets |

The ML result workbooks contain 8 thresholds. For each threshold there is one `baseline_all_pos` row and 35 classifier-ablation rows: 5 classifiers × 7 ablations.

## Raw data and redistribution

The original article text, collection files, raw social-media traces, and some embedding caches are not redistributed because they may be subject to publisher, API, or platform restrictions.

The released feature matrices provide URLs, consensus labels, and engineered features. They support inspection and supervised-stage reruns without redistributing restricted raw content. The collection and analysis pipeline can be applied to a new local corpus that follows the schemas in `DOCS/TABLE_SCHEMAS.md`.

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

The French spaCy model is used for named-entity features. If it is not installed, named-entity counts are set to zero by the feature script.

## Rerun the supervised stage from the released feature matrices

HYDRONEWSFR:

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

CLIMATENEWSFR:

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

Because the estimators use stochastic components, rerun values may differ slightly from the released workbooks even with `random_state=42`. The expected structure is one `ml_metrics_with_ablation` sheet for ML results and three SHAP sheets for interpretation.

## Run the workflow on a new local corpus

Copy the example configuration and edit the local paths:

```bash
cp CONFIG/config.example.yaml CONFIG/my_corpus.yaml
```

At minimum, the article table should contain a stable article identifier or canonical URL, a publication date, a title, and a lead/body-text field. Social-sharing data are optional; if no sharing file is provided, social features are filled with zeros.

Run the full sequence:

```bash
bash run_all.sh CONFIG/my_corpus.yaml
```

The example configuration assumes private local inputs under `DATA/private/`. Those files are not included in this repository.

## Pipeline scripts

1. `00_validate_inputs.py`: checks required columns, date parsing, optional social-sharing data, and configured embedding files.
2. `01_dynamic_topic_reconstruction.py`: builds cumulative daily topic snapshots for each embedding model.
3. `02_trajectory_annotation_matrix.py`: builds the model-by-model article trajectory matrix.
4. `03_calculate_agreement.py`: creates consensus labels for thresholds `k`.
5. `04_create_features_long_tables.py`: creates publication-time geometric, text, and social features.
6. `05_export_feature_matrix.py`: exports the shareable article-threshold feature matrix.
7. `06_run_ml_experiments.py`: runs classifiers, the `baseline_all_pos` baseline, and the released feature-family ablations.
8. `07_shap_oof_interpretation.py`: computes global XGBoost SHAP summaries and out-of-fold local SHAP tables for the selected agreement threshold.

## Main experimental settings

The paper setting uses cumulative daily snapshots, UMAP with 20 dimensions, HDBSCAN clustering, centroid-based topic alignment with threshold `0.30`, and an ensemble of embedding models. The supervised task is evaluated at publication time.

The main selected agreement thresholds are:

- HYDRONEWSFR: `outlier_k=4`, `toa_k=4`, `toa_max_neg=0`;
- CLIMATENEWSFR: `outlier_k=6`, `toa_k=6`, `toa_max_neg=0`.
