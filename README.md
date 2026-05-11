# Can Topic-Model Outliers Predict Future Topics?

This repository accompanies the EMNLP submission **Can Topic-Model Outliers Predict Future Topics? A Prospective Study of Weak Signals in Embedding Space**.

It contains the scripts, feature tables, agreement tables, machine-learning results, and interpretation workbooks used to reproduce the analysis on a new news corpus and to inspect the reported experiments.

## How the repository is organized

The repository is organized around three kinds of material.

First, `DATA/` contains the article-level feature matrices used in the supervised experiments. These files include the article URLs, agreement threshold `k`, binary target, and final features. They make it possible to inspect the URLs used in the experiments and to rerun the machine-learning and SHAP stages without redistributing the original article text or raw social-media traces.

Second, `RESULTS/` contains the reported experimental outputs. The result workbooks include machine-learning performance across several agreement thresholds `k`, all classifiers, feature-family ablations, and simple baselines. They report F1-score, recall, precision, average precision, ROC AUC, and the corresponding cross-validation standard deviations. The folder also contains the agreement matrices and the SHAP interpretation workbooks used for the analysis.

Third, `SCRIPTS/`, `CONFIG/`, and `DOCS/` document and reproduce the full pipeline on a new corpus with the same table structure.

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

## Included workbooks

### Article-level feature matrices in `DATA/`

The two files below contain the supervised article-level tables used by the modeling scripts:

```text
DATA/hydronewsfr_article-url_target_features_all_k.xlsx
DATA/climatenewsfr_article-url_target_features_all_k.xlsx
```

Each workbook has a `feature_matrix` sheet. The main columns are:

- `article_url`: URL used as the article identifier in the released table;
- `agreement_k`: agreement threshold used to define the target;
- `label_TOA`: binary target, where 1 marks anticipatory outliers and 0 marks non-anticipatory outliers;
- `n_models_present`: number of embedding-model representations available for the article;
- geometric, text, and social features used in the supervised experiments.


### Machine-learning results in `RESULTS/`

The main performance workbooks are:

```text
RESULTS/hydronewsfr/hydronewsfr_results.xlsx
RESULTS/climatenewsfr/climatenewsfr_results.xlsx
```

Each contains two sheets.

`ml_metrics_with_ablation` reports the cross-validated results across agreement settings and model variants. It includes:

- horizon and agreement rule: `horizon`, `outlier_maj`, `toa_min_pos`, `toa_max_neg`;
- model and ablation setting: `clf_name`, `ablation`;
- retained sample size: `n_articles`, `n_pos_articles_est`, `n_neg_articles_est`;
- main metrics: `F1_mean`, `Precision_mean`, `Recall_mean`;
- additional ranking metrics: `AP_mean`, `ROC_AUC_mean`;
- standard deviations for each metric.

The classifiers include Logistic Regression, Linear SVC, Decision Tree, Random Forest, XGBoost, and simple baselines. The ablation rows compare the full feature set with geometry-only, text-only, social-only, and feature-removal settings.

`uncertain_scoring` summarizes predictions for articles that are not assigned a confident consensus label under a given agreement rule. It reports the number of uncertain articles and the distribution of predicted positive-class probabilities: mean, median, 10th percentile, and 90th percentile.

### Agreement matrices in `RESULTS/agreement/`

```text
RESULTS/agreement/agreement_hdbscan_th30_d20.xlsx
RESULTS/agreement/agreement_hdbscan_th30_d20_climat.xlsx
```

These files contain the model-by-model trajectory labels and agreement summaries. The sheet `TOA matrix and agreement` includes the article URL, publication date, first topic information, one trajectory column per embedding model, and agreement counts such as `toa_agreement` and `num_toa`.

### SHAP interpretation workbooks in `RESULTS/`

```text
RESULTS/hydronewsfr/hydronewsfr_interpretability_k440_xgboost.xlsx
RESULTS/climatenewsfr/climatenewsfr_interpretability_k660_xgboost.xlsx
```

These workbooks explain out-of-fold XGBoost predictions for the selected agreement thresholds: `k=4` for HYDRONEWSFR and `k=6` for CLIMATENEWSFR.

They contain:

- `xgb_global_shap`: global feature importance, mean SHAP values, feature-value summaries, and Spearman correlations between feature values and SHAP contributions;
- `xgb_local_predictions`: out-of-fold predicted probabilities for individual articles;
- `xgb_local_shap_long`: article-feature-level SHAP contributions;
- `xgb_local_topk_per_article`: strongest local explanations per article;
- `xgb_shap_foldwise`: fold-level SHAP importance;
- `xgb_shap_stability`: stability of important features across folds.

### Appendix files

`APPENDIX/ml_feature_glossary.xlsx` provides the feature glossary. Additional appendix workbooks are placed under `APPENDIX/additional_experiments/`.

## Raw data and redistribution

The original article text, collection files, and raw social-media traces are not redistributed in this repository. They may be subject to publisher, API, or platform restrictions.

The released article-level workbooks provide URLs, labels, and features. This allows the modeling results to be inspected and rerun while avoiding redistribution of restricted raw content. The full pipeline can still be applied to any local corpus that follows the schemas in `DOCS/TABLE_SCHEMAS.md`.

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

## Running the full workflow on a new corpus

Copy the example configuration and edit the local input paths:

```bash
cp CONFIG/config.example.yaml CONFIG/my_corpus.yaml
```

At minimum, the article table should contain:

- a stable article identifier or canonical URL;
- a publication date;
- a title;
- a lead paragraph, description, or body-text field.

Social-sharing data are optional. If no sharing file is provided, social features are filled with zeros.

Run the full sequence:

```bash
bash run_all.sh CONFIG/my_corpus.yaml
```

The example configuration assumes local private inputs under `DATA/private/`. Those files are not included in the repository.

## Pipeline scripts

### 0. Validate inputs

```bash
python SCRIPTS/00_validate_inputs.py --config CONFIG/my_corpus.yaml
```

Checks required columns, date parsing, optional social-sharing data, and configured embedding files.

### 1. Dynamic topic reconstruction

```bash
python SCRIPTS/01_dynamic_topic_reconstruction.py --config CONFIG/my_corpus.yaml
```

Builds cumulative daily topic snapshots for each embedding model. The output for each model is:

```text
RESULTS/<corpus>/models/<model>/results.csv
```

This file contains article assignments, aligned topic IDs, UMAP coordinates, outlier indicators, and HDBSCAN outlier scores.

### 2. Trajectory annotation matrix

```bash
python SCRIPTS/02_trajectory_annotation_matrix.py \
  --config CONFIG/my_corpus.yaml \
  --output RESULTS/my_corpus/trajectory_matrix.xlsx
```

Builds the model-by-model article trajectory matrix.

### 3. Agreement and consensus labels

```bash
python SCRIPTS/03_calculate_agreement.py \
  --matrix RESULTS/my_corpus/trajectory_matrix.xlsx \
  --max-k 11 \
  --toa-max-neg 0 \
  --output RESULTS/my_corpus/agreement_labels.xlsx
```

For a threshold `k`, positive cases require at least `k` anticipatory votes. In the paper setting, negative cases require publication-time outlier support and zero anticipatory votes.

### 4. Publication-time features

```bash
python SCRIPTS/04_create_features_long_tables.py --config CONFIG/my_corpus.yaml
```

Creates article-model-level geometric features and article-level text/social features. The main outputs are:

```text
RESULTS/<corpus>/feature_long_model_level.csv
RESULTS/<corpus>/feature_article_level.csv
```

### 5. Export supervised feature matrix

```bash
python SCRIPTS/05_export_feature_matrix.py \
  --features RESULTS/my_corpus/feature_article_level.csv \
  --labels RESULTS/my_corpus/agreement_labels.xlsx \
  --id-col media_url \
  --max-k 11 \
  --output RESULTS/my_corpus/article_url_target_features_all_k.xlsx
```

Exports the article URL, target, agreement threshold, and feature table used for supervised learning.

### 6. Machine-learning experiments

```bash
python SCRIPTS/06_run_ml_experiments.py \
  --feature-matrix RESULTS/my_corpus/article_url_target_features_all_k.xlsx \
  --thresholds 1 2 3 4 5 6 7 8 \
  --output RESULTS/my_corpus/results.xlsx
```

Runs the classifiers, baselines, and ablations. The output workbook contains cross-validated F1, precision, recall, average precision, ROC AUC, and standard deviations.

### 7. SHAP interpretation

```bash
python SCRIPTS/07_shap_oof_interpretation.py \
  --feature-matrix RESULTS/my_corpus/article_url_target_features_all_k.xlsx \
  --agreement-k 4 \
  --dataset-name MyCorpus \
  --output RESULTS/my_corpus/interpretability_k440_xgboost.xlsx
```

Computes global and local SHAP explanations from out-of-fold XGBoost predictions.

## Rerunning the supervised stage from the released feature matrices

The feature matrices in `DATA/` can be used directly with scripts 6 and 7.

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

## Main experimental settings

The paper uses cumulative daily snapshots, UMAP with 20 dimensions, HDBSCAN clustering, centroid-based topic alignment with threshold `0.30`, and an ensemble of embedding models. The supervised task is evaluated at publication time.

The main agreement thresholds are:

- HYDRONEWSFR: `k=4`
- CLIMATENEWSFR: `k=6`

Exact numerical reproduction depends on the same raw corpus, cached embeddings, package versions (cf. requirements.txt), preprocessing choices, and random seed. The default random seed is `42`. 
