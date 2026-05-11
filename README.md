# Predicting anticipatory topic outliers

This repository contains standalone Python scripts for reproducing the article-level experiments on a new news corpus. The workflow reconstructs dynamic topics from cumulative daily snapshots, derives trajectory-based labels, creates publication-time features, runs supervised models, exports SHAP interpretation tables, creates plots, and runs the appendix chronological robustness check.

The code is intentionally script-based. It is not a Python package and does not require installation with `pip install -e .`.

## Repository layout

```text
config/
  config.example.yaml              # edit this for a new corpus
data/
  README.md                        # where to place local/private data
  toy/                             # tiny schema example, not for metrics
scripts/
  00_validate_inputs.py
  01_dynamic_topic_reconstruction.py
  02_trajectory_annotation_matrix.py
  03_calculate_agreement.py
  04_create_features_long_tables.py
  05_export_feature_matrix.py
  06_run_ml_experiments.py
  07_shap_oof_interpretation.py
  08_make_plots.py
  09_chronological_robustness_check.py
docs/
  TABLE_SCHEMAS.md
requirements.txt
environment.yml
run_all.sh
LICENSE
CITATION.cff
outputs/
```

## Input data expected for a new corpus

At minimum, provide one article table with:

- one stable article identifier or URL;
- publication date;
- title;
- lead paragraph, description, or body text.

Social-sharing data are optional. If missing, all social features are set to zero. See `docs/TABLE_SCHEMAS.md` for the full schema.

Raw news articles and social traces may be subject to licensing, platform, or publisher restrictions. This repository therefore documents schemas and code paths, but users should provide their own corpus unless redistribution is explicitly allowed.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download fr_core_news_md   # optional but recommended for French NER
```

For local embedding models, the scripts use `sentence-transformers`. For API-based embedding models, keep API calls outside this repository or provide precomputed embeddings as `data/embeddings/<short_name>.npy` or `.csv`. The scripts do not store keys and expect credentials to be managed through environment variables or external preprocessing.

Alternatively, with Conda/Mamba:

```bash
mamba env create -f environment.yml
conda activate topic-outlier-reproduction
python -m spacy download fr_core_news_md
```

## Configure a run

Copy and edit the example config:

```bash
cp config/config.example.yaml config/my_corpus.yaml
```

Important fields:

- `input.articles`: CSV/XLSX article file.
- `input.shares`: optional CSV/XLSX social-sharing file.
- `input.article_id_col`: article ID or canonical URL column.
- `input.date_col`: publication date column.
- `embedding.models`: embedding models to run or match to precomputed embedding files.
- `output_dir`: destination for all intermediate and final outputs.

## One-command run

After editing the config, the full main pipeline can be run with:

```bash
bash run_all.sh config/my_corpus.yaml
```

The chronological appendix check is intentionally not included in `run_all.sh`, because it is a robustness analysis rather than the main result pipeline.

## End-to-end workflow

### 0. Validate inputs

```bash
python scripts/00_validate_inputs.py --config config/my_corpus.yaml
```

This checks that required article columns, date parsing, optional social-sharing columns, and configured precomputed embeddings are available.

### 1. Dynamic topic reconstruction

```bash
python scripts/01_dynamic_topic_reconstruction.py --config config/my_corpus.yaml
```

This creates one folder per embedding model under `outputs/<corpus>/models/<model>/` and writes `results.csv` for each model. These files contain cumulative snapshot assignments, topic IDs after alignment, UMAP coordinates, and HDBSCAN outlier scores.

### 2. Trajectory annotation matrix

```bash
python scripts/02_trajectory_annotation_matrix.py \
  --config config/my_corpus.yaml \
  --output outputs/my_corpus/trajectory_matrix.xlsx
```

This converts each model-specific `results.csv` into article-level trajectory votes and a wide agreement matrix.

### 3. Agreement and consensus labels

```bash
python scripts/03_calculate_agreement.py \
  --matrix outputs/my_corpus/trajectory_matrix.xlsx \
  --max-k 8 \
  --toa-max-neg 0 \
  --output outputs/my_corpus/agreement_labels.xlsx
```

For each threshold `k`, positives require at least `k` anticipatory votes, negatives require outlier support and zero anticipatory votes, and uncertain cases are excluded from supervised training.

### 4. Publication-time features and long tables

```bash
python scripts/04_create_features_long_tables.py --config config/my_corpus.yaml
```

Outputs:

- `feature_long_model_level.csv`: model-level publication-time geometry plus article-level text/social features.
- `feature_article_level.csv`: article-level feature table after aggregating geometric values across models with mean, median, and standard deviation.

### 5. Export review feature matrix

```bash
python scripts/05_export_feature_matrix.py \
  --features outputs/my_corpus/feature_article_level.csv \
  --labels outputs/my_corpus/agreement_labels.xlsx \
  --id-col media_url \
  --max-k 8 \
  --output outputs/my_corpus/article-url_target_features_all_k.xlsx
```

The exported workbook contains the same type of table used for review: `article_url`, `agreement_k`, `label_TOA`, and the final model features.

### 6. Supervised ML experiments

```bash
python scripts/06_run_ml_experiments.py \
  --feature-matrix outputs/my_corpus/article-url_target_features_all_k.xlsx \
  --thresholds 1 2 3 4 5 6 7 8 \
  --output outputs/my_corpus/results.xlsx
```

The script evaluates Logistic Regression, Linear SVC, Decision Tree, Random Forest, XGBoost, and constant baselines. It also runs feature-family ablations.

### 7. SHAP interpretation with out-of-fold local explanations

```bash
python scripts/07_shap_oof_interpretation.py \
  --feature-matrix outputs/my_corpus/article-url_target_features_all_k.xlsx \
  --agreement-k 4 \
  --dataset-name MyCorpus \
  --output outputs/my_corpus/interpretability_k440_xgboost.xlsx
```

Use the selected `k` for the corpus. The output workbook contains global SHAP rankings, out-of-fold local predictions, long SHAP values, top-k local explanations per article, and foldwise SHAP summaries.

### 8. Plots

```bash
python scripts/08_make_plots.py \
  --results outputs/my_corpus/results.xlsx \
  --shap outputs/my_corpus/interpretability_k440_xgboost.xlsx \
  --dataset-name MyCorpus \
  --selected-k 4 \
  --output-dir outputs/my_corpus/plots
```

### 9. Appendix chronological robustness check

```bash
python scripts/09_chronological_robustness_check.py \
  --feature-matrix outputs/my_corpus/article-url_target_features_all_k.xlsx \
  --articles data/articles.csv \
  --id-col media_url \
  --date-col publication_date_cleaned \
  --thresholds 1 2 3 4 5 6 7 8 \
  --feature-set geometry_text \
  --model xgboost \
  --output outputs/my_corpus/chronological_robustness.xlsx
```

This evaluates the trained signal under chronological train-before-test folds and exports:

- `chronological_summary`;
- `chronological_folds`;
- `chronological_oof_predictions`;
- `features_used`.

Use `--feature-set geometry_text` for an extended corpus when social features are unavailable over the full period.

## Reproducing the paper settings

The paper setting uses cumulative daily snapshots, UMAP with 20 dimensions, HDBSCAN clustering, centroid alignment threshold `0.30`, and an embedding-model ensemble. Main supervised settings are `k=4` for the hydrogen corpus and `k=6` for the climate corpus. The broad threshold `k=1` can be used to inspect the coverage-confidence tradeoff.

The appendix robustness check for the extended hydrogen corpus uses the same publication-time prediction setup on a longer chronological window, with the reduced geometry+text feature set when social features are not uniformly available.

## Expected outputs

```text
outputs/<corpus>/models/<model>/results.csv
outputs/<corpus>/trajectory_matrix.xlsx
outputs/<corpus>/agreement_labels.xlsx
outputs/<corpus>/feature_long_model_level.csv
outputs/<corpus>/feature_article_level.csv
outputs/<corpus>/article-url_target_features_all_k.xlsx
outputs/<corpus>/results.xlsx
outputs/<corpus>/interpretability_k*_xgboost.xlsx
outputs/<corpus>/chronological_robustness.xlsx
outputs/<corpus>/plots/*.pdf
outputs/<corpus>/plots/*.png
```

## Notes on reproducibility

- `random_state` defaults to `42`.
- UMAP and HDBSCAN can vary with implementation versions and threading; exact reproduction is strongest with cached embeddings and pinned package versions.
- Preprocessing, imputation, scaling, and model fitting are done inside each cross-validation fold.
- Article IDs are used as groups in cross-validation.
- Linear models use median imputation and standardization.
- Tree models use median imputation without standardization.
- XGBoost uses `scale_pos_weight = N_negative / max(N_positive, 1)`.
- Social features are zero-filled when no publication-time X-sharing trace is observed.
- API embeddings should be cached before running the pipeline so that the experiments can be rerun without changing representations.
