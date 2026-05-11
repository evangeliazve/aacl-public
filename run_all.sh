#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-CONFIG/config.example.yaml}

PROJECT=$(python - <<PY
import yaml
with open('$CONFIG', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('project_name', 'sample_news_corpus'))
PY
)

OUT=$(python - <<PY
import yaml
with open('$CONFIG', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('output_dir', f'RESULTS/{cfg.get("project_name", "sample_news_corpus")}'))
PY
)

ID_COL=$(python - <<PY
import yaml
with open('$CONFIG', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('input', {}).get('article_id_col', 'media_url'))
PY
)

MAX_K=$(python - <<PY
import yaml
with open('$CONFIG', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('labeling', {}).get('max_k', 11))
PY
)

SELECTED_K=$(python - <<PY
import yaml
with open('$CONFIG', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('labeling', {}).get('selected_k', 4))
PY
)

TOA_MAX_NEG=$(python - <<PY
import yaml
with open('$CONFIG', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('labeling', {}).get('toa_max_neg', 0))
PY
)

THRESHOLDS=$(python - <<PY
import yaml
with open('$CONFIG', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print(' '.join(map(str, cfg.get('ml', {}).get('thresholds', [1,2,3,4,5,6,7,8]))))
PY
)

K_TAG="${SELECTED_K}${SELECTED_K}${TOA_MAX_NEG}"

mkdir -p "$OUT"

python SCRIPTS/00_validate_inputs.py --config "$CONFIG"

python SCRIPTS/01_dynamic_topic_reconstruction.py \
  --config "$CONFIG"

python SCRIPTS/02_trajectory_annotation_matrix.py \
  --config "$CONFIG" \
  --output "$OUT/trajectory_matrix.xlsx"

python SCRIPTS/03_calculate_agreement.py \
  --matrix "$OUT/trajectory_matrix.xlsx" \
  --max-k "$MAX_K" \
  --toa-max-neg "$TOA_MAX_NEG" \
  --output "$OUT/agreement_labels.xlsx"

python SCRIPTS/04_create_features_long_tables.py \
  --config "$CONFIG"

python SCRIPTS/05_export_feature_matrix.py \
  --features "$OUT/feature_article_level.csv" \
  --labels "$OUT/agreement_labels.xlsx" \
  --id-col "$ID_COL" \
  --max-k "$MAX_K" \
  --output "$OUT/article_url_target_features_all_k.xlsx"

python SCRIPTS/06_run_ml_experiments.py \
  --feature-matrix "$OUT/article_url_target_features_all_k.xlsx" \
  --thresholds $THRESHOLDS \
  --output "$OUT/results.xlsx"

python SCRIPTS/07_shap_oof_interpretation.py \
  --feature-matrix "$OUT/article_url_target_features_all_k.xlsx" \
  --agreement-k "$SELECTED_K" \
  --dataset-name "$PROJECT" \
  --output "$OUT/interpretability_k${K_TAG}_xgboost.xlsx"
