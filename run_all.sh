#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-config/config.example.yaml}
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
print(cfg.get('output_dir', f'outputs/{cfg.get("project_name", "sample_news_corpus")}'))
PY
)
SELECTED_K=$(python - <<PY
import yaml
with open('$CONFIG', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('labeling', {}).get('selected_k', 4))
PY
)
THRESHOLDS=$(python - <<PY
import yaml
with open('$CONFIG', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print(' '.join(map(str, cfg.get('ml', {}).get('thresholds', [1,2,3,4,5,6,7,8]))))
PY
)

python scripts/00_validate_inputs.py --config "$CONFIG"
python scripts/01_dynamic_topic_reconstruction.py --config "$CONFIG"
python scripts/02_trajectory_annotation_matrix.py --config "$CONFIG" --output "$OUT/trajectory_matrix.xlsx"
python scripts/03_calculate_agreement.py --matrix "$OUT/trajectory_matrix.xlsx" --max-k 8 --toa-max-neg 0 --output "$OUT/agreement_labels.xlsx"
python scripts/04_create_features_long_tables.py --config "$CONFIG"
python scripts/05_export_feature_matrix.py --features "$OUT/feature_article_level.csv" --labels "$OUT/agreement_labels.xlsx" --id-col media_url --max-k 8 --output "$OUT/article-url_target_features_all_k.xlsx"
python scripts/06_run_ml_experiments.py --feature-matrix "$OUT/article-url_target_features_all_k.xlsx" --thresholds $THRESHOLDS --output "$OUT/results.xlsx"
python scripts/07_shap_oof_interpretation.py --feature-matrix "$OUT/article-url_target_features_all_k.xlsx" --agreement-k "$SELECTED_K" --dataset-name "$PROJECT" --output "$OUT/interpretability_k${SELECTED_K}40_xgboost.xlsx"
python scripts/08_make_plots.py --results "$OUT/results.xlsx" --shap "$OUT/interpretability_k${SELECTED_K}40_xgboost.xlsx" --dataset-name "$PROJECT" --selected-k "$SELECTED_K" --output-dir "$OUT/plots"
