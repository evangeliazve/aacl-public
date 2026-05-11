#!/usr/bin/env python3
"""Export the shareable feature matrices used in the supervised experiments.

The exported workbook matches the paper/review schema:
article_url, agreement_k, label_TOA, n_models_present, and the final feature columns.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

FINAL_FEATURES = [
    'n_models_present',
    'd1_nearest_centroid_pct_mean','d1_nearest_centroid_pct_median','d1_nearest_centroid_pct_std',
    'd2_second_centroid_pct_mean','d2_second_centroid_pct_median','d2_second_centroid_pct_std',
    'margin_d2_minus_d1_pct_mean','margin_d2_minus_d1_pct_median','margin_d2_minus_d1_pct_std',
    'mahal_nearest_pct_mean','mahal_nearest_pct_median','mahal_nearest_pct_std',
    'knn_mean_k20_pct_mean','knn_mean_k20_pct_median','knn_mean_k20_pct_std',
    'knn_std_k20_pct_mean','knn_std_k20_pct_median','knn_std_k20_pct_std',
    'outlier_proto_mean_dist_pct_mean','outlier_proto_mean_dist_pct_median','outlier_proto_mean_dist_pct_std',
    'outlier_score_mean','outlier_score_median','outlier_score_std',
    'has_recent_outliers_mean','has_recent_outliers_median','has_recent_outliers_std',
    'n_recent_outliers_mean','n_recent_outliers_median','n_recent_outliers_std',
    'soc_unique_users','soc_median_user_public_metrics_followers_count',
    'soc_median_user_public_metrics_tweet_count','soc_median_user_public_metrics_listed_count',
    'media_weighted_clustering','media_bridge_ratio','media_community_size',
    'text_subjectivity','text_neutrality','avg_sentence_len_words','avg_word_len_chars',
    'total_syllables','avg_syllables_per_word','len_chars','len_words',
    'ner_total_ents','ner_distinct_ents','ner_person','ner_org','ner_loc','ner_misc',
]


def read_table(path: str | Path, sheet: str | None = None) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {'.xlsx', '.xls'}:
        return pd.read_excel(p, sheet_name=sheet or 0)
    return pd.read_csv(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True, help='feature_article_level.csv from script 04')
    ap.add_argument('--labels', required=True, help='agreement workbook from script 03')
    ap.add_argument('--label-sheet', default='article_labels')
    ap.add_argument('--id-col', default='media_url')
    ap.add_argument('--max-k', type=int, default=8)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    feat = read_table(args.features)
    labels = read_table(args.labels, args.label_sheet)
    id_col = args.id_col
    if id_col not in feat.columns and 'article_url' in feat.columns:
        id_col = 'article_url'
    if args.id_col not in labels.columns and 'article_url' in labels.columns:
        label_id_col = 'article_url'
    else:
        label_id_col = args.id_col

    missing = [c for c in FINAL_FEATURES if c not in feat.columns]
    for c in missing:
        feat[c] = 0.0

    exported = []
    for k in range(1, args.max_k + 1):
        label_col = f'label_k{k}'
        if label_col not in labels.columns:
            continue
        lab = labels[[label_id_col, label_col]].dropna().copy()
        lab = lab.rename(columns={label_id_col: 'article_url', label_col: 'label_TOA'})
        f = feat.rename(columns={id_col: 'article_url'})[['article_url'] + FINAL_FEATURES].copy()
        one = lab.merge(f, on='article_url', how='inner')
        one.insert(1, 'agreement_k', k)
        exported.append(one)

    if not exported:
        raise ValueError('No label_k* columns were found in the labels file.')
    out = pd.concat(exported, ignore_index=True)
    ordered = ['article_url', 'agreement_k', 'label_TOA'] + FINAL_FEATURES
    out = out[ordered].fillna(0.0)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path) as xw:
        out.to_excel(xw, sheet_name='feature_matrix', index=False)
    out.to_csv(out_path.with_suffix('.csv'), index=False)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
