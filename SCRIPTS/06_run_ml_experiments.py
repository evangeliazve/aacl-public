#!/usr/bin/env python3
"""Run cross-validated ML experiments and feature-family ablations."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

GEOM_PREFIXES = (
    'd1_', 'd2_', 'margin_', 'mahal_', 'knn_', 'outlier_proto_', 'outlier_score',
    'has_recent_outliers', 'n_recent_outliers', 'n_models_present'
)
TEXT_PREFIXES = ('text_', 'avg_', 'total_syllables', 'len_', 'ner_')
SOCIAL_PREFIXES = ('soc_', 'media_')
META_COLS = {'article_url', 'agreement_k', 'label_TOA', 'horizon'}


def read_matrix(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {'.xlsx', '.xls'}:
        return pd.read_excel(p, sheet_name='feature_matrix')
    return pd.read_csv(p)


def feature_groups(columns: List[str]) -> Dict[str, List[str]]:
    feats = [c for c in columns if c not in META_COLS]
    geom = [c for c in feats if c.startswith(GEOM_PREFIXES)]
    text = [c for c in feats if c.startswith(TEXT_PREFIXES)]
    social = [c for c in feats if c.startswith(SOCIAL_PREFIXES)]
    return {
        'all_features': feats,
        'geometry_only': geom,
        'text_only': text,
        'social_only': social,
        'without_geometry': [c for c in feats if c not in geom],
        'without_text': [c for c in feats if c not in text],
        'without_social': [c for c in feats if c not in social],
    }


def make_models(y: np.ndarray, random_state: int):
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = int((y == 0).sum())
    scale_pos_weight = n_neg / n_pos
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, objective='binary:logistic',
            eval_metric='logloss', scale_pos_weight=scale_pos_weight,
            random_state=random_state, n_jobs=-1,
        )
    except Exception:
        xgb = None
    models = {
        'logreg_l2': Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()),
                              ('clf', LogisticRegression(penalty='l2', solver='liblinear', max_iter=2000, class_weight='balanced'))]),
        'linear_svc': Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()),
                               ('clf', LinearSVC(max_iter=5000, class_weight='balanced'))]),
        'decision_tree': Pipeline([('imputer', SimpleImputer(strategy='median')),
                                  ('clf', DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, class_weight='balanced', random_state=random_state))]),
        'random_forest': Pipeline([('imputer', SimpleImputer(strategy='median')),
                                  ('clf', RandomForestClassifier(n_estimators=600, max_depth=8, min_samples_split=20,
                                                                 min_samples_leaf=10, max_features='sqrt',
                                                                 class_weight='balanced', random_state=random_state, n_jobs=-1))]),
    }
    if xgb is not None:
        models['xgboost'] = Pipeline([('imputer', SimpleImputer(strategy='median')), ('clf', xgb)])
    return models


def scores(y_true, y_pred, y_score) -> Dict[str, float]:
    out = {
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
    }
    try:
        out['AveragePrecision'] = average_precision_score(y_true, y_score)
    except Exception:
        out['AveragePrecision'] = np.nan
    try:
        out['ROCAUC'] = roc_auc_score(y_true, y_score)
    except Exception:
        out['ROCAUC'] = np.nan
    return out


def predict_scores(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, 'decision_function'):
        raw = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-raw))
    return model.predict(X)


def evaluate(df: pd.DataFrame, k: int, ablation: str, features: List[str], random_state: int, cv_folds: int) -> List[Dict]:
    sub = df[df['agreement_k'] == k].dropna(subset=['label_TOA']).copy()
    sub['label_TOA'] = sub['label_TOA'].astype(int)
    if len(sub) < 2 or sub['label_TOA'].nunique() < 2 or not features:
        return []
    X = sub[features].apply(pd.to_numeric, errors='coerce')
    y = sub['label_TOA'].to_numpy()
    groups = sub['article_url'].to_numpy() if 'article_url' in sub else np.arange(len(sub))
    n_splits = min(cv_folds, len(sub), int(np.bincount(y).min()) if len(np.bincount(y)) > 1 else 1)
    if n_splits < 2:
        return []
    splitter = GroupKFold(n_splits=n_splits)
    models = make_models(y, random_state)
    rows = []

    # baselines
    for name, constant in [('baseline_all_pos', 1), ('baseline_all_neg', 0)]:
        fold_scores = []
        for _, test_idx in splitter.split(X, y, groups):
            yp = np.full(len(test_idx), constant)
            ys = yp.astype(float)
            fold_scores.append(scores(y[test_idx], yp, ys))
        rows.append(summarize(k, name, ablation='n/a', fold_scores=fold_scores, n_splits=n_splits, y=y))

    for name, model in models.items():
        fold_scores = []
        for train_idx, test_idx in splitter.split(X, y, groups):
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                continue
            m = clone(model)
            m.fit(X.iloc[train_idx], y[train_idx])
            ys = predict_scores(m, X.iloc[test_idx])
            yp = (ys >= 0.5).astype(int)
            fold_scores.append(scores(y[test_idx], yp, ys))
        if fold_scores:
            rows.append(summarize(k, name, ablation, fold_scores, n_splits, y))
    return rows


def summarize(k, clf_name, ablation, fold_scores, n_splits, y):
    row = {'horizon': 'TA', 'outlier_maj': k, 'toa_min_pos': k, 'toa_max_neg': 0,
           'cv_n_splits': n_splits, 'clf_name': clf_name, 'ablation': ablation,
           'n_articles': len(y), 'n_pos_articles_est': int((y == 1).sum()), 'n_neg_articles_est': int((y == 0).sum())}
    for metric in ['F1', 'Precision', 'Recall', 'AveragePrecision', 'ROCAUC']:
        vals = [s[metric] for s in fold_scores if metric in s]
        row[f'{metric}_mean'] = float(np.nanmean(vals)) if vals else np.nan
        row[f'{metric}_std'] = float(np.nanstd(vals)) if vals else np.nan
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--feature-matrix', required=True)
    ap.add_argument('--thresholds', nargs='+', type=int, default=list(range(1, 9)))
    ap.add_argument('--cv-folds', type=int, default=5)
    ap.add_argument('--random-state', type=int, default=42)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    df = read_matrix(args.feature_matrix)
    groups = feature_groups(list(df.columns))
    rows = []
    for k in args.thresholds:
        for ablation, cols in groups.items():
            rows.extend(evaluate(df, k, ablation, cols, args.random_state, args.cv_folds))
    metrics = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out) as xw:
        metrics.to_excel(xw, sheet_name='ml_metrics_with_ablation', index=False)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
