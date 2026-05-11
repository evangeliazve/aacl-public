#!/usr/bin/env python3
"""Run supervised ML experiments and feature-family ablations.

The released feature matrices contain one row per article and agreement threshold.
This script evaluates the diagonal consensus settings used in the release:
`outlier_k = toa_k = agreement_k` and `toa_max_neg = 0`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

GEOM_PREFIXES = (
    "d1_", "d2_", "margin_", "mahal_", "knn_", "outlier_proto_", "outlier_score",
    "has_recent_outliers", "n_recent_outliers", "n_models_present",
)
TEXT_PREFIXES = ("text_", "avg_", "total_syllables", "len_", "ner_")
SOCIAL_PREFIXES = ("soc_", "media_")
META_COLS = {"article_url", "agreement_k", "label_TOA", "horizon"}
ABLATION_ORDER = ["all_features", "no_geom", "no_social", "no_text", "only_geom", "only_social", "only_text"]
MODEL_ORDER = ["xgb", "rf", "logreg", "linear_svc", "dt"]


def read_matrix(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p, sheet_name="feature_matrix")
    return pd.read_csv(p)


def feature_groups(columns: List[str]) -> Dict[str, List[str]]:
    feats = [c for c in columns if c not in META_COLS]
    geom = [c for c in feats if c.startswith(GEOM_PREFIXES)]
    text = [c for c in feats if c.startswith(TEXT_PREFIXES)]
    social = [c for c in feats if c.startswith(SOCIAL_PREFIXES)]
    groups = {
        "all_features": feats,
        "no_geom": [c for c in feats if c not in geom],
        "no_social": [c for c in feats if c not in social],
        "no_text": [c for c in feats if c not in text],
        "only_geom": geom,
        "only_social": social,
        "only_text": text,
    }
    return {name: groups[name] for name in ABLATION_ORDER if groups.get(name)}


def make_models(y: np.ndarray, random_state: int) -> Dict[str, object]:
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = int((y == 0).sum())
    scale_pos_weight = n_neg / n_pos

    models: Dict[str, object] = {}
    try:
        from xgboost import XGBClassifier
        models["xgb"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
                n_jobs=-1,
            )),
        ])
    except Exception:
        pass

    models.update({
        "rf": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=600,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
        "logreg": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                solver="liblinear",
                max_iter=2000,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
        "linear_svc": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(max_iter=5000, class_weight="balanced", random_state=random_state)),
        ]),
        "dt": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(
                max_depth=6,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
    })
    return {name: models[name] for name in MODEL_ORDER if name in models}


def scores(y_true, y_pred, y_score) -> Dict[str, float]:
    out = {
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
    }
    try:
        out["AP"] = average_precision_score(y_true, y_score)
    except Exception:
        out["AP"] = np.nan
    try:
        out["ROC_AUC"] = roc_auc_score(y_true, y_score)
    except Exception:
        out["ROC_AUC"] = np.nan
    return out


def predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-raw))
    return model.predict(X)


def summarize(k: int, clf_name: str, ablation: str | None, fold_scores: List[Dict[str, float]], n_splits: int, y: np.ndarray) -> Dict:
    row = {
        "horizon": "TA",
        "outlier_k": k,
        "toa_k": k,
        "toa_max_neg": 0,
        "cv_n_splits": n_splits,
        "clf_name": clf_name,
        "ablation": ablation,
        "n_articles": len(y),
        "n_pos_articles_est": int((y == 1).sum()),
        "n_neg_articles_est": int((y == 0).sum()),
    }
    for metric in ["F1", "Precision", "Recall", "AP", "ROC_AUC"]:
        vals = [s[metric] for s in fold_scores if metric in s]
        row[f"{metric}_mean"] = float(np.nanmean(vals)) if vals else np.nan
        row[f"{metric}_std"] = float(np.nanstd(vals)) if vals else np.nan
    return row


def evaluate_threshold(
    df: pd.DataFrame,
    k: int,
    groups: Dict[str, List[str]],
    random_state: int,
    cv_folds: int,
    include_all_negative_baseline: bool = False,
) -> List[Dict]:
    sub = df[df["agreement_k"] == k].dropna(subset=["label_TOA"]).copy()
    if sub.empty:
        print(f"warning: agreement_k={k} is not present in the feature matrix; skipped")
        return []
    sub["label_TOA"] = sub["label_TOA"].astype(int)
    if len(sub) < 2 or sub["label_TOA"].nunique() < 2:
        print(f"warning: agreement_k={k} has fewer than two classes after filtering; skipped")
        return []

    y = sub["label_TOA"].to_numpy()
    groups_cv = sub["article_url"].to_numpy() if "article_url" in sub else np.arange(len(sub))
    n_splits = min(cv_folds, len(sub), int(np.bincount(y).min()) if len(np.bincount(y)) > 1 else 1)
    if n_splits < 2:
        print(f"warning: agreement_k={k} has too few minority-class examples for cross-validation; skipped")
        return []

    # Baseline rows are computed once per agreement threshold, not once per ablation.
    rows: List[Dict] = []
    splitter = GroupKFold(n_splits=n_splits)
    baseline_specs = [("baseline_all_pos", 1)]
    if include_all_negative_baseline:
        baseline_specs.append(("baseline_all_neg", 0))
    for name, constant in baseline_specs:
        first_features = next(iter(groups.values()))
        X_dummy = sub[first_features].apply(pd.to_numeric, errors="coerce")
        fold_scores = []
        for _, test_idx in splitter.split(X_dummy, y, groups_cv):
            yp = np.full(len(test_idx), constant)
            fold_scores.append(scores(y[test_idx], yp, yp.astype(float)))
        rows.append(summarize(k, name, ablation="n/a", fold_scores=fold_scores, n_splits=n_splits, y=y))

    models = make_models(y, random_state)
    for ablation, cols in groups.items():
        if not cols:
            continue
        X = sub[cols].apply(pd.to_numeric, errors="coerce")
        for name, model in models.items():
            fold_scores = []
            for train_idx, test_idx in splitter.split(X, y, groups_cv):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-matrix", required=True)
    ap.add_argument("--thresholds", nargs="+", type=int, default=list(range(1, 9)))
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--include-all-negative-baseline", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = read_matrix(args.feature_matrix)
    groups = feature_groups(list(df.columns))
    rows: List[Dict] = []
    for k in args.thresholds:
        rows.extend(evaluate_threshold(
            df,
            k,
            groups,
            args.random_state,
            args.cv_folds,
            include_all_negative_baseline=args.include_all_negative_baseline,
        ))

    metrics = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out) as xw:
        metrics.to_excel(xw, sheet_name="ml_metrics_with_ablation", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
