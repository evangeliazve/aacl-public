#!/usr/bin/env python3
"""Run the supervised classifiers, feature-family ablations, and selected-setting ablation tests."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
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

ABLATION_ORDER = [
    "all_features",
    "no_geom",
    "no_social",
    "no_text",
    "only_geom",
    "only_social",
    "only_text",
]
MODEL_ORDER = ["xgb", "rf", "logreg", "linear_svc", "dt"]
PAPER_METRICS = ["F1", "Precision", "Recall"]

SELECTED_PAIRED_COMPARISONS = [
    ("all_features", "no_geom"),
    ("all_features", "only_geom"),
    ("all_features", "no_social"),
    ("all_features", "no_text"),
    ("only_geom", "only_text"),
    ("only_geom", "only_social"),
]


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


def scores(y_true, y_pred) -> Dict[str, float]:
    return {
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
    }


def predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-raw))
    return model.predict(X)


def summarize(
    k: int,
    clf_name: str,
    ablation: str | None,
    fold_scores: List[Dict[str, float]],
    n_splits: int,
    y: np.ndarray,
) -> Dict:
    row = {
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
    for metric in PAPER_METRICS:
        vals = [s[metric] for s in fold_scores if metric in s]
        row[f"{metric}_mean"] = float(np.nanmean(vals)) if vals else np.nan
        row[f"{metric}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else np.nan
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

    rows: List[Dict] = []
    splitter = GroupKFold(n_splits=n_splits)

    # Baseline rows are computed once per agreement threshold, not once per ablation.
    baseline_specs = [("baseline_all_pos", 1)]
    if include_all_negative_baseline:
        baseline_specs.append(("baseline_all_neg", 0))

    first_features = next(iter(groups.values()))
    X_dummy = sub[first_features].apply(pd.to_numeric, errors="coerce")

    for name, constant in baseline_specs:
        fold_scores = []
        for _, test_idx in splitter.split(X_dummy, y, groups_cv):
            yp = np.full(len(test_idx), constant)
            fold_scores.append(scores(y[test_idx], yp))
        rows.append(
            summarize(
                k,
                name,
                ablation="baseline",
                fold_scores=fold_scores,
                n_splits=n_splits,
                y=y,
            )
        )

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
                fold_scores.append(scores(y[test_idx], yp))
            if fold_scores:
                rows.append(summarize(k, name, ablation, fold_scores, n_splits, y))
    return rows


def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full(len(pvals), np.nan)

    valid = ~np.isnan(pvals)
    if valid.sum() == 0:
        return qvals

    pv = pvals[valid]
    order = np.argsort(pv)
    ranked = pv[order]

    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    out = np.empty_like(pv)
    out[order] = q
    qvals[valid] = out
    return qvals


def cohens_dz(diff):
    diff = np.asarray(diff, dtype=float)
    if len(diff) < 2:
        return np.nan

    sd = np.std(diff, ddof=1)
    if sd == 0:
        if np.mean(diff) == 0:
            return 0.0
        return np.inf

    return np.mean(diff) / sd


def significance_label(q):
    if pd.isna(q):
        return "n/a"
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return "ns"


def selected_setting_fold_ablation(
    df: pd.DataFrame,
    selected_k: int,
    groups: Dict[str, List[str]],
    random_state: int,
    cv_folds: int,
) -> pd.DataFrame:
    """
    Run selected-setting XGBoost ablations at one agreement threshold.

    This uses the same released feature matrix, feature-family groups, GroupKFold
    split logic, XGBoost model specification, and whole-selected-setting
    scale_pos_weight logic as the main ML experiment.
    """
    sub = df[df["agreement_k"] == selected_k].dropna(subset=["label_TOA"]).copy()
    if sub.empty:
        raise ValueError(f"agreement_k={selected_k} is not present in the feature matrix.")

    sub["label_TOA"] = sub["label_TOA"].astype(int)
    if len(sub) < 2 or sub["label_TOA"].nunique() < 2:
        raise ValueError(f"agreement_k={selected_k} has fewer than two classes after filtering.")

    y = sub["label_TOA"].to_numpy()
    groups_cv = sub["article_url"].to_numpy() if "article_url" in sub else np.arange(len(sub))
    n_splits = min(cv_folds, len(sub), int(np.bincount(y).min()) if len(np.bincount(y)) > 1 else 1)
    if n_splits < 2:
        raise ValueError(f"agreement_k={selected_k} has too few minority-class examples for cross-validation.")

    if "xgb" not in make_models(y, random_state):
        raise RuntimeError("XGBoost is not available. Install xgboost to run selected-setting ablation tests.")

    splitter = GroupKFold(n_splits=n_splits)
    xgb_model = make_models(y, random_state)["xgb"]

    fold_records = []

    for ablation in [
        "all_features",
        "only_geom",
        "only_text",
        "only_social",
        "no_geom",
        "no_social",
        "no_text",
    ]:
        if ablation not in groups:
            print(f"warning: ablation={ablation} has no feature columns; skipped")
            continue

        cols = groups[ablation]
        X = sub[cols].apply(pd.to_numeric, errors="coerce")

        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups_cv), start=1):
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                print(f"warning: selected k={selected_k}, fold={fold_id} skipped because train/test has one class only")
                continue

            m = clone(xgb_model)
            m.fit(X.iloc[train_idx], y[train_idx])
            ys = predict_scores(m, X.iloc[test_idx])
            yp = (ys >= 0.5).astype(int)
            fold_scores = scores(y[test_idx], yp)

            fold_records.append({
                "outlier_maj": selected_k,
                "toa_min_pos": selected_k,
                "toa_max_neg": 0,
                "clf_name": "xgb",
                "ablation": ablation,
                "fold": fold_id,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_train_pos": int(y[train_idx].sum()),
                "n_train_neg": int(len(train_idx) - y[train_idx].sum()),
                "n_test_pos": int(y[test_idx].sum()),
                "n_test_neg": int(len(test_idx) - y[test_idx].sum()),
                "n_features": int(len(cols)),
                "F1": float(fold_scores["F1"]),
                "Precision": float(fold_scores["Precision"]),
                "Recall": float(fold_scores["Recall"]),
            })

    return pd.DataFrame(fold_records)


def paired_ablation_tests(
    df_fold: pd.DataFrame,
    selected_k: int,
    comparisons=None,
) -> pd.DataFrame:
    if comparisons is None:
        comparisons = SELECTED_PAIRED_COMPARISONS

    test_rows = []

    for reference, comparison in comparisons:
        for metric in PAPER_METRICS:
            wide = (
                df_fold[df_fold["ablation"].isin([reference, comparison])]
                .pivot_table(index="fold", columns="ablation", values=metric, aggfunc="mean")
                .dropna()
            )

            if reference not in wide.columns or comparison not in wide.columns:
                continue
            if len(wide) < 2:
                continue

            ref_values = wide[reference].astype(float).values
            comp_values = wide[comparison].astype(float).values
            diff = ref_values - comp_values

            try:
                t_stat, p_t = ttest_rel(ref_values, comp_values)
            except Exception:
                t_stat, p_t = np.nan, np.nan

            test_rows.append({
                "k": selected_k,
                "clf_name": "xgb",
                "metric": metric,
                "reference": reference,
                "comparison": comparison,
                "n_folds": int(len(wide)),
                "reference_mean": float(np.mean(ref_values)),
                "comparison_mean": float(np.mean(comp_values)),
                "mean_diff_ref_minus_comp": float(np.mean(diff)),
                "std_diff": float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan,
                "cohens_dz": float(cohens_dz(diff)),
                "paired_t_stat": float(t_stat) if not pd.isna(t_stat) else np.nan,
                "paired_t_p": float(p_t) if not pd.isna(p_t) else np.nan,
                "diffs_by_fold": ", ".join([f"{x:.4f}" for x in diff]),
            })

    df_tests = pd.DataFrame(test_rows)

    test_cols = [
        "k",
        "clf_name",
        "metric",
        "reference",
        "comparison",
        "n_folds",
        "reference_mean",
        "comparison_mean",
        "mean_diff_ref_minus_comp",
        "std_diff",
        "cohens_dz",
        "paired_t_stat",
        "paired_t_p",
        "diffs_by_fold",
        "paired_t_q_fdr",
        "paired_t_sig",
    ]

    if df_tests.empty:
        return pd.DataFrame(columns=test_cols)

    df_tests["paired_t_q_fdr"] = bh_fdr(df_tests["paired_t_p"].values)

    # Significance label is based on the BH-corrected q-value, matching the
    # paper caption: significance after Benjamini-Hochberg correction.
    df_tests["paired_t_sig"] = df_tests["paired_t_q_fdr"].apply(significance_label)

    return df_tests[test_cols].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-matrix", required=True)
    ap.add_argument("--thresholds", nargs="+", type=int, default=list(range(1, 9)))
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--include-all-negative-baseline", action="store_true")
    ap.add_argument("--selected-ablation-k", type=int, default=None)
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

    metric_cols = [
        "outlier_k",
        "toa_k",
        "toa_max_neg",
        "cv_n_splits",
        "clf_name",
        "ablation",
        "n_articles",
        "n_pos_articles_est",
        "n_neg_articles_est",
        "F1_mean",
        "F1_std",
        "Precision_mean",
        "Precision_std",
        "Recall_mean",
        "Recall_std",
    ]
    metrics = metrics[[c for c in metric_cols if c in metrics.columns]].copy()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        metrics.to_excel(xw, sheet_name="ml_metrics_with_ablation", index=False)

        if args.selected_ablation_k is not None:
            df_fold = selected_setting_fold_ablation(
                df=df,
                selected_k=args.selected_ablation_k,
                groups=groups,
                random_state=args.random_state,
                cv_folds=args.cv_folds,
            )
            df_tests = paired_ablation_tests(
                df_fold=df_fold,
                selected_k=args.selected_ablation_k,
            )

            df_fold.to_excel(xw, sheet_name="fold_ablation", index=False)
            df_tests.to_excel(xw, sheet_name="fold_ablation_paired_tests", index=False)

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
