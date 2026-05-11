#!/usr/bin/env python3
"""XGBoost SHAP interpretation for the selected agreement threshold."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

META = {"article_url", "agreement_k", "label_TOA", "horizon"}


def read_matrix(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p, sheet_name="feature_matrix")
    return pd.read_csv(p)


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META]


def xgb_model(y: np.ndarray, random_state: int) -> XGBClassifier:
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = int((y == 0).sum())
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=n_neg / n_pos,
        random_state=random_state,
        n_jobs=-1,
    )


def sig(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def direction(r: float) -> str:
    if pd.isna(r):
        return "no_monotonic_direction"
    return "higher_feature_value_pushes_toward_TOA" if r > 0 else "higher_feature_value_pushes_away_from_TOA"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-matrix", required=True)
    ap.add_argument("--agreement-k", type=int, required=True)
    ap.add_argument("--dataset-name", default="dataset", help="Accepted for filename/logging compatibility; not written to the released workbook sheets.")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = read_matrix(args.feature_matrix)
    df = df[df["agreement_k"] == args.agreement_k].dropna(subset=["label_TOA"]).copy()
    if df.empty:
        raise ValueError(f"No labeled rows found for agreement_k={args.agreement_k}")
    df["label_TOA"] = df["label_TOA"].astype(int)
    if df["label_TOA"].nunique() < 2:
        raise ValueError(f"agreement_k={args.agreement_k} has fewer than two target classes")

    feats = feature_cols(df)
    X_raw = df[feats].apply(pd.to_numeric, errors="coerce")
    y = df["label_TOA"].to_numpy()
    groups = df["article_url"].to_numpy() if "article_url" in df else np.arange(len(df))

    imp = SimpleImputer(strategy="median")
    X = pd.DataFrame(imp.fit_transform(X_raw), columns=feats, index=df.index)

    final_model = xgb_model(y, args.random_state)
    final_model.fit(X, y)
    explainer = shap.TreeExplainer(final_model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]

    global_rows = []
    for j, feat in enumerate(feats):
        try:
            r, p = spearmanr(X[feat], sv[:, j])
        except Exception:
            r, p = np.nan, np.nan
        global_rows.append({
            "feature": feat,
            "mean_abs_shap": float(np.abs(sv[:, j]).mean()),
            "mean_shap": float(sv[:, j].mean()),
            "mean_feature_value": float(X[feat].mean()),
            "median_feature_value": float(X[feat].median()),
            "corr_feature_shap_spearman": r,
            "corr_feature_shap_pvalue": p,
            "corr_feature_shap_sig": sig(p),
            "direction_hint": direction(r),
            "n_articles": len(df),
            "n_pos_articles_est": int((y == 1).sum()),
            "n_neg_articles_est": int((y == 0).sum()),
        })
    global_df = pd.DataFrame(global_rows).sort_values("mean_abs_shap", ascending=False)

    n_splits = min(args.cv_folds, len(df), int(np.bincount(y).min()) if len(np.bincount(y)) > 1 else 1)
    local_pred_rows = []
    local_long_rows = []
    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        for train_idx, test_idx in gkf.split(X, y, groups):
            if len(np.unique(y[train_idx])) < 2:
                continue
            m = xgb_model(y[train_idx], args.random_state)
            m.fit(X.iloc[train_idx], y[train_idx])
            ex = shap.TreeExplainer(m)
            shap_test = ex.shap_values(X.iloc[test_idx])
            if isinstance(shap_test, list):
                shap_test = shap_test[1]
            proba = m.predict_proba(X.iloc[test_idx])[:, 1]
            base_value = float(np.ravel(ex.expected_value)[0])

            for row_pos, idx in enumerate(test_idx):
                url = df.iloc[idx].get("article_url", idx)
                local_pred_rows.append({
                    "media_url": url,
                    "y": int(y[idx]),
                    "pred_proba": float(proba[row_pos]),
                    "base_value": base_value,
                })
                for j, feat in enumerate(feats):
                    sh = float(shap_test[row_pos, j])
                    local_long_rows.append({
                        "media_url": url,
                        "feature": feat,
                        "feature_value": float(X.iloc[idx, j]),
                        "shap_value": sh,
                        "y": int(y[idx]),
                        "pred_proba": float(proba[row_pos]),
                        "abs_shap": abs(sh),
                        "impact_direction": "pushes_toward_TOA" if sh > 0 else "pushes_away_from_TOA",
                    })
    else:
        print("warning: too few minority-class examples for out-of-fold SHAP predictions")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out) as xw:
        global_df.to_excel(xw, sheet_name="xgb_global_shap", index=False)
        pd.DataFrame(local_pred_rows).to_excel(xw, sheet_name="xgb_local_predictions", index=False)
        pd.DataFrame(local_long_rows).to_excel(xw, sheet_name="xgb_local_shap_long", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
