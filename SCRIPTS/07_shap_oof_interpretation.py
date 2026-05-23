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
    if r > 0:
        return "higher_feature_value_pushes_toward_TOA"
    if r < 0:
        return "higher_feature_value_pushes_away_from_TOA"
    return "no_monotonic_direction"


def impact_direction(sh: float) -> str:
    if sh > 0:
        return "pushes_toward_TOA"
    if sh < 0:
        return "pushes_away_from_TOA"
    return "no_effect"


def probability_shap_values(
    model: XGBClassifier,
    X: np.ndarray,
    background: np.ndarray,
    random_state: int,
    max_bg: int,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(random_state)
    bg_idx = rng.choice(len(background), size=min(len(background), max_bg), replace=False)
    bg = background[bg_idx]

    explainer = shap.TreeExplainer(
        model,
        data=bg,
        model_output="probability",
        feature_perturbation="interventional",
    )

    sv = explainer.shap_values(X, check_additivity=False)

    if isinstance(sv, list):
        sv = sv[1]
    elif getattr(sv, "ndim", 0) == 3:
        sv = sv[:, :, 1]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = np.ravel(base_value)
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]

    return np.asarray(sv), float(base_value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-matrix", required=True)
    ap.add_argument("--agreement-k", type=int, required=True)
    ap.add_argument("--dataset-name", default="dataset")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--max-bg", type=int, default=200)
    ap.add_argument("--output", required=True)

    # Use xgb_local_shap to match the Hydro workbook.
    # Use xgb_all_local_shap to match the Climate workbook.
    ap.add_argument("--all-local-sheet-name", default="xgb_local_shap")

    args = ap.parse_args()

    df = read_matrix(args.feature_matrix)
    df = df[df["agreement_k"] == args.agreement_k].dropna(subset=["label_TOA"]).copy()
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No labeled rows found for agreement_k={args.agreement_k}")

    df["label_TOA"] = df["label_TOA"].astype(int)

    if df["label_TOA"].nunique() < 2:
        raise ValueError(f"agreement_k={args.agreement_k} has fewer than two target classes")

    feats = feature_cols(df)
    X_raw = df[feats].apply(pd.to_numeric, errors="coerce")
    y = df["label_TOA"].to_numpy()
    groups = df["article_url"].to_numpy() if "article_url" in df.columns else np.arange(len(df))

    # ------------------------------------------------------------------
    # Final-model SHAP, aligned with notebook probability-scale logic.
    # ------------------------------------------------------------------
    final_imputer = SimpleImputer(strategy="median")
    X_all = final_imputer.fit_transform(X_raw)

    final_model = xgb_model(y, args.random_state)
    final_model.fit(X_all, y)

    sv_all, base_all = probability_shap_values(
        model=final_model,
        X=X_all,
        background=X_all,
        random_state=args.random_state,
        max_bg=args.max_bg,
    )

    X_all_df = pd.DataFrame(X_all, columns=feats)
    sv_all_df = pd.DataFrame(sv_all, columns=feats)

    global_rows = []
    for feat in feats:
        x = X_all_df[feat]
        s = sv_all_df[feat]

        try:
            if x.nunique() > 1 and s.nunique() > 1:
                r, p = spearmanr(x, s, nan_policy="omit")
            else:
                r, p = np.nan, np.nan
        except Exception:
            r, p = np.nan, np.nan

        global_rows.append(
            {
                "feature": feat,
                "mean_abs_shap": float(np.abs(s).mean()),
                "mean_shap": float(s.mean()),
                "mean_feature_value": float(x.mean()),
                "median_feature_value": float(x.median()),
                "corr_feature_shap_spearman": float(r) if not pd.isna(r) else np.nan,
                "corr_feature_shap_pvalue": float(p) if not pd.isna(p) else np.nan,
                "corr_feature_shap_sig": sig(p),
                "direction_hint": direction(r),
                "n_articles": int(len(df)),
                "n_pos_articles_est": int((y == 1).sum()),
                "n_neg_articles_est": int((y == 0).sum()),
            }
        )

    global_df = (
        pd.DataFrame(global_rows)
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # Final-model local SHAP for all selected articles.
    all_local_rows = []
    final_proba = final_model.predict_proba(X_all)[:, 1]

    for i in range(len(df)):
        url = df.loc[i, "article_url"] if "article_url" in df.columns else i
        for j, feat in enumerate(feats):
            sh = float(sv_all[i, j])
            all_local_rows.append(
                {
                    "media_url": url,
                    "feature": feat,
                    "feature_value": float(X_all[i, j]),
                    "shap_value": sh,
                    "y": int(y[i]),
                    "pred_proba": float(final_proba[i]),
                    "impact_direction": impact_direction(sh),
                }
            )

    all_local_df = pd.DataFrame(
        all_local_rows,
        columns=[
            "media_url",
            "feature",
            "feature_value",
            "shap_value",
            "y",
            "pred_proba",
            "impact_direction",
        ],
    )

    # ------------------------------------------------------------------
    # Out-of-fold local SHAP.
    # Imputer is fit inside each fold.
    # ------------------------------------------------------------------
    n_splits = min(
        args.cv_folds,
        len(df),
        int(np.bincount(y).min()) if len(np.bincount(y)) > 1 else 1,
    )

    oof_rows = []

    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)

        for fold_id, (train_idx, test_idx) in enumerate(gkf.split(X_raw, y, groups), start=1):
            if len(np.unique(y[train_idx])) < 2:
                continue

            fold_imputer = SimpleImputer(strategy="median")
            X_train = fold_imputer.fit_transform(X_raw.iloc[train_idx])
            X_test = fold_imputer.transform(X_raw.iloc[test_idx])

            model = xgb_model(y[train_idx], args.random_state)
            model.fit(X_train, y[train_idx])

            proba = model.predict_proba(X_test)[:, 1]

            sv_test, _ = probability_shap_values(
                model=model,
                X=X_test,
                background=X_train,
                random_state=args.random_state + fold_id,
                max_bg=args.max_bg,
            )

            for row_pos, idx in enumerate(test_idx):
                url = df.loc[idx, "article_url"] if "article_url" in df.columns else idx
                for j, feat in enumerate(feats):
                    sh = float(sv_test[row_pos, j])
                    oof_rows.append(
                        {
                            "media_url": url,
                            "feature": feat,
                            "feature_value": float(X_test[row_pos, j]),
                            "shap_value": sh,
                            "impact_direction": impact_direction(sh),
                            "fold": int(fold_id),
                            "y": int(y[idx]),
                            "pred_proba": float(proba[row_pos]),
                        }
                    )
    else:
        print("warning: too few minority-class examples for out-of-fold SHAP predictions")

    oof_df = pd.DataFrame(
        oof_rows,
        columns=[
            "media_url",
            "feature",
            "feature_value",
            "shap_value",
            "impact_direction",
            "fold",
            "y",
            "pred_proba",
        ],
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out) as xw:
        global_df.to_excel(xw, sheet_name="xgb_global_shap", index=False)
        oof_df.to_excel(xw, sheet_name="xgb_oof_local_shap", index=False)
        all_local_df.to_excel(xw, sheet_name=args.all_local_sheet_name, index=False)

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
