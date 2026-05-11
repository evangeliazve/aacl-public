#!/usr/bin/env python3
"""Calculate consensus labels and agreement diagnostics from a trajectory matrix."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def read_table(path: str | Path, sheet: str | None = None) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p, sheet_name=sheet or 0)
    return pd.read_csv(p)


def fleiss_kappa(binary_votes: pd.DataFrame) -> float:
    votes = binary_votes.fillna(0).astype(int).to_numpy()
    n_items, n_raters = votes.shape
    if n_items == 0 or n_raters <= 1:
        return np.nan
    counts = np.column_stack([(votes == 0).sum(axis=1), (votes == 1).sum(axis=1)])
    p = counts.sum(axis=0) / (n_items * n_raters)
    pbar = ((counts * (counts - 1)).sum(axis=1) / (n_raters * (n_raters - 1))).mean()
    pe = (p ** 2).sum()
    return float((pbar - pe) / (1 - pe)) if pe < 1 else np.nan


def label_for_k(row: pd.Series, k: int, toa_max_neg: int = 0) -> float:
    if row["n_outlier_votes"] < k:
        return np.nan
    if row["n_TOA_votes"] >= k:
        return 1.0
    if row["n_TOA_votes"] <= toa_max_neg:
        return 0.0
    return np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--sheet", default="TOA matrix and agreement")
    ap.add_argument("--max-k", type=int, default=11)
    ap.add_argument("--toa-max-neg", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = read_table(args.matrix, args.sheet)
    toa_cols = [c for c in df.columns if c.startswith("toa_")]
    outlier_cols = [c for c in df.columns if c.startswith("outlier_")]
    if "n_TOA_votes" not in df:
        df["n_TOA_votes"] = df[toa_cols].fillna(0).sum(axis=1).astype(int)
    if "n_outlier_votes" not in df:
        df["n_outlier_votes"] = df[outlier_cols].fillna(0).sum(axis=1).astype(int)

    summaries = []
    labeled = df.copy()
    for k in range(1, args.max_k + 1):
        col = f"label_k{k}"
        labeled[col] = labeled.apply(label_for_k, axis=1, k=k, toa_max_neg=args.toa_max_neg)
        kept = labeled[col].notna()
        summaries.append({
            "agreement_k": k,
            "retained": int(kept.sum()),
            "positive": int((labeled[col] == 1).sum()),
            "negative": int((labeled[col] == 0).sum()),
            "coverage": float(kept.mean()),
        })

    model_kappa = fleiss_kappa(df[toa_cols]) if toa_cols else np.nan
    summary = pd.DataFrame(summaries)
    summary["fleiss_kappa_TOA_votes"] = model_kappa

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out) as xw:
        labeled.to_excel(xw, sheet_name="article_labels", index=False)
        summary.to_excel(xw, sheet_name="agreement_summary", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
