#!/usr/bin/env python3
"""Reconstruct outlier trajectories and build the model-agreement matrix.

Input is the `models/*/results.csv` created by 01_dynamic_topic_reconstruction.py.
The output matrix has one row per article and one binary TOA vote column per model.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def first_topic_creation_time(df: pd.DataFrame) -> pd.Series:
    assigned = df[df["topic_id"].astype(int) >= 0]
    return assigned.groupby("topic_id")["snapshot_date"].min()


def annotate_model(
    results: pd.DataFrame,
    id_col: str,
    date_col: str,
    model_name: str,
    cfg: dict,
) -> pd.DataFrame:
    df = results.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df["topic_id"] = pd.to_numeric(df["topic_id"], errors="coerce").fillna(-1).astype(int)
    df["is_outlier"] = df["is_outlier"].astype(str).str.lower().isin(["true", "1", "yes"])

    # Match the notebook's warm-up filtering: ignore the first N cumulative snapshots
    # before computing trajectory labels, because the earliest topic snapshots are unstable.
    warmup_days = int(cfg.get("labeling", {}).get("warmup_days", 0))
    if warmup_days > 0 and df["snapshot_date"].notna().any():
        start = df["snapshot_date"].min()
        cutoff = start + pd.Timedelta(days=warmup_days)
        df = df[df["snapshot_date"] >= cutoff].copy()

    topic_tt = first_topic_creation_time(df)
    rows: List[Dict] = []
    for article_id, g in df.sort_values("snapshot_date").groupby(id_col):
        pub = pd.to_datetime(g[date_col].iloc[0])
        at_pub = g[g["snapshot_date"] >= pub]
        if at_pub.empty:
            continue
        row_ta = at_pub.iloc[0]
        outlier_at_pub = bool(row_ta["is_outlier"] or int(row_ta["topic_id"]) == -1)

        later_assigned = g[(g["snapshot_date"] >= row_ta["snapshot_date"]) & (g["topic_id"] >= 0)]
        final_topic = pd.NA
        tt = pd.NaT
        ti = pd.NaT
        category = "not_publication_outlier" if not outlier_at_pub else "Oold"
        toa_vote = 0

        if outlier_at_pub and not later_assigned.empty:
            first_join = later_assigned.iloc[0]
            final_topic = int(first_join["topic_id"])
            ti = pd.to_datetime(first_join["snapshot_date"])
            tt = pd.to_datetime(topic_tt.loc[final_topic]) if final_topic in topic_tt.index else pd.NaT
            if pd.notna(tt) and pub < tt <= ti:
                category = "TOAfirst" if tt == ti else "TOAlate"
                toa_vote = 1
            elif pd.notna(tt) and tt <= pub < ti:
                category = "TODlate"
            else:
                category = "other"

        rows.append({
            id_col: article_id,
            "publication_date_cleaned": pub,
            f"outlier_{model_name}": int(outlier_at_pub),
            f"toa_{model_name}": int(toa_vote),
            f"trajectory_{model_name}": category,
            f"first_topic_id_{model_name}": final_topic,
            f"first_topic_time_{model_name}": tt,
            f"integration_time_{model_name}": ti,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input-dir", default=None, help="Directory containing models/*/results.csv. Defaults to output_dir from config.")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    id_col = cfg["input"]["article_id_col"]
    date_col = cfg["input"]["date_col"]
    root = Path(args.input_dir or cfg["output_dir"])
    out_path = Path(args.output or root / "trajectory_matrix.xlsx")

    matrices = []
    for result_file in sorted((root / "models").glob("*/results.csv")):
        model = result_file.parent.name
        df = pd.read_csv(result_file)
        matrices.append(annotate_model(df, id_col, date_col, model, cfg))

    if not matrices:
        raise FileNotFoundError(f"No results.csv files found below {root / 'models'}")

    wide = matrices[0]
    for m in matrices[1:]:
        wide = wide.merge(m, on=[id_col, "publication_date_cleaned"], how="outer")

    outlier_cols = [c for c in wide.columns if c.startswith("outlier_")]
    toa_cols = [c for c in wide.columns if c.startswith("toa_")]
    wide["n_outlier_votes"] = wide[outlier_cols].fillna(0).sum(axis=1).astype(int)
    wide["n_TOA_votes"] = wide[toa_cols].fillna(0).sum(axis=1).astype(int)
    wide["n_models"] = len(toa_cols)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path) as xw:
        wide.to_excel(xw, sheet_name="TOA matrix and agreement", index=False)
    wide.to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
