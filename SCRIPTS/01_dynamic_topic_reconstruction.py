#!/usr/bin/env python3
"""Build cumulative dynamic topic snapshots and article-level trajectory inputs.

For each embedding model, the script creates one output folder containing:
  - embeddings.npy
  - snapshots.parquet/csv: article assignments in cumulative daily snapshots
  - topics.parquet/csv: aligned topic metadata per snapshot
  - results.csv: compact table used by downstream trajectory annotation

The implementation intentionally keeps the workflow script-based rather than a package.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_distances


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return text[:120] or "model"


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, path_no_suffix: Path) -> None:
    path_no_suffix.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path_no_suffix.with_suffix(".parquet"), index=False)
    except Exception:
        df.to_csv(path_no_suffix.with_suffix(".csv"), index=False)


def clean_articles(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    icfg = cfg["input"]
    out = df.copy()
    id_col = icfg["article_id_col"]
    date_col = icfg["date_col"]
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce", utc=True).dt.tz_convert(None)
    out = out.dropna(subset=[id_col, date_col]).drop_duplicates(id_col).copy()
    out = out.sort_values(date_col).reset_index(drop=True)
    out["snapshot_date"] = out[date_col].dt.floor(cfg.get("clustering", {}).get("snapshot_frequency", "D"))
    return out


def article_texts(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[str]:
    tpl = cfg.get("embedding", {}).get("text_template", "{title} {description}")
    title_col = cfg["input"].get("title_col", "title")
    text_col = cfg["input"].get("text_col", "description")
    texts = []
    for _, row in df.iterrows():
        title = "" if pd.isna(row.get(title_col, "")) else str(row.get(title_col, ""))
        desc = "" if pd.isna(row.get(text_col, "")) else str(row.get(text_col, ""))
        texts.append(tpl.format(title=title, description=desc).strip())
    return texts


def load_or_compute_embeddings(df: pd.DataFrame, model_cfg: Dict[str, Any], cfg: Dict[str, Any], out_dir: Path) -> np.ndarray:
    short = model_cfg.get("short_name") or slug(model_cfg["name"])
    cache = out_dir / "embeddings.npy"
    if cache.exists():
        return np.load(cache)

    pre_dir = cfg.get("embedding", {}).get("precomputed_dir")
    if pre_dir:
        for suffix in (".npy", ".csv", ".parquet"):
            p = Path(pre_dir) / f"{short}{suffix}"
            if p.exists():
                if suffix == ".npy":
                    arr = np.load(p)
                elif suffix == ".parquet":
                    arr = pd.read_parquet(p).to_numpy(dtype=float)
                else:
                    arr = pd.read_csv(p).to_numpy(dtype=float)
                np.save(cache, arr)
                return arr

    if model_cfg.get("type") != "sentence_transformer":
        raise ValueError(
            f"No precomputed embeddings found for {short}. "
            "For API-based models, create data/embeddings/<short_name>.npy or .csv first."
        )
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_cfg["name"])
    arr = model.encode(article_texts(df, cfg), batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    arr = np.asarray(arr, dtype=np.float32)
    np.save(cache, arr)
    return arr


def reduce_embeddings(x: np.ndarray, cfg: Dict[str, Any], random_state: int) -> np.ndarray:
    ccfg = cfg["clustering"]
    if ccfg.get("reducer", "umap").lower() != "umap":
        raise ValueError("Only UMAP is supported in this reproduction script.")
    import umap.umap_ as umap

    reducer = umap.UMAP(
        n_components=int(ccfg.get("n_components", 20)),
        n_neighbors=int(ccfg.get("umap_n_neighbors", 15)),
        min_dist=float(ccfg.get("umap_min_dist", 0.0)),
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(x)


def cluster_snapshot(x: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    ccfg = cfg["clustering"]
    if ccfg.get("clusterer", "hdbscan").lower() == "hdbscan":
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(ccfg.get("min_cluster_size", 5)),
            min_samples=int(ccfg.get("min_samples", 5)),
            prediction_data=False,
        )
        labels = clusterer.fit_predict(x)
        scores = getattr(clusterer, "outlier_scores_", np.zeros(len(x), dtype=float))
        scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
        return labels.astype(int), scores.astype(float)
    raise ValueError("Only HDBSCAN is supported in this reproduction script.")


def cluster_centroids(x: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    return {int(c): x[labels == c].mean(axis=0) for c in sorted(set(labels)) if int(c) != -1}


def align_topics(prev: Dict[int, np.ndarray], cur: Dict[int, np.ndarray], next_topic_id: int, threshold: float) -> Tuple[Dict[int, int], int]:
    if not cur:
        return {}, next_topic_id
    if not prev:
        out = {}
        for c in sorted(cur):
            out[c] = next_topic_id
            next_topic_id += 1
        return out, next_topic_id

    prev_labels = list(prev)
    cur_labels = list(cur)
    cost = cosine_distances(np.vstack([prev[k] for k in prev_labels]), np.vstack([cur[k] for k in cur_labels]))
    r_ind, c_ind = linear_sum_assignment(cost)
    matched_cur = set()
    out: Dict[int, int] = {}
    for r, c in zip(r_ind, c_ind):
        cur_cluster = cur_labels[c]
        if cost[r, c] <= threshold:
            out[cur_cluster] = prev_labels[r]
            matched_cur.add(cur_cluster)
    for c in cur_labels:
        if c not in matched_cur:
            out[c] = next_topic_id
            next_topic_id += 1
    return out, next_topic_id


def run_model(df: pd.DataFrame, embeddings: np.ndarray, model_cfg: Dict[str, Any], cfg: Dict[str, Any], out_dir: Path) -> None:
    id_col = cfg["input"]["article_id_col"]
    date_col = cfg["input"]["date_col"]
    random_state = int(cfg.get("random_state", 42))
    threshold = float(cfg["clustering"].get("alignment_threshold", 0.30))

    z = reduce_embeddings(embeddings, cfg, random_state)
    dim_cols = [f"umap_{i}" for i in range(z.shape[1])]
    z_df = pd.DataFrame(z, columns=dim_cols)
    z_df[id_col] = df[id_col].to_numpy()

    rows: List[pd.DataFrame] = []
    topic_rows: List[Dict[str, Any]] = []
    prev_topic_centroids: Dict[int, np.ndarray] = {}
    next_topic_id = 0

    for snap_date in sorted(df["snapshot_date"].dropna().unique()):
        idx = np.flatnonzero(df["snapshot_date"].to_numpy() <= snap_date)
        x_snap = z[idx]
        labels, scores = cluster_snapshot(x_snap, cfg)
        local_centroids = cluster_centroids(x_snap, labels)
        topic_map, next_topic_id = align_topics(prev_topic_centroids, local_centroids, next_topic_id, threshold)

        snap = df.iloc[idx][[id_col, date_col, "snapshot_date"]].copy()
        snap["snapshot_date"] = pd.Timestamp(snap_date)
        snap["model"] = model_cfg.get("short_name") or slug(model_cfg["name"])
        snap["cluster_id"] = labels
        snap["topic_id"] = [topic_map.get(int(c), -1) if int(c) != -1 else -1 for c in labels]
        snap["is_outlier"] = labels == -1
        snap["outlier_score"] = scores
        snap = pd.concat([snap.reset_index(drop=True), pd.DataFrame(x_snap, columns=dim_cols)], axis=1)
        rows.append(snap)

        aligned_centroids: Dict[int, np.ndarray] = {}
        for c, centroid in local_centroids.items():
            tid = topic_map[c]
            aligned_centroids[tid] = centroid
            topic_rows.append({
                "snapshot_date": pd.Timestamp(snap_date),
                "model": model_cfg.get("short_name") or slug(model_cfg["name"]),
                "cluster_id": c,
                "topic_id": tid,
                "n_articles": int((labels == c).sum()),
                **{f"centroid_{j}": float(centroid[j]) for j in range(len(centroid))},
            })
        prev_topic_centroids = aligned_centroids

    snapshots = pd.concat(rows, ignore_index=True)
    topics = pd.DataFrame(topic_rows)
    write_table(snapshots, out_dir / "snapshots")
    write_table(topics, out_dir / "topics")

    results_cols = [id_col, date_col, "snapshot_date", "model", "cluster_id", "topic_id", "is_outlier", "outlier_score"] + dim_cols
    snapshots[results_cols].to_csv(out_dir / "results.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_root = Path(cfg["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    articles = clean_articles(read_table(cfg["input"]["articles"]), cfg)
    articles.to_csv(out_root / "articles_clean.csv", index=False)
    (out_root / "run_config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")

    for model_cfg in cfg["embedding"]["models"]:
        short = model_cfg.get("short_name") or slug(model_cfg["name"])
        model_dir = out_root / "models" / slug(short)
        model_dir.mkdir(parents=True, exist_ok=True)
        emb = load_or_compute_embeddings(articles, model_cfg, cfg, model_dir)
        run_model(articles, emb, model_cfg, cfg, model_dir)
        print(f"wrote {model_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
