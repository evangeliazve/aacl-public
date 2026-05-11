#!/usr/bin/env python3
"""Create publication-time feature tables used by the experiments.

Outputs:
  - feature_long_model_level.csv: one row per article and embedding model
  - feature_article_level.csv: one row per article after model aggregation

Only features used in the reported experiments are created.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml
from sklearn.neighbors import NearestNeighbors

GEOM_BASE = [
    "d1_nearest_centroid_pct", "d2_second_centroid_pct", "margin_d2_minus_d1_pct",
    "mahal_nearest_pct", "knn_mean_k20_pct", "knn_std_k20_pct",
    "outlier_proto_mean_dist_pct", "outlier_score", "has_recent_outliers", "n_recent_outliers",
]
TEXT_SOCIAL = [
    "soc_unique_users", "soc_median_user_public_metrics_followers_count",
    "soc_median_user_public_metrics_tweet_count", "soc_median_user_public_metrics_listed_count",
    "media_weighted_clustering", "media_bridge_ratio", "media_community_size",
    "text_subjectivity", "text_neutrality", "avg_sentence_len_words", "avg_word_len_chars",
    "total_syllables", "avg_syllables_per_word", "len_chars", "len_words",
    "ner_total_ents", "ner_distinct_ents", "ner_person", "ner_org", "ner_loc", "ner_misc",
]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_table(path):
    p = Path(path)
    if not p or str(p).lower() in {"", "none", "null"}:
        return None
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    return pd.read_csv(p)


def clean_text(x) -> str:
    return "" if pd.isna(x) else str(x)


def count_syllables_fr(word: str) -> int:
    word = re.sub(r"[^a-zàâäéèêëîïôöùûüÿçœ]", "", word.lower())
    if not word:
        return 0
    groups = re.findall(r"[aeiouyàâäéèêëîïôöùûüÿœ]+", word)
    return max(1, len(groups))


def text_features(articles: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    id_col = cfg["input"]["article_id_col"]
    title_col = cfg["input"].get("title_col", "title")
    text_col = cfg["input"].get("text_col", "description")
    spacy_model = cfg.get("features", {}).get("spacy_model", "fr_core_news_md")

    try:
        from textblob import TextBlob
        from textblob_fr import PatternTagger, PatternAnalyzer
        has_textblob = True
    except Exception:
        has_textblob = False
    try:
        from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer
        vader = SentimentIntensityAnalyzer()
    except Exception:
        vader = None
    try:
        import spacy
        nlp = spacy.load(spacy_model)
    except Exception:
        nlp = None

    rows = []
    for _, r in articles.iterrows():
        text = (clean_text(r.get(title_col, "")) + " " + clean_text(r.get(text_col, ""))).strip()
        words = re.findall(r"\w+", text, flags=re.UNICODE)
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        syllables = sum(count_syllables_fr(w) for w in words)
        if has_textblob and text:
            try:
                subj = float(TextBlob(text, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer()).sentiment[1])
            except Exception:
                subj = 0.0
        else:
            subj = 0.0
        compound = vader.polarity_scores(text)["compound"] if vader is not None and text else 0.0
        ents = []
        if nlp is not None and text:
            ents = [(e.text, e.label_) for e in nlp(text).ents]
        labels = [lab for _, lab in ents]
        rows.append({
            id_col: r[id_col],
            "text_subjectivity": subj,
            "text_neutrality": 1.0 - abs(float(compound)),
            "avg_sentence_len_words": len(words) / max(len(sentences), 1),
            "avg_word_len_chars": np.mean([len(w) for w in words]) if words else 0.0,
            "total_syllables": syllables,
            "avg_syllables_per_word": syllables / max(len(words), 1),
            "len_chars": len(text),
            "len_words": len(words),
            "ner_total_ents": len(ents),
            "ner_distinct_ents": len(set(t for t, _ in ents)),
            "ner_person": sum(lab in {"PER", "PERSON"} for lab in labels),
            "ner_org": sum(lab in {"ORG"} for lab in labels),
            "ner_loc": sum(lab in {"LOC", "GPE"} for lab in labels),
            "ner_misc": sum(lab not in {"PER", "PERSON", "ORG", "LOC", "GPE"} for lab in labels),
        })
    return pd.DataFrame(rows)


def social_features(shares: pd.DataFrame | None, articles: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    id_col = cfg["input"]["article_id_col"]
    date_col = cfg["input"]["date_col"]
    base = pd.DataFrame({id_col: articles[id_col].drop_duplicates()})
    for c in [
        "soc_unique_users", "soc_median_user_public_metrics_followers_count",
        "soc_median_user_public_metrics_tweet_count", "soc_median_user_public_metrics_listed_count",
        "media_weighted_clustering", "media_bridge_ratio", "media_community_size",
    ]:
        base[c] = 0.0
    if shares is None or shares.empty:
        return base

    import networkx as nx
    try:
        import community as community_louvain
    except Exception:
        community_louvain = None

    url_col = cfg["input"].get("url_col", id_col)
    user_col = "user_id" if "user_id" in shares.columns else "author_id"
    share_url_col = url_col if url_col in shares.columns else id_col
    share_date_col = "created_at" if "created_at" in shares.columns else date_col
    shares = shares.copy()
    shares[share_date_col] = pd.to_datetime(shares[share_date_col], errors="coerce", utc=True).dt.tz_convert(None)
    art_dates = articles[[id_col, date_col]].copy()
    art_dates[date_col] = pd.to_datetime(art_dates[date_col], errors="coerce")
    shares = shares.merge(art_dates, left_on=share_url_col, right_on=id_col, how="inner", suffixes=("", "_article"))
    shares = shares[shares[share_date_col].isna() | (shares[share_date_col] <= shares[date_col])]

    metrics = [
        ("followers_count", "soc_median_user_public_metrics_followers_count"),
        ("tweet_count", "soc_median_user_public_metrics_tweet_count"),
        ("listed_count", "soc_median_user_public_metrics_listed_count"),
    ]
    g = shares.groupby(id_col)
    out = g[user_col].nunique().rename("soc_unique_users").to_frame()
    for raw, name in metrics:
        col = raw if raw in shares.columns else f"user_public_metrics_{raw}"
        out[name] = g[col].median() if col in shares.columns else 0.0

    # Co-sharing graph.
    B = nx.Graph()
    for user, urls in shares.groupby(user_col)[id_col]:
        urls = list(pd.unique(urls.dropna()))
        for u in urls:
            B.add_node(u, bipartite="url")
        for i, u in enumerate(urls):
            for v in urls[i + 1:]:
                if B.has_edge(u, v):
                    B[u][v]["weight"] += 1
                else:
                    B.add_edge(u, v, weight=1)
    url_nodes = [n for n, d in B.nodes(data=True) if d.get("bipartite") == "url"]
    if url_nodes:
        clustering = nx.clustering(B.subgraph(url_nodes), weight="weight")
        if community_louvain is not None and len(url_nodes) > 1:
            part = community_louvain.best_partition(B.subgraph(url_nodes), weight="weight", random_state=42)
        else:
            part = {u: i for i, u in enumerate(url_nodes)}
        comm_size = pd.Series(part).value_counts().to_dict()
        graph_rows = []
        for u in url_nodes:
            total_w = sum(d.get("weight", 1) for _, _, d in B.edges(u, data=True))
            bridge_w = sum(d.get("weight", 1) for _, v, d in B.edges(u, data=True) if part.get(v) != part.get(u))
            graph_rows.append({
                id_col: u,
                "media_weighted_clustering": clustering.get(u, 0.0),
                "media_bridge_ratio": bridge_w / total_w if total_w else 0.0,
                "media_community_size": comm_size.get(part.get(u), 1),
            })
        out = out.join(pd.DataFrame(graph_rows).set_index(id_col), how="left")
    out = out.reset_index()
    base = base.drop(columns=[c for c in out.columns if c in base.columns and c != id_col]).merge(out, on=id_col, how="left")
    return base.fillna(0.0)


def publication_time_rows(results: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    id_col = cfg["input"]["article_id_col"]
    date_col = cfg["input"]["date_col"]
    df = results.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df = df[df["snapshot_date"] >= df[date_col]].sort_values([id_col, "snapshot_date"])
    return df.groupby(id_col).head(1).copy()


def model_geometric_features(results: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    id_col = cfg["input"]["article_id_col"]
    date_col = cfg["input"]["date_col"]
    knn_k = int(cfg.get("features", {}).get("knn_k", 20))
    proto_k = int(cfg.get("features", {}).get("recent_outlier_neighbors", 10))
    dim_cols = [c for c in results.columns if c.startswith("umap_")]
    ta = publication_time_rows(results, cfg)
    rows = []
    for _, row in ta.iterrows():
        tau = row["snapshot_date"]
        snap = results[results["snapshot_date"] == tau].copy()
        x = row[dim_cols].to_numpy(dtype=float)
        assigned = snap[pd.to_numeric(snap["topic_id"], errors="coerce").fillna(-1).astype(int) >= 0]
        centroids = assigned.groupby("topic_id")[dim_cols].mean()
        if len(centroids) >= 1:
            dists = np.linalg.norm(centroids.to_numpy(dtype=float) - x, axis=1)
            d1 = float(np.min(dists))
            d2 = float(np.partition(dists, 1)[1]) if len(dists) > 1 else d1
        else:
            d1 = d2 = np.nan
        mahal = np.nan
        if not assigned.empty:
            vals = []
            for _, grp in assigned.groupby("topic_id"):
                if len(grp) >= 5:
                    arr = grp[dim_cols].to_numpy(dtype=float)
                    mu = arr.mean(axis=0)
                    var = arr.var(axis=0) + 1e-9
                    vals.append(np.sum(((x - mu) ** 2) / var))
            mahal = float(np.min(vals)) if vals else np.nan
        all_x = snap[dim_cols].to_numpy(dtype=float)
        n_neighbors = min(knn_k + 1, len(all_x))
        if n_neighbors > 1:
            nn = NearestNeighbors(n_neighbors=n_neighbors).fit(all_x)
            d_nn = nn.kneighbors([x], return_distance=True)[0][0][1:]
            knn_mean = float(np.mean(d_nn)); knn_std = float(np.std(d_nn))
        else:
            knn_mean = knn_std = np.nan
        outliers = snap[snap["is_outlier"].astype(str).str.lower().isin(["true", "1"])]
        has_recent = int(len(outliers) > 0)
        n_recent = int(len(outliers))
        if len(outliers) > 0:
            od = np.linalg.norm(outliers[dim_cols].to_numpy(dtype=float) - x, axis=1)
            proto_dist = float(np.mean(np.sort(od)[:min(proto_k, len(od))]))
        else:
            proto_dist = np.nan
        rows.append({
            id_col: row[id_col],
            "model": row.get("model", "model"),
            "horizon": "TA",
            "d1_nearest_centroid": d1,
            "d2_second_centroid": d2,
            "margin_d2_minus_d1": d2 - d1 if pd.notna(d1) and pd.notna(d2) else np.nan,
            "mahal_nearest": mahal,
            "knn_mean_k20": knn_mean,
            "knn_std_k20": knn_std,
            "outlier_proto_mean_dist": proto_dist,
            "outlier_score": float(row.get("outlier_score", 0.0)),
            "has_recent_outliers": has_recent,
            "n_recent_outliers": n_recent,
        })
    out = pd.DataFrame(rows)
    for c in [c for c in out.columns if c not in {id_col, "model", "horizon", "has_recent_outliers", "n_recent_outliers", "outlier_score"}]:
        out[c + "_pct"] = out.groupby("model")[c].rank(pct=True, method="average")
    out["outlier_score"] = pd.to_numeric(out["outlier_score"], errors="coerce").fillna(0.0)
    return out


def aggregate_article_level(long_df: pd.DataFrame, id_col: str, single_features: List[str]) -> pd.DataFrame:
    geom_cols = [c for c in long_df.columns if c.endswith("_pct") or c in ["outlier_score", "has_recent_outliers", "n_recent_outliers"]]
    agg = long_df.groupby(id_col)[geom_cols].agg(["mean", "median", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg = agg.merge(long_df.groupby(id_col).size().rename("n_models_present").reset_index(), on=id_col, how="left")
    singles = long_df[[id_col] + single_features].drop_duplicates(id_col)
    return agg.merge(singles, on=id_col, how="left").fillna(0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(args.output_dir or cfg["output_dir"])
    id_col = cfg["input"]["article_id_col"]

    articles = read_table(cfg["input"]["articles"])
    articles[cfg["input"]["date_col"]] = pd.to_datetime(articles[cfg["input"]["date_col"]], errors="coerce", utc=True).dt.tz_convert(None)
    articles = articles.drop_duplicates(id_col)
    shares = read_table(cfg["input"].get("shares")) if cfg["input"].get("shares") else None

    txt = text_features(articles, cfg)
    soc = social_features(shares, articles, cfg)
    singles = txt.merge(soc, on=id_col, how="outer").fillna(0.0)

    frames = []
    for p in sorted((root / "models").glob("*/results.csv")):
        res = pd.read_csv(p)
        frames.append(model_geometric_features(res, cfg))
    if not frames:
        raise FileNotFoundError(f"No model results found in {root / 'models'}")
    long_df = pd.concat(frames, ignore_index=True).merge(singles, on=id_col, how="left").fillna(0.0)
    article_df = aggregate_article_level(long_df, id_col, TEXT_SOCIAL)

    long_df.to_csv(root / "feature_long_model_level.csv", index=False)
    article_df.to_csv(root / "feature_article_level.csv", index=False)
    print(f"wrote {root / 'feature_long_model_level.csv'}")
    print(f"wrote {root / 'feature_article_level.csv'}")


if __name__ == "__main__":
    main()
