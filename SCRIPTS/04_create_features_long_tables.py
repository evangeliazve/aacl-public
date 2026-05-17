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
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

SOCIAL_ZERO_FEATURES = [
    "soc_unique_users", "soc_median_user_public_metrics_followers_count",
    "soc_median_user_public_metrics_tweet_count", "soc_median_user_public_metrics_listed_count",
    "media_weighted_clustering", "media_bridge_ratio", "media_community_size",
]


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_table(path: str | Path | None) -> Optional[pd.DataFrame]:
    if path is None or str(path).lower() in {"", "none", "null"}:
        return None
    p = Path(path)
    if not p.exists():
        return None
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    return pd.read_csv(p)


def to_dt_naive(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(None)


def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def clean_text(x: Any) -> str:
    return "" if pd.isna(x) else str(x)


def entropy(values: Iterable[Any]) -> float:
    vals = [str(v) for v in values if pd.notna(v)]
    if not vals:
        return 0.0
    counts = Counter(vals)
    n = float(sum(counts.values()))
    p = np.array(list(counts.values()), dtype=float) / n
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


# ---------------------------------------------------------------------------
# Text / NER features: reduced final feature set only
# ---------------------------------------------------------------------------

def approx_syllables_fr(word: str) -> int:
    w = str(word).lower()
    if not w:
        return 0
    vowels = "aeiouyàâäéèêëïîôöùûüÿœ"
    in_vowel = False
    count = 0
    for ch in w:
        if ch in vowels:
            if not in_vowel:
                count += 1
                in_vowel = True
        else:
            in_vowel = False
    return max(count, 1) if any(c.isalpha() for c in w) else 0


def text_features(articles: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    id_col = cfg["input"]["article_id_col"]
    title_col = cfg["input"].get("title_col", "title")
    text_col = cfg["input"].get("text_col", "description")
    spacy_model = cfg.get("features", {}).get("spacy_model", "fr_core_news_md")

    try:
        from textblob import Blobber, TextBlob
        from textblob_fr import PatternTagger, PatternAnalyzer
        tb_fr = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
        has_textblob_fr = True
        has_textblob = True
    except Exception:
        tb_fr = None
        has_textblob_fr = False
        try:
            from textblob import TextBlob  # type: ignore
            has_textblob = True
        except Exception:
            TextBlob = None  # type: ignore
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

    rows: List[Dict[str, Any]] = []
    for _, r in articles.iterrows():
        text = (clean_text(r.get(title_col, "")) + " " + clean_text(r.get(text_col, ""))).strip()
        words = re.findall(r"\w+", text, flags=re.UNICODE)
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        sentence_lengths = [len(re.findall(r"\w+", s, flags=re.UNICODE)) for s in sentences]
        syllables = [approx_syllables_fr(w) for w in words]

        subj = np.nan
        if text and has_textblob_fr and tb_fr is not None:
            try:
                subj = float(tb_fr(text).sentiment[1])
            except Exception:
                subj = np.nan
        elif text and has_textblob and TextBlob is not None:  # type: ignore[name-defined]
            try:
                subj = float(TextBlob(text).sentiment.subjectivity)  # type: ignore[union-attr]
            except Exception:
                subj = np.nan

        neutrality = np.nan
        if text and vader is not None:
            try:
                neutrality = float(1.0 - abs(float(vader.polarity_scores(text)["compound"])))
            except Exception:
                neutrality = np.nan

        ents = []
        if text and nlp is not None:
            try:
                ents = [(e.text, e.label_) for e in nlp(text).ents]
            except Exception:
                ents = []
        labels = [lab for _, lab in ents]

        rows.append({
            id_col: r[id_col],
            "text_subjectivity": subj,
            "text_neutrality": neutrality,
            "avg_sentence_len_words": float(np.mean(sentence_lengths)) if sentence_lengths else 0.0,
            "avg_word_len_chars": float(np.mean([len(w) for w in words])) if words else 0.0,
            "total_syllables": float(sum(syllables)) if syllables else 0.0,
            "avg_syllables_per_word": float(np.mean(syllables)) if syllables else 0.0,
            "len_chars": float(len(text)),
            "len_words": float(len(words)),
            "ner_total_ents": float(len(ents)),
            "ner_distinct_ents": float(len({t for t, _ in ents})),
            "ner_person": float(sum(lab in {"PER", "PERSON"} for lab in labels)),
            "ner_org": float(sum(lab in {"ORG"} for lab in labels)),
            "ner_loc": float(sum(lab in {"LOC", "GPE"} for lab in labels)),
            "ner_misc": float(sum(lab not in {"PER", "PERSON", "ORG", "LOC", "GPE"} for lab in labels)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Social / media graph features: final reduced set only, computed at article tau
# ---------------------------------------------------------------------------

def tau_edges_effective(tau: pd.Timestamp) -> pd.Timestamp:
    tau = pd.Timestamp(tau)
    if tau.hour == 0 and tau.minute == 0 and tau.second == 0 and tau.microsecond == 0 and tau.nanosecond == 0:
        return tau + pd.Timedelta(days=1)
    return tau


def prepare_shares(shares: Optional[pd.DataFrame], cfg: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if shares is None or shares.empty:
        return None, None
    df = shares.copy()
    id_col = cfg["input"]["article_id_col"]
    url_col = cfg["input"].get("url_col", id_col)
    share_url_col = first_existing(df, ["media_url", url_col, id_col, "article_url", "url"])
    if share_url_col is None:
        return None, None
    if share_url_col != id_col:
        df = df.rename(columns={share_url_col: id_col})
    time_col = first_existing(df, ["created_at", "publication_date_cleaned", cfg["input"].get("date_col", "")])
    if time_col is None:
        return None, None
    df[time_col] = to_dt_naive(df[time_col])
    if "author_id" not in df.columns and "user_id" in df.columns:
        df["author_id"] = df["user_id"]
    if "author_id" in df.columns:
        df["author_id"] = df["author_id"].astype(str)
    return df, time_col


def social_aggregate_for_media(hist: pd.DataFrame, media_id: Any, id_col: str) -> Dict[str, float]:
    out = {c: 0.0 for c in SOCIAL_ZERO_FEATURES if c.startswith("soc_")}
    if hist.empty or id_col not in hist.columns:
        return out
    g = hist[hist[id_col] == media_id]
    if g.empty:
        return out

    user_col = "author_id" if "author_id" in g.columns else ("user_id" if "user_id" in g.columns else None)
    out["soc_unique_users"] = float(g[user_col].nunique()) if user_col else 0.0
    for raw, name in [
        ("followers_count", "soc_median_user_public_metrics_followers_count"),
        ("tweet_count", "soc_median_user_public_metrics_tweet_count"),
        ("listed_count", "soc_median_user_public_metrics_listed_count"),
    ]:
        col = raw if raw in g.columns else f"user_public_metrics_{raw}"
        out[name] = float(pd.to_numeric(g[col], errors="coerce").median()) if col in g.columns else 0.0
    return out


def media_graph_features_at_tau(hist: pd.DataFrame, media_id: Any, id_col: str) -> Dict[str, float]:
    graph_defaults = {
        "media_weighted_clustering": 0.0,
        "media_bridge_ratio": 0.0,
        "media_community_size": 0.0,
    }
    if hist.empty or "author_id" not in hist.columns or id_col not in hist.columns:
        return graph_defaults
    try:
        import networkx as nx
        from networkx.algorithms import bipartite as nx_bip
    except Exception:
        return graph_defaults
    try:
        import community as community_louvain
    except Exception:
        community_louvain = None

    pairs = hist.dropna(subset=["author_id", id_col])[["author_id", id_col]].drop_duplicates()
    if pairs.empty:
        return graph_defaults

    B = nx.Graph()
    for _, r in pairs.iterrows():
        u = r["author_id"]
        m = r[id_col]
        B.add_node(u, bipartite="user")
        B.add_node(m, bipartite="media")
        B.add_edge(u, m)

    media_nodes = [n for n, d in B.nodes(data=True) if d.get("bipartite") == "media"]
    if not media_nodes:
        return graph_defaults
    Gm = nx_bip.weighted_projected_graph(B, media_nodes)
    if not Gm.has_node(media_id):
        Gm.add_node(media_id)

    clustering = nx.clustering(Gm, media_id, weight="weight") if Gm.degree(media_id) > 0 else 0.0
    partition = {}
    comm_size = 0.0
    bridge_ratio = 0.0
    if Gm.number_of_edges() > 0 and community_louvain is not None:
        try:
            partition = community_louvain.best_partition(Gm, weight="weight", random_state=42)
        except TypeError:
            partition = community_louvain.best_partition(Gm, weight="weight")
        except Exception:
            partition = {}
    if partition:
        c_id = partition.get(media_id, -1)
        comm_counts = Counter(partition.values())
        comm_size = float(comm_counts.get(c_id, 0.0))
        within = 0.0
        between = 0.0
        for nbr, data in Gm[media_id].items():
            w = float(data.get("weight", 1.0))
            if partition.get(nbr, -1) == c_id:
                within += w
            else:
                between += w
        bridge_ratio = float(between / (within + between + 1e-6))
    else:
        comm_size = 1.0 if Gm.has_node(media_id) else 0.0
        bridge_ratio = 0.0

    return {
        "media_weighted_clustering": float(clustering),
        "media_bridge_ratio": float(bridge_ratio),
        "media_community_size": float(comm_size),
    }


def social_features(shares: Optional[pd.DataFrame], articles: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    id_col = cfg["input"]["article_id_col"]
    date_col = cfg["input"]["date_col"]
    base = articles[[id_col, date_col]].drop_duplicates(id_col).copy()
    for c in SOCIAL_ZERO_FEATURES:
        base[c] = 0.0
    shares_prepared, share_time_col = prepare_shares(shares, cfg)
    if shares_prepared is None or share_time_col is None:
        return base.drop(columns=[date_col]).fillna(0.0)

    cache: Dict[pd.Timestamp, pd.DataFrame] = {}
    rows: List[Dict[str, Any]] = []
    for _, r in base[[id_col, date_col]].iterrows():
        media_id = r[id_col]
        tau = pd.Timestamp(r[date_col]) if pd.notna(r[date_col]) else pd.NaT
        if pd.isna(tau):
            rec = {id_col: media_id, **{c: 0.0 for c in SOCIAL_ZERO_FEATURES}}
            rows.append(rec)
            continue
        tau_eff = tau_edges_effective(tau)
        if tau_eff not in cache:
            cache[tau_eff] = shares_prepared[shares_prepared[share_time_col] < tau_eff].copy()
        hist = cache[tau_eff]
        rec = {id_col: media_id}
        rec.update(social_aggregate_for_media(hist, media_id, id_col))
        rec.update(media_graph_features_at_tau(hist, media_id, id_col))
        rows.append(rec)
    return pd.DataFrame(rows).fillna(0.0)


# ---------------------------------------------------------------------------
# Geometry features: notebook-compatible context for final reduced feature set
# ---------------------------------------------------------------------------

def normalized_results(results: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, str, str, List[str]]:
    df = results.copy()
    id_col = cfg["input"]["article_id_col"]
    date_col = cfg["input"]["date_col"]
    time_col = first_existing(df, ["snapshot_date", "time_window"])
    if time_col is None:
        raise ValueError("results.csv must contain snapshot_date or time_window")
    topic_col = first_existing(df, ["topic_id", "predicted_topic"])
    if topic_col is None:
        raise ValueError("results.csv must contain topic_id or predicted_topic")
    dim_cols = [c for c in df.columns if c.startswith("umap_dim_")]
    if not dim_cols:
        dim_cols = [c for c in df.columns if c.startswith("umap_")]
    if not dim_cols:
        raise ValueError("results.csv must contain reduced-dimension columns starting with umap_dim_ or umap_")

    df["__time__"] = to_dt_naive(df[time_col])
    df[date_col] = to_dt_naive(df[date_col])
    df["__topic__"] = pd.to_numeric(df[topic_col], errors="coerce").fillna(-1).astype(int)
    if "is_outlier" in df.columns:
        df["__is_outlier__"] = df["is_outlier"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        df["__is_outlier__"] = df["__topic__"].eq(-1)
    df["__model__"] = df.get("model", df.get("model_name", "model"))
    return df, id_col, date_col, dim_cols


def publication_time_rows(results: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, str, str, List[str]]:
    df, id_col, date_col, dim_cols = normalized_results(results, cfg)
    warmup_days = int(cfg.get("labeling", {}).get("warmup_days", 0))
    if warmup_days > 0 and df["__time__"].notna().any():
        cutoff = df["__time__"].min() + pd.Timedelta(days=warmup_days)
        df = df[df["__time__"] >= cutoff].copy()

    eligible = df[df["__time__"] >= df[date_col]].sort_values([id_col, "__time__"])
    ta = eligible.groupby(id_col).head(1).copy()
    return ta, df, id_col, date_col, dim_cols


def centroids(hist: pd.DataFrame, dim_cols: List[str]) -> Dict[int, np.ndarray]:
    assigned = hist[hist["__topic__"] != -1]
    out: Dict[int, np.ndarray] = {}
    if assigned.empty:
        return out
    for tid, grp in assigned.groupby("__topic__"):
        out[int(tid)] = grp[dim_cols].to_numpy(dtype=float).mean(axis=0)
    return out


def mahal_stats(hist: pd.DataFrame, dim_cols: List[str]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    assigned = hist[hist["__topic__"] != -1]
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    if assigned.empty:
        return out
    for tid, grp in assigned.groupby("__topic__"):
        if len(grp) >= 5:
            arr = grp[dim_cols].to_numpy(dtype=float)
            out[int(tid)] = (arr.mean(axis=0), arr.var(axis=0) + 1e-8)
    return out


def dist_to_centroids(x: np.ndarray, cents: Dict[int, np.ndarray]) -> np.ndarray:
    if not cents:
        return np.array([])
    M = np.vstack([cents[k] for k in cents.keys()])
    return np.sort(np.linalg.norm(M - x[None, :], axis=1))


def mahal_nearest(x: np.ndarray, stats: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> float:
    if not stats:
        return np.nan
    vals = []
    for mu, var in stats.values():
        z = (x - mu) / np.sqrt(var)
        vals.append(float((z ** 2).sum()))
    return float(np.min(vals)) if vals else np.nan


def knn_density(X_hist: np.ndarray, x: np.ndarray, k: int) -> Tuple[float, float]:
    if len(X_hist) == 0:
        return np.nan, np.nan
    k_eff = min(k, len(X_hist))
    nbrs = NearestNeighbors(n_neighbors=k_eff, metric="euclidean").fit(X_hist)
    dists, _ = nbrs.kneighbors(x.reshape(1, -1), n_neighbors=k_eff, return_distance=True)
    d = dists.flatten()
    return float(np.mean(d)), float(np.std(d))


def model_geometric_features(results: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    knn_k = int(cfg.get("features", {}).get("knn_k", 20))
    proto_k = int(cfg.get("features", {}).get("recent_outlier_neighbors", 10))
    outlier_lookback_days = int(cfg.get("features", {}).get("outlier_lookback_days", 1000))

    ta, df, id_col, _date_col, dim_cols = publication_time_rows(results, cfg)
    rows: List[Dict[str, Any]] = []
    for _, row in ta.iterrows():
        tau = pd.Timestamp(row["__time__"])
        hist = df[df["__time__"] <= tau].copy().sort_values("__time__")
        x = row[dim_cols].to_numpy(dtype=float)

        d_sorted = dist_to_centroids(x, centroids(hist, dim_cols))
        d1 = float(d_sorted[0]) if d_sorted.size >= 1 else np.nan
        d2 = float(d_sorted[1]) if d_sorted.size >= 2 else np.nan
        margin = d2 - d1 if np.isfinite(d1) and np.isfinite(d2) else np.nan
        mahal = mahal_nearest(x, mahal_stats(hist, dim_cols))
        knn_mean, knn_std = knn_density(hist[dim_cols].to_numpy(dtype=float), x, knn_k)

        recent_cut = tau - pd.Timedelta(days=outlier_lookback_days)
        
        out_recent = hist[
            hist["__is_outlier__"]
            & (hist["__time__"] >= recent_cut)
        ].copy()
        
        # Count each recent outlier article only once.
        out_recent = (
            out_recent
            .sort_values("__time__")
            .drop_duplicates(id_col, keep="last")
        )
        
        has_recent = float(len(out_recent) > 0)
        n_recent = float(len(out_recent))

        #recent_cut = tau - pd.Timedelta(days=outlier_lookback_days)
        #out_recent = hist[hist["__is_outlier__"] & (hist["__time__"] >= recent_cut)]
        #has_recent = float(len(out_recent) > 0)
        #n_recent = float(len(out_recent))
      
        if len(out_recent) > 0:
            X_out = out_recent[dim_cols].to_numpy(dtype=float)
            nbrs = NearestNeighbors(n_neighbors=min(proto_k, len(X_out)), metric="euclidean").fit(X_out)
            dists, _ = nbrs.kneighbors(x.reshape(1, -1), n_neighbors=min(proto_k, len(X_out)), return_distance=True)
            proto_dist = float(np.mean(dists.flatten()))
        else:
            proto_dist = np.nan

        rows.append({
            id_col: row[id_col],
            "model": row.get("model", row.get("model_name", row.get("__model__", "model"))),
            "model_name": row.get("model_name", row.get("model", row.get("__model__", "model"))),
            "horizon": "TA",
            "d1_nearest_centroid": d1,
            "d2_second_centroid": d2,
            "margin_d2_minus_d1": margin,
            "mahal_nearest": mahal,
            "knn_mean_k20": knn_mean,
            "knn_std_k20": knn_std,
            "outlier_proto_mean_dist": proto_dist,
            "outlier_score": float(row.get("outlier_score", np.nan)) if "outlier_score" in row.index else np.nan,
            "has_recent_outliers": has_recent,
            "n_recent_outliers": n_recent,
        })

    out = pd.DataFrame(rows)
    raw_geom = [
        "d1_nearest_centroid", "d2_second_centroid", "margin_d2_minus_d1",
        "mahal_nearest", "knn_mean_k20", "knn_std_k20", "outlier_proto_mean_dist",
    ]
    model_col = "model" if "model" in out.columns else "model_name"
    for c in raw_geom:
        out[c + "_pct"] = out.groupby(model_col)[c].rank(pct=True, method="average")
    out["outlier_score"] = pd.to_numeric(out["outlier_score"], errors="coerce").fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# Article-level aggregation
# ---------------------------------------------------------------------------

def aggregate_article_level(long_df: pd.DataFrame, id_col: str, single_features: List[str]) -> pd.DataFrame:
    geom_cols = [c for c in GEOM_BASE if c in long_df.columns]
    agg = long_df.groupby(id_col)[geom_cols].agg(["mean", "median", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg = agg.merge(long_df.groupby(id_col).size().rename("n_models_present").reset_index(), on=id_col, how="left")
    singles = long_df[[id_col] + [c for c in single_features if c in long_df.columns]].drop_duplicates(id_col)
    return agg.merge(singles, on=id_col, how="left").fillna(0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(args.output_dir or cfg["output_dir"])
    id_col = cfg["input"]["article_id_col"]
    date_col = cfg["input"]["date_col"]

    articles = read_table(cfg["input"]["articles"])
    if articles is None:
        raise FileNotFoundError(cfg["input"]["articles"])
    articles = articles.copy()
    articles[date_col] = to_dt_naive(articles[date_col])
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
