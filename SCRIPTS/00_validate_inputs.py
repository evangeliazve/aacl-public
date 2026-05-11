#!/usr/bin/env python3
"""Validate input tables before running the full reproduction pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'missing input file: {p}')
    if p.suffix.lower() in {'.xlsx', '.xls'}:
        return pd.read_excel(p)
    return pd.read_csv(p)


def require_columns(df: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    missing = [c for c in columns if c and c not in df.columns]
    if missing:
        raise ValueError(f'{table_name} is missing required columns: {missing}')


def check_dates(df: pd.DataFrame, date_col: str) -> None:
    dates = pd.to_datetime(df[date_col], errors='coerce')
    if dates.isna().any():
        n_bad = int(dates.isna().sum())
        raise ValueError(f'{date_col} contains {n_bad} non-parseable dates')
    if dates.nunique() < 2:
        print(f'warning: {date_col} contains fewer than two distinct dates')


def validate_articles(cfg: dict) -> pd.DataFrame:
    inp = cfg.get('input', {})
    articles = read_table(inp['articles'])
    required = [
        inp.get('article_id_col'),
        inp.get('date_col'),
        inp.get('title_col'),
        inp.get('text_col'),
        inp.get('url_col'),
    ]
    require_columns(articles, required, 'article table')
    check_dates(articles, inp['date_col'])

    article_id_col = inp.get('article_id_col')
    if articles[article_id_col].isna().any():
        raise ValueError(f'{article_id_col} contains missing article identifiers')
    if articles[article_id_col].duplicated().any():
        n_dup = int(articles[article_id_col].duplicated().sum())
        print(f'warning: {n_dup} duplicate article identifiers detected; downstream scripts keep the first occurrence')
    return articles


def validate_shares(cfg: dict, articles: pd.DataFrame) -> None:
    inp = cfg.get('input', {})
    share_path = inp.get('shares')
    if not share_path:
        print('social-sharing file not configured; social features will be zero-filled')
        return
    p = Path(share_path)
    if not p.exists():
        print(f'social-sharing file not found: {p}; social features will be zero-filled')
        return
    shares = read_table(p)
    url_col = inp.get('url_col') or inp.get('article_id_col')
    candidates = [url_col, 'media_url', 'url', 'article_url']
    share_url_col = next((c for c in candidates if c in shares.columns), None)
    if share_url_col is None:
        raise ValueError(f'share table must contain one URL/article column among: {candidates}')
    overlap = set(articles[inp.get('article_id_col')].astype(str)).intersection(set(shares[share_url_col].astype(str)))
    if not overlap:
        print('warning: no article identifiers overlap between article and social-sharing tables')


def validate_embeddings(cfg: dict, articles: pd.DataFrame) -> None:
    emb = cfg.get('embedding', {})
    precomputed_dir = Path(emb.get('precomputed_dir', 'data/embeddings'))
    models = emb.get('models', [])
    if not models:
        raise ValueError('embedding.models is empty')
    for model in models:
        short = model.get('short_name') or model.get('name')
        if model.get('type') == 'precomputed':
            npy = precomputed_dir / f'{short}.npy'
            csv = precomputed_dir / f'{short}.csv'
            if not npy.exists() and not csv.exists():
                raise FileNotFoundError(f'precomputed embeddings missing for {short}: expected {npy} or {csv}')
    print(f'configured embedding models: {len(models)}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    articles = validate_articles(cfg)
    validate_shares(cfg, articles)
    validate_embeddings(cfg, articles)
    print(f'validated {len(articles)} articles')


if __name__ == '__main__':
    main()
