#!/usr/bin/env python3
"""
Collect French climate-change news articles with the GNews Python library.

Example:
    python climatenewsfr_data_collection_gnews.py \
        --start-date 2025-04-02 \
        --end-date 2025-05-25 \
        --query "changement climatique" \
        --output articles_climate_cleaned.xlsx
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import pandas as pd
from gnews import GNews


DEFAULT_QUERY = "changement climatique"
DEFAULT_START_DATE = "2025-04-02"
DEFAULT_END_DATE = "2025-05-25"

ILLEGAL_CHARACTERS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def parse_date(value: str) -> date:
    """Parse a YYYY-MM-DD string."""
    return datetime.strptime(value, "%Y-%m-%d").date()


def daterange(start: date, end: date):
    """Yield each date in an inclusive date range."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def normalize_url(url: str) -> str:
    """
    Normalize URL for deduplication.

    Keeps the publisher URL but removes common tracking parameters and fragments.
    """
    if not url:
        return ""

    parsed = urlparse(url.strip())
    query_params = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if not (
            k.lower().startswith("utm_")
            or k.lower() in {"fbclid", "gclid", "mc_cid", "mc_eid"}
        )
    ]
    cleaned = parsed._replace(query=urlencode(query_params), fragment="")
    return urlunparse(cleaned)


def make_article_id(url: str, title: str) -> str:
    """Create a stable article id from normalized URL, falling back to title."""
    key = normalize_url(url) or title
    return hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()[:16]


def extract_domain(url: str) -> str:
    """Return the URL domain."""
    return urlparse(url).netloc.replace("www.", "").lower() if url else ""


def infer_country_from_url(url: str) -> str:
    """Infer a broad country label from URL top-level domain."""
    domain = extract_domain(url)
    tld = domain.split(".")[-1].lower() if domain else ""

    tld_country_map = {
        "fr": "France",
        "ca": "Canada",
        "be": "Belgium",
        "ch": "Switzerland",
        "ma": "Morocco",
        "dz": "Algeria",
        "tn": "Tunisia",
        "sn": "Senegal",
        "cm": "Cameroon",
        "ci": "Ivory Coast",
        "lu": "Luxembourg",
        "ht": "Haiti",
    }
    return tld_country_map.get(tld, "Unknown")


def clean_illegal_chars(value: Any) -> Any:
    """Remove characters that cannot be written to OpenXML spreadsheets."""
    if isinstance(value, str):
        value = ILLEGAL_CHARACTERS_RE.sub("", value)
        value = value.replace("\xa0", " ").strip()
    return value


def clean_text(value: Any) -> str:
    """Normalize text values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text)
    return clean_illegal_chars(text)


def first_paragraph(text: str) -> str:
    """Extract a first paragraph-like lead from full article text."""
    if not text:
        return ""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}|\r\n{2,}", text) if p.strip()]
    if paragraphs:
        return clean_text(paragraphs[0])

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return clean_text(" ".join(sentences[:2]))


def parse_gnews_published_date(value: Any) -> str:
    """Parse the published date returned by GNews, returning YYYY-MM-DD when possible."""
    if not value:
        return ""

    if isinstance(value, datetime):
        return value.date().isoformat()

    text = str(value).strip()
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(parsed):
        return text
    return parsed.date().isoformat()


def get_publisher_name(item: dict[str, Any]) -> str:
    """Extract publisher name from a GNews result."""
    publisher = item.get("publisher", "")
    if isinstance(publisher, dict):
        return clean_text(publisher.get("title") or publisher.get("name") or "")
    return clean_text(publisher)


def get_result_url(item: dict[str, Any]) -> str:
    """Extract URL from a GNews result."""
    return clean_text(item.get("url") or item.get("link") or "")


def collect_one_day(
    query: str,
    day: date,
    language: str,
    country: str,
    max_results: int,
    fetch_full_article: bool,
    request_delay: float,
) -> list[dict[str, Any]]:
    """
    Collect Google News results for a single day.

    GNews date windows are set per day to preserve the temporally ordered
    corpus structure used in the paper.
    """
    google_news = GNews(
        language=language,
        country=country,
        start_date=(day.year, day.month, day.day),
        end_date=(day.year, day.month, day.day),
        max_results=max_results,
    )

    results = google_news.get_news(query)
    rows: list[dict[str, Any]] = []

    for item in results:
        title = clean_text(item.get("title", ""))
        description = clean_text(item.get("description", ""))
        url = normalize_url(get_result_url(item))
        publisher = get_publisher_name(item)
        published_date = parse_gnews_published_date(item.get("published date") or item.get("published_date"))

        full_text = ""
        full_article_title = ""
        lead_paragraph = description

        if fetch_full_article and url:
            try:
                article = google_news.get_full_article(url)
                if article is not None:
                    full_article_title = clean_text(getattr(article, "title", "") or "")
                    full_text = clean_text(getattr(article, "text", "") or "")
                    lead_paragraph = first_paragraph(full_text) or description
                    if not title:
                        title = full_article_title
                time.sleep(request_delay)
            except Exception as exc:  # noqa: BLE001
                print(f"Could not fetch full article for {url}: {exc}")

        embedding_text = clean_text(f"{title}. {lead_paragraph}".strip(". "))

        rows.append(
            {
                "article_id": make_article_id(url, title),
                "query": query,
                "collection_day": day.isoformat(),
                "publication_date": published_date or day.isoformat(),
                "title": title,
                "lead_paragraph": lead_paragraph,
                "description": description,
                "embedding_text": embedding_text,
                "media_url": url,
                "source": publisher,
                "domain": extract_domain(url),
                "country_inferred_from_tld": infer_country_from_url(url),
                "body_text": full_text,
                "gnews_raw_title": clean_text(item.get("title", "")),
                "gnews_raw_published_date": clean_text(item.get("published date", "")),
            }
        )

    return rows


def collect_articles(
    start_date: str,
    end_date: str,
    query: str = DEFAULT_QUERY,
    language: str = "fr",
    country: str = "FR",
    max_results_per_day: int = 100,
    fetch_full_article: bool = True,
    request_delay: float = 1.0,
) -> pd.DataFrame:
    """Collect articles over an inclusive date range."""
    start = parse_date(start_date)
    end = parse_date(end_date)

    if end < start:
        raise ValueError("end_date must be on or after start_date")

    all_rows: list[dict[str, Any]] = []

    for day in daterange(start, end):
        print(f"Collecting Google News results for {day.isoformat()}...")
        try:
            rows = collect_one_day(
                query=query,
                day=day,
                language=language,
                country=country,
                max_results=max_results_per_day,
                fetch_full_article=fetch_full_article,
                request_delay=request_delay,
            )
            all_rows.extend(rows)
            print(f"  collected {len(rows)} raw results")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed on {day.isoformat()}: {exc}")

    df = pd.DataFrame(all_rows)

    if df.empty:
        return df

    for col in df.columns:
        df[col] = df[col].map(clean_illegal_chars)

    # Deduplicate after daily collection.
    # Prefer rows with a non-empty lead paragraph and body text.
    df["has_lead"] = df["lead_paragraph"].astype(str).str.len() > 0
    df["has_body"] = df["body_text"].astype(str).str.len() > 0
    df = (
        df.sort_values(["article_id", "has_lead", "has_body"], ascending=[True, False, False])
        .drop_duplicates(subset=["article_id"], keep="first")
        .drop(columns=["has_lead", "has_body"])
        .reset_index(drop=True)
    )

    # Keep a cleaned date column for compatibility with the previous script.
    df["publication_date_cleaned"] = pd.to_datetime(
        df["publication_date"], errors="coerce"
    ).dt.date.astype("string")

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect CLIMATENEWSFR-style Google News articles using GNews."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Inclusive start date, YYYY-MM-DD")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="Inclusive end date, YYYY-MM-DD")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Google News search query")
    parser.add_argument("--language", default="fr", help="GNews language code")
    parser.add_argument("--country", default="FR", help="GNews country code")
    parser.add_argument("--max-results-per-day", type=int, default=100, help="Maximum Google News results per day")
    parser.add_argument("--no-full-article", action="store_true", help="Do not fetch publisher pages with newspaper3k")
    parser.add_argument("--request-delay", type=float, default=1.0, help="Delay after full-article requests")
    parser.add_argument(
        "--output",
        default="articles_climate_cleaned.xlsx",
        help="Output path: .xlsx, .csv, or .jsonl",
    )
    return parser.parse_args()


def save_dataframe(df: pd.DataFrame, output: str) -> None:
    """Save output based on extension."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        df.to_excel(path, index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in {".jsonl", ".ndjson"}:
        df.to_json(path, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError("Output must end with .xlsx, .csv, .jsonl, or .ndjson")


def main() -> None:
    args = parse_args()

    df = collect_articles(
        start_date=args.start_date,
        end_date=args.end_date,
        query=args.query,
        language=args.language,
        country=args.country,
        max_results_per_day=args.max_results_per_day,
        fetch_full_article=not args.no_full_article,
        request_delay=args.request_delay,
    )

    save_dataframe(df, args.output)
    print(f"Saved {len(df)} unique articles to {args.output}")


