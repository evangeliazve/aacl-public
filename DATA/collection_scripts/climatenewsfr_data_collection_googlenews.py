"""Collect French climate-change news articles from Google News RSS.

This script documents the collection procedure used for the CLIMATENEWSFR
corpus. It is intentionally parameterized and does not execute at import time.

Example:
    python DATA/collection_scripts/climatenewsfr_data_collection_googlenews.py \
        --start-date 2025-04-02 \
        --end-date 2025-05-25 \
        --output DATA/private/climatenewsfr/articles_climate_cleaned.xlsx
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup
from googlenewsdecoder import gnewsdecoder


DEFAULT_QUERY = "changement climatique"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
}
ILLEGAL_CHARACTERS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def extract_country_from_url(url: str) -> str:
    """Infer a broad country label from the URL top-level domain."""
    domain = urlparse(url).netloc
    country_code = domain.split(".")[-1].lower()
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
    return tld_country_map.get(country_code, "Unknown")


def decode_google_news_url(google_news_url: str, interval: float = 1.0, proxy: str | None = None) -> str:
    """Decode a Google News RSS redirect URL to the publisher URL when possible."""
    try:
        result = gnewsdecoder(google_news_url, interval=interval, proxy=proxy)
        if result.get("status"):
            return result["decoded_url"]
        print(f"Decoding failed: {result.get('message', 'unknown error')}")
        return google_news_url
    except Exception as exc:  # noqa: BLE001 - collection script should continue on noisy publisher failures
        print(f"Error decoding URL: {exc}")
        return google_news_url


def extract_article_content(
    url: str,
    request_delay: float = 2.0,
    proxy: str | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[str, str, str, str]:
    """Fetch a publisher page and extract basic article fields."""
    try:
        proxies = {"http": proxy, "https": proxy} if proxy else None
        response = requests.get(url, headers=headers or DEFAULT_HEADERS, timeout=10, proxies=proxies)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        title_tag = soup.find("h1") or soup.find("h2")
        title = title_tag.get_text(strip=True) if title_tag else ""

        paragraphs = soup.find_all("p")
        body_text = "\n".join(p.get_text(strip=True) for p in paragraphs)

        date = ""
        meta_date = (
            soup.find("meta", {"name": "pubdate"})
            or soup.find("meta", {"property": "article:published_time"})
            or soup.find("meta", {"name": "date"})
        )
        if meta_date and meta_date.has_attr("content"):
            date = meta_date["content"]

        description = ""
        meta_description = soup.find("meta", {"name": "description"})
        if meta_description and meta_description.has_attr("content"):
            description = meta_description["content"]

        time.sleep(request_delay)
        return title, body_text, date, description
    except Exception as exc:  # noqa: BLE001 - collection script should continue on noisy publisher failures
        print(f"Failed to extract content from {url}: {exc}")
        return "", "", "", ""


def clean_date_value(value: Any) -> Any:
    """Parse heterogeneous publisher date strings into dates where possible."""
    if pd.isna(value):
        return "Invalid"

    text = str(value).strip()

    if re.fullmatch(r"\d{2}/\d{2}/\d{4}( \d{2}:\d{2}:\d{2})?", text):
        for fmt in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y"):
            parsed = pd.to_datetime(text, format=fmt, errors="coerce")
            if not pd.isna(parsed):
                return parsed.date()
        return "Invalid"

    if re.fullmatch(r"\d{10}", text):
        try:
            return datetime.utcfromtimestamp(int(text)).date()
        except ValueError:
            return "Invalid"

    text = re.split(r"C", text)[0]

    if re.fullmatch(r"\d{8}", text):
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
        return parsed.date() if not pd.isna(parsed) else "Invalid"

    parsed = pd.to_datetime(text, errors="coerce")
    return parsed.date() if not pd.isna(parsed) else "Invalid"


def clean_illegal_chars(value: Any) -> Any:
    """Remove characters that cannot be written to OpenXML spreadsheets."""
    if isinstance(value, str):
        value = ILLEGAL_CHARACTERS_RE.sub("", value)
        value = value.replace("\xa0", " ")
    return value


def clean_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dates, URLs, and Excel-incompatible characters."""
    if df.empty:
        return df

    df = df.copy()
    df["publication_date_cleaned"] = df["publication_date"].apply(clean_date_value)
    df["media_url"] = df["media_url"].astype(str).str.replace(r"\?.*$", "", regex=True).str.strip()
    return df.map(clean_illegal_chars) if hasattr(df, "map") else df.applymap(clean_illegal_chars)


def collect_articles(
    start_date: str | datetime,
    end_date: str | datetime,
    query: str = DEFAULT_QUERY,
    proxy: str | None = None,
    interval_time: float = 1.0,
    request_delay: float = 2.0,
) -> pd.DataFrame:
    """Collect articles from Google News RSS for an inclusive date range."""
    if isinstance(start_date, str):
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    else:
        start_dt = start_date

    if isinstance(end_date, str):
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    else:
        end_dt = end_date

    if end_dt < start_dt:
        raise ValueError("end_date must be on or after start_date")

    encoded_query = "+".join(query.split())
    articles: list[dict[str, Any]] = []
    current_date = start_dt

    while current_date <= end_dt:
        after = current_date.strftime("%Y-%m-%d")
        before = (current_date + timedelta(days=1)).strftime("%Y-%m-%d")
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}+after:{after}+before:{before}&hl=fr"

        print(f"Fetching articles for: {after}")
        feed = feedparser.parse(rss_url)

        for entry in feed.entries:
            try:
                pub_date = datetime(*entry.published_parsed[:6]) if hasattr(entry, "published_parsed") else current_date
                google_news_url = entry.link
                final_url = decode_google_news_url(google_news_url, interval=interval_time, proxy=proxy)
                country = extract_country_from_url(final_url)
                title, body_text, date_meta, description = extract_article_content(
                    final_url,
                    request_delay=request_delay,
                    proxy=proxy,
                    headers=DEFAULT_HEADERS,
                )

                articles.append(
                    {
                        "media_url": final_url,
                        "title": title or getattr(entry, "title", ""),
                        "body_text": body_text,
                        "publication_date": date_meta or pub_date.strftime("%Y-%m-%d"),
                        "description": description or getattr(entry, "description", ""),
                        "country": country,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - collection script should continue across feeds
                print(f"Error processing article: {exc}")
                continue

        current_date += timedelta(days=1)

    return clean_articles(pd.DataFrame(articles))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect CLIMATENEWSFR Google News articles.")
    parser.add_argument("--start-date", required=True, help="Inclusive start date, YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Inclusive end date, YYYY-MM-DD")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Google News search query")
    parser.add_argument("--output", default=None, help="Output .xlsx path. Defaults to articles_climate_cleaned_<date>.xlsx")
    parser.add_argument("--proxy", default=None, help="Optional HTTP(S) proxy URL")
    parser.add_argument("--interval-time", type=float, default=1.0, help="Delay used by googlenewsdecoder")
    parser.add_argument("--request-delay", type=float, default=2.0, help="Delay after publisher-page requests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df_articles = collect_articles(
        start_date=args.start_date,
        end_date=args.end_date,
        query=args.query,
        proxy=args.proxy,
        interval_time=args.interval_time,
        request_delay=args.request_delay,
    )

    output_path = Path(args.output) if args.output else Path(f"articles_climate_cleaned_{datetime.now():%Y-%m-%d}.xlsx")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_articles.to_excel(output_path, index=False)
    print(f"Saved {len(df_articles)} articles to {output_path}")


if __name__ == "__main__":
    main()
