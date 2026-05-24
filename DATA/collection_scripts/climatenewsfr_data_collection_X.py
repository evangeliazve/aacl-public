"""Collect French climate-change URL-sharing activity from the X API.

This script documents the collection procedure used for the CLIMATENEWSFR
corpus. It is intentionally parameterized and does not execute at import time.

Example:
    python DATA/collection_scripts/climatenewsfr_data_collection_X.py \
        --start-time 2025-04-02T00:00:00Z \
        --end-time 2025-05-25T23:59:59Z \
        --bearer-token "$X_BEARER_TOKEN" \
        --iterations 50 \
        --output-dir DATA/private/climatenewsfr/x
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


X_SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"
DEFAULT_QUERY = "changement climatique has:links lang:fr"


def fetch_tweets_from_api(
    start_time: str | None,
    end_time: str | None,
    iterations: int = 1,
    bearer_token: str | None = None,
    query: str = DEFAULT_QUERY,
    pause_seconds: float = 1.0,
) -> list[dict[str, Any]]:
    """Fetch tweets from the X API recent-search endpoint.

    Parameters
    ----------
    start_time, end_time:
        RFC3339 timestamps, for example ``2025-04-02T00:00:00Z``.
    iterations:
        Maximum number of paginated API requests.
    bearer_token:
        X API bearer token. The token is required for real collection.
    query:
        X API query string.
    pause_seconds:
        Delay between paginated requests.
    """
    if not bearer_token:
        raise ValueError("A bearer token is required. Pass --bearer-token or set X_BEARER_TOKEN.")

    headers = {"Authorization": f"Bearer {bearer_token}"}
    params: dict[str, str] = {
        "max_results": "100",
        "query": query,
        "tweet.fields": ",".join(
            [
                "attachments",
                "author_id",
                "context_annotations",
                "conversation_id",
                "created_at",
                "edit_controls",
                "edit_history_tweet_ids",
                "entities",
                "geo",
                "id",
                "in_reply_to_user_id",
                "lang",
                "possibly_sensitive",
                "public_metrics",
                "referenced_tweets",
                "reply_settings",
                "source",
                "text",
                "withheld",
            ]
        ),
        "expansions": ",".join(
            [
                "attachments.media_keys",
                "author_id",
                "geo.place_id",
                "referenced_tweets.id",
                "referenced_tweets.id.author_id",
            ]
        ),
        "media.fields": ",".join(
            [
                "alt_text",
                "duration_ms",
                "height",
                "media_key",
                "preview_image_url",
                "public_metrics",
                "type",
                "url",
                "variants",
                "width",
            ]
        ),
        "user.fields": ",".join(
            [
                "created_at",
                "description",
                "id",
                "location",
                "name",
                "public_metrics",
                "url",
                "username",
                "verified",
                "verified_type",
                "withheld",
            ]
        ),
        "place.fields": ",".join(
            [
                "contained_within",
                "country",
                "country_code",
                "full_name",
                "geo",
                "id",
                "name",
                "place_type",
            ]
        ),
    }

    if start_time:
        params["start_time"] = start_time
    if end_time:
        params["end_time"] = end_time

    all_tweets: list[dict[str, Any]] = []
    next_token: str | None = None

    for page_idx in range(iterations):
        request_params = dict(params)
        if next_token:
            request_params["next_token"] = next_token

        response = requests.get(X_SEARCH_URL, headers=headers, params=request_params, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"X API error {response.status_code}: {response.text}")

        payload = response.json()
        tweets = payload.get("data", [])
        includes = payload.get("includes", {})

        users = {user["id"]: user for user in includes.get("users", [])}
        media = {item["media_key"]: item for item in includes.get("media", [])}
        places = {place["id"]: place for place in includes.get("places", [])}
        referenced_tweets = {tweet["id"]: tweet for tweet in includes.get("tweets", [])}

        for tweet in tweets:
            tweet["user"] = users.get(tweet.get("author_id"))

            media_keys = tweet.get("attachments", {}).get("media_keys", [])
            if media_keys:
                tweet["media"] = [media[key] for key in media_keys if key in media]

            place_id = tweet.get("geo", {}).get("place_id")
            if place_id:
                tweet["place"] = places.get(place_id)

            if "referenced_tweets" in tweet:
                tweet["referenced_full"] = []
                for ref in tweet["referenced_tweets"]:
                    ref_data = referenced_tweets.get(ref.get("id"))
                    if ref_data:
                        ref_data["user"] = users.get(ref_data.get("author_id"))
                        tweet["referenced_full"].append(ref_data)

            all_tweets.append(tweet)

        print(f"Page {page_idx + 1}: fetched {len(tweets)} tweets")

        next_token = payload.get("meta", {}).get("next_token")
        if not next_token:
            break
        time.sleep(pause_seconds)

    return all_tweets


def save_tweets(tweets: list[dict[str, Any]], output_dir: str | Path = ".") -> tuple[Path, Path]:
    """Save collected tweets as JSON and flattened Excel files."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"tweets_climate_{timestamp}.json"
    xlsx_path = out_dir / f"tweets_climate_{timestamp}.xlsx"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(tweets, file, ensure_ascii=False, indent=2)

    pd.json_normalize(tweets, sep="_").to_excel(xlsx_path, index=False)
    return json_path, xlsx_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect CLIMATENEWSFR X-sharing activity.")
    parser.add_argument("--start-time", required=True, help="RFC3339 start timestamp, e.g. 2025-04-02T00:00:00Z")
    parser.add_argument("--end-time", required=True, help="RFC3339 end timestamp, e.g. 2025-05-25T23:59:59Z")
    parser.add_argument("--bearer-token", default=None, help="X API bearer token. Defaults to X_BEARER_TOKEN env var if omitted.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="X API query string.")
    parser.add_argument("--iterations", type=int, default=50, help="Maximum number of API pages to fetch.")
    parser.add_argument("--pause-seconds", type=float, default=1.0, help="Delay between paginated requests.")
    parser.add_argument("--output-dir", default=".", help="Directory for JSON and Excel outputs.")
    return parser.parse_args()


def main() -> None:
    import os

    args = parse_args()
    token = args.bearer_token or os.environ.get("X_BEARER_TOKEN")
    tweets = fetch_tweets_from_api(
        start_time=args.start_time,
        end_time=args.end_time,
        iterations=args.iterations,
        bearer_token=token,
        query=args.query,
        pause_seconds=args.pause_seconds,
    )
    json_path, xlsx_path = save_tweets(tweets, args.output_dir)
    print(f"Saved {len(tweets)} tweets to {json_path} and {xlsx_path}")


if __name__ == "__main__":
    main()
