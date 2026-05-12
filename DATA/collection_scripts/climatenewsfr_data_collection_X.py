# Load Libraries
import requests
import json
from datetime import datetime
import time
import pandas as pd


def fetch_tweets(start_time, end_time, iterations=1, bearer_token=None, lang="fr"):
    url = "https://api.twitter.com/2/tweets/search/recent"

    query = 'changement climatique has:links lang:fr'

    headers = {
        "Authorization": f"Bearer {bearer_token or 'YOUR_DEFAULT_BEARER_TOKEN'}"
    }

    querystring = {
        "max_results": "100",
        "query": query,
        "tweet.fields": ",".join([
            "article",
            "attachments",
            "author_id",
            "card_uri",
            "community_id",
            "context_annotations",
            "conversation_id",
            "created_at",
            "display_text_range",
            "edit_controls",
            "edit_history_tweet_ids",
            "entities",
            "geo",
            "id",
            "in_reply_to_user_id",
            "lang",
            "media_metadata",
            "note_tweet",
            "possibly_sensitive",
            "public_metrics",
            "referenced_tweets",
            "reply_settings",
            "scopes",
            "source",
            "text",
            "withheld",
        ]),
        "expansions": ",".join([
            "article.cover_media",
            "article.media_entities",
            "attachments.media_keys",
            "attachments.media_source_tweet",
            "attachments.poll_ids",
            "author_id",
            "edit_history_tweet_ids",
            "entities.mentions.username",
            "geo.place_id",
            "in_reply_to_user_id",
            "entities.note.mentions.username",
            "referenced_tweets.id",
            "referenced_tweets.id.attachments.media_keys",
            "referenced_tweets.id.author_id",
        ]),
        "media.fields": ",".join([
            "alt_text",
            "duration_ms",
            "height",
            "media_key",
            "non_public_metrics",
            "organic_metrics",
            "preview_image_url",
            "promoted_metrics",
            "public_metrics",
            "type",
            "url",
            "variants",
            "width",
        ]),
        "user.fields": ",".join([
            #"affiliation",
            #"confirmed_email",
            #"connection_status",
            "created_at",
            "description",
            #"entities",
            "id",
            "is_identity_verified",
            "location",
            #"most_recent_tweet_id",
            #"name",
            #"parody",
            #"pinned_tweet_id",
            #"profile_banner_url",
            #"profile_image_url",
            #"protected",
            "public_metrics",
            #"receives_your_dm",
            #"subscription",
            #"subscription_type",
            #"url",
            #"username",
            #"verified",
            "verified_followers_count",
            #"verified_type",
            #"withheld",
        ]),
        "place.fields": ",".join([
            "contained_within",
            "country",
            "country_code",
            "full_name",
            "geo",
            "id",
            "name",
            "place_type",
        ]),
    }

    if start_time:
        querystring["start_time"] = start_time
    if end_time:
        querystring["end_time"] = end_time

    all_tweets = []
    next_token = None

    for i in range(iterations):
        if next_token:
            querystring["next_token"] = next_token

        response = requests.get(url, headers=headers, params=querystring)

        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            break

        data = response.json()
        tweets = data.get("data", [])
        includes = data.get("includes", {})

        # Build lookup dictionaries
        users = {u["id"]: u for u in includes.get("users", [])}
        media = {m["media_key"]: m for m in includes.get("media", [])}
        places = {p["id"]: p for p in includes.get("places", [])}
        polls = {p["id"]: p for p in includes.get("polls", [])}
        ref_tweets = {t["id"]: t for t in includes.get("tweets", [])}

        for tweet in tweets:
            # Add user info
            tweet["user"] = users.get(tweet.get("author_id"))

            # Add media info
            if "attachments" in tweet and "media_keys" in tweet["attachments"]:
                tweet["media"] = [
                    media.get(k) for k in tweet["attachments"]["media_keys"] if k in media
                ]

            # Add place info
            if "geo" in tweet and "place_id" in tweet["geo"]:
                tweet["place"] = places.get(tweet["geo"]["place_id"])

            # Add poll info
            if "attachments" in tweet and "poll_ids" in tweet["attachments"]:
                tweet["polls"] = [
                    polls.get(pid) for pid in tweet["attachments"]["poll_ids"] if pid in polls
                ]

            # Add full referenced tweets
            if "referenced_tweets" in tweet:
                tweet["referenced_full"] = []
                for ref in tweet["referenced_tweets"]:
                    ref_data = ref_tweets.get(ref["id"])
                    if ref_data:
                        ref_data["user"] = users.get(ref_data.get("author_id"))
                        tweet["referenced_full"].append(ref_data)

            all_tweets.append(tweet)

        print(f"Iteration {i+1}: {len(tweets)} tweets fetched")

        next_token = data.get("meta", {}).get("next_token")
        if not next_token:
            break  # No more pages
        time.sleep(1)  # Respect rate limits

    return all_tweets


def fetch_tweets(start_time, end_time, iterations=1, bearer_token=None):
    all_tweets = fetch_tweets(start_time, end_time, iterations, bearer_token, lang="fr")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"/your_path/tweets_hydrogen_{timestamp}.json"
    xls_filename = f"/your_path/tweets_hydrogen_{timestamp}.xlsx"

    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(all_tweets, f, ensure_ascii=False, indent=2)

    df = pd.json_normalize(all_tweets, sep="_")
    df.to_excel(xls_filename)

    print(f"Saved {len(all_tweets)} tweets")

    return all_tweets


# Get Results
start="SET_YOUR_START_DATE_XXX-XX-XXTXX:XX:00Z"
end="SET_YOUR_END_DATE_XXX-XX-XXTXX:XX:00Z"
token="SET_YOUR_TOKEN"

fetch_tweets(start_time=start, end_time=end, iterations=50, bearer_token=token)
