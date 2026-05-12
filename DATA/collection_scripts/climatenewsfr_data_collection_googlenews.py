import feedparser
import pandas as pd
import requests
import re
import time
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from urllib.parse import urlparse
from googlenewsdecoder import gnewsdecoder

start_date = "SET_YOUR_START_DATE" #e.g. datetime(2025, 3, 20)
end_date = "SET_YOUR_START_DATE" #e.g. datetime(2025, 6, 8)

base_url = (
    f"https://news.google.com/rss/search?"
    f"q=changement+climatique+after:{after}+before:{before}&hl=fr"
)
articles = []
proxy = None  # Example: "http://user:pass@localhost:8080"
interval_time = 1  # Delay between decoding requests
request_delay = 2  # Delay between HTTP requests to avoid 429 errors

# HTTP Headers with User-Agent to mimic a browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

def extract_country_from_url(url):
    domain = urlparse(url).netloc
    country_code = domain.split('.')[-1]
    tld_country_map = {
        'fr': 'France', 'ca': 'Canada', 'be': 'Belgium', 'ch': 'Switzerland',
        'ma': 'Morocco', 'dz': 'Algeria', 'tn': 'Tunisia', 'sn': 'Senegal',
        'cm': 'Cameroon', 'ci': 'Ivory Coast', 'lu': 'Luxembourg', 'ht': 'Haiti'
    }
    return tld_country_map.get(country_code, 'Unknown')

def decode_google_news_url(google_news_url):
    try:
        result = gnewsdecoder(google_news_url, interval=interval_time, proxy=proxy)
        if result.get("status"):
            return result["decoded_url"]
        else:
            print(f"Decoding failed: {result['message']}")
            return google_news_url
    except Exception as e:
        print(f"Error decoding URL: {e}")
        return google_news_url

def extract_article_content(url):
    try:
        req = requests.get(url, headers=headers, timeout=10, proxies={"http": proxy, "https": proxy} if proxy else None)
        req.raise_for_status()
        soup = BeautifulSoup(req.text, "html.parser")

        # Extract Title
        title_tag = soup.find("h1") or soup.find("h2")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Extract Body Text
        paragraphs = soup.find_all("p")
        body_text = "\n".join(p.get_text(strip=True) for p in paragraphs)

        # Extract Publication Date
        date = ""
        meta_date = soup.find("meta", {"name": "pubdate"}) or \
                    soup.find("meta", {"property": "article:published_time"}) or \
                    soup.find("meta", {"name": "date"})
        if meta_date and meta_date.has_attr("content"):
            date = meta_date["content"]

        # Extract Description
        description = ""
        meta_description = soup.find("meta", {"name": "description"})
        if meta_description and meta_description.has_attr("content"):
            description = meta_description["content"]

        return title, body_text, date, description

    except Exception as e:
        print(f"Failed to extract content from {url}: {e}")
        return "", "", "", ""

current_date = start_date
while current_date <= end_date:
    after = current_date.strftime("%Y-%m-%d")
    before = (current_date + timedelta(days=1)).strftime("%Y-%m-%d")
    rss_url = base_url.format(after=after, before=before)
    
    print(f"Fetching articles for: {after}")
    feed = feedparser.parse(rss_url)
    
    for entry in feed.entries:
        try:
            pub_date = datetime(*entry.published_parsed[:6])

            # Get Google News URL from RSS Feed
            google_news_url = entry.link

            # Decode to Final Publisher URL
            final_url = decode_google_news_url(google_news_url)

            # Extract Content from Publisher Site
            country = extract_country_from_url(final_url)
            title, body_text, date_meta, description = extract_article_content(final_url)

            articles.append({
                "media_url": final_url,
                "title": title or entry.title,
                "body_text": body_text,
                "publication_date": date_meta or pub_date.strftime("%Y-%m-%d"),
                "description": description or entry.description,
                "country": country
            })
        except Exception as e:
            print(f"Error processing article: {e}")
            continue  # Skip problematic entries safely

        # Delay after each request
        time.sleep(request_delay)

    current_date += timedelta(days=1)

# Create dataframe with the result
df_articles = pd.DataFrame(articles)

#####  Data Cleaning
def clean_date_column_precise(df, column_name):
    def extract_date(value):
        if pd.isna(value):
            return "Invalid"
        value = str(value).strip()

        # Handle exact 'DD/MM/YYYY HH:MM:SS' format or 'DD/MM/YYYY' with optional time
        if re.fullmatch(r'\d{2}/\d{2}/\d{4}( \d{2}:\d{2}:\d{2})?', value):
            try:
                return pd.to_datetime(value, format='%d/%m/%Y %H:%M:%S', errors='coerce').date()
            except:
                try:
                    return pd.to_datetime(value, format='%d/%m/%Y', errors='coerce').date()
                except:
                    return "Invalid"

        # Handle Unix timestamp-like values
        if re.fullmatch(r'\d{10}', value):
            try:
                return datetime.utcfromtimestamp(int(value)).date()
            except:
                return "Invalid"

        # Remove everything starting from 'C' (CET, CEST, etc.)
        value = re.split(r'C', value)[0]

        # Handle compact date formats like 20250317
        if re.fullmatch(r'\d{8}', value):
            parsed_date = pd.to_datetime(value, format='%Y%m%d', errors='coerce')
            return parsed_date.date() if not pd.isna(parsed_date) else "Invalid"

        # Default parsing attempt
        try:
            parsed_date = pd.to_datetime(value, errors='coerce')
            return parsed_date.date() if not pd.isna(parsed_date) else "Invalid"
        except:
            return "Invalid"

    new_column = f"{column_name}_cleaned"
    df[new_column] = df[column_name].apply(extract_date)
    return df


df_cleaned = clean_date_column_precise(df_articles, 'publication_date')

df_cleaned['media_url'] = df_cleaned['media_url'].astype(str).str.replace(r'\?.*$', '', regex=True).str.strip()

### Save cleaned results

# Generate current date
current_date = datetime.now().strftime("%Y-%m-%d")

# Define output filename with date
output_filename = f"articles_climate_cleaned_{current_date}.xlsx"


# Define the regex pattern for illegal characters (based on OpenXML)
ILLEGAL_CHARACTERS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

# Function to clean illegal characters from strings
def clean_illegal_chars(value):
    if isinstance(value, str):
        value = ILLEGAL_CHARACTERS_RE.sub("", value)
        value = value.replace('\xa0', ' ')  # Replace non-breaking space with normal space
    return value

# Apply to the entire DataFrame
df_cleaned = df_cleaned.applymap(clean_illegal_chars)

# Save to Excel after cleaning
df_cleaned.to_excel(output_filename, index=False)