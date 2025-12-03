from kafka import KafkaProducer
import json

REDPANDA_BROKER = "d4luu1og819mdlmr61u0.any.ap-south-1.mpx.prd.cloud.redpanda.com:9092"
REDPANDA_USERNAME = "forall123"
REDPANDA_PASSWORD = "geIDt40rmgOZbzbmtogttp1nJkMTsr"
REDPANDA_TOPIC = "bronze.social"


producer = KafkaProducer(
    bootstrap_servers=REDPANDA_BROKER,
    security_protocol="SASL_SSL",
    sasl_mechanism="SCRAM-SHA-256",
    sasl_plain_username=REDPANDA_USERNAME,
    sasl_plain_password=REDPANDA_PASSWORD,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

def send_to_kafka(event: dict):
    try:
        # Use the original source field directly
        key_str = event.get("source", "unknown")

        producer.send(
            REDPANDA_TOPIC,
            key=key_str.encode("utf-8"),   # KEY = original source
            value=event
        )

        producer.flush()
        print(f" Sent [{key_str}]:", event.get("id"))

    except Exception as e:
        print(" Kafka send error:", e)


import requests
from bs4 import BeautifulSoup
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def scrape_reddit(company="tesla", subreddit="stocks", limit=20):
    print(f"[reddit_html] Searching r/{subreddit} for '{company}'")

    search_query = company.replace(" ", "+")
    url = f"https://old.reddit.com/r/{subreddit}/search/?q={search_query}&restrict_sr=1&sort=new"

    try:
        response = requests.get(url, headers=HEADERS)
    except Exception as e:
        print("Request failed:", e)
        return []

    if response.status_code != 200:
        print("Request blocked:", response.status_code)
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    posts = soup.find_all("div", class_="search-result")

    events = []
    count = 0

    for p in posts:
        title_tag = p.find("a", class_="search-title")
        if not title_tag:
            continue

        title = title_tag.text.strip()
        link = title_tag.get("href", "")

        events.append({
            "source": "reddit_html",
            "company": company,
            "subreddit": subreddit,
            "id": link,
            "text": title,
            "created_at": ""
        })

        count += 1
        if count >= limit:
            break

    return events


from tavily import TavilyClient
import time

TAVILY_KEY = "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"
client = TavilyClient(api_key=TAVILY_KEY)

def build_queries(company):
    return [
        f"{company} twitter reactions",
        f"{company} tweets news",
        f"{company} viral tweets",
        f"{company} trending sentiment",
        f"{company} latest twitter buzz",
    ]

def scrape_twitter_tavily(company="tesla"):
    queries = build_queries(company)
    results_out = []

    for q in queries:
        try:
            response = client.search(q, search_depth="advanced", max_results=30)
        except Exception as e:
            print("[Tavily error]:", e)
            continue

        results = response.get("results", [])
        for item in results:
            results_out.append({
                "source": "twitter_tavily",
                "company": company,
                "query": q,
                "id": item.get("url"),
                "title": item.get("title"),
                "text": item.get("content", ""),
                "score": item.get("score"),
                "created_at": item.get("published_date", "")
            })

        time.sleep(0.15)

    return results_out
import time
from kafka_producer import send_to_kafka

from reddit_producer import scrape_reddit
from twitter_tavily_producer import scrape_twitter_tavily
from tavily_websearch_producer import scrape_tavily_web

EQUITY_TICKERS = ["NVDA","MSFT","AAPL","GOOGL","AMZN","META","AVGO","TSLA"]

def run_once():
    for ticker in EQUITY_TICKERS:

        print(f"\n====== FETCHING DATA FOR {ticker} ======\n")

        # 1. Reddit HTML
        reddit_posts = scrape_reddit(company=ticker.lower(), subreddit="stocks", limit=10)
        for event in reddit_posts:
            send_to_kafka(event)

        # 2. Twitter Tavily Search
        tavily_twitter_results = scrape_twitter_tavily(company=ticker.lower())
        for event in tavily_twitter_results:
            send_to_kafka(event)

        # 3. Tavily Web Search (News + Blogs + Analysis)
        # tavily_web_results = scrape_tavily_web(company=ticker.lower())
        # for event in tavily_web_results:
        #     send_to_kafka(event)

        print(f"Completed tickers batch for {ticker}.\n")
        time.sleep(2)

if __name__ == "__main__":
    print(" Unified Producer Started — Streaming to Redpanda...")
    while True:
        run_once()
        print("\n Sleeping for 3 minutes before next cycle...\n")
        time.sleep(180)   # repeat every 3 min


