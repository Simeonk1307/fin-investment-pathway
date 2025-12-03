from kafka import KafkaProducer
import json

REDPANDA_BROKER = "d4luu1og819mdlmr61u0.any.ap-south-1.mpx.prd.cloud.redpanda.com:9092"
REDPANDA_USERNAME = "forall123"
REDPANDA_PASSWORD = "geIDt40rmgOZbzbmtogttp1nJkMTsr"
REDPANDA_TOPIC = "bronze.news"

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




from tavily import TavilyClient
import time

TAVILY_KEY = "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"
client = TavilyClient(api_key=TAVILY_KEY)

def build_queries(company):
    return [
        f"{company} stock news",
        f"{company} analysis today",
        f"{company} financial news",
        f"{company} company updates",
        f"{company} latest market news",
        f"{company} earnings insights",
    ]

def scrape_tavily_web(company="tesla"):
    queries = build_queries(company)
    collected = []

    print(f"[tavily_web] Running WebSearch for {company}")

    for q in queries:
        print(f"  -> Query: {q}")

        try:
            response = client.search(
                q,
                search_depth="advanced",
                max_results=50,
            )
        except Exception as e:
            print("[ERROR calling Tavily WebSearch]:", e)
            continue

        items = response.get("results", [])

        for item in items:
            rec = {
                "source": "tavily_web",
                "company": company,
                "query": q,
                "id": item.get("url", ""),
                "title": item.get("title", ""),
                "text": item.get("content") or item.get("raw_content") or "",
                "score": item.get("score"),
                "created_at": item.get("published_date", "")
            }
            collected.append(rec)

        time.sleep(0.15)

    return collected

import time
from kafka_producer import send_to_kafka

from reddit_producer import scrape_reddit
from twitter_tavily_producer import scrape_twitter_tavily
from tavily_websearch_producer import scrape_tavily_web

EQUITY_TICKERS = ["NVDA","MSFT","AAPL","GOOGL","AMZN","META","AVGO","TSLA"]

def run_once():
    for ticker in EQUITY_TICKERS:

        print(f"\n====== FETCHING DATA FOR {ticker} ======\n")

        # # 1. Reddit HTML
        # reddit_posts = scrape_reddit(company=ticker.lower(), subreddit="stocks", limit=10)
        # for event in reddit_posts:
        #     send_to_kafka(event)

        # # 2. Twitter Tavily Search
        # tavily_twitter_results = scrape_twitter_tavily(company=ticker.lower())
        # for event in tavily_twitter_results:
        #     send_to_kafka(event)

        # 3. Tavily Web Search (News + Blogs + Analysis)
        tavily_web_results = scrape_tavily_web(company=ticker.lower())
        for event in tavily_web_results:
            send_to_kafka(event)

        print(f"Completed tickers batch for {ticker}.\n")
        time.sleep(2)

if __name__ == "__main__":
    print(" Unified Producer Started — Streaming to Redpanda...")
    while True:
        run_once()
        print("\n Sleeping for 3 minutes before next cycle...\n")
        time.sleep(180)   # repeat every 3 min


# import os, json, sys
# from src.utils.producers.news_producer import FinnHubNewsProducer
# from dotenv import load_dotenv
# from src.config.logger_config import get_module_logger
# from src.utils.common import common_config, profiles
# import pathway as pw

# # Extend this to settings in src.config.settings import Config
# load_dotenv()
# pw.set_license_key(os.getenv("PATHWAY_LICENSE_KEY")) 

# tickers = json.loads(os.getenv("TICKERS"))
# logger = get_module_logger("FinnHubNewsProducer")
# topic = os.getenv("REDPANDA_BRONZE_NEWS_TOPIC")
# api_key = os.getenv("FINNHUB_API_KEY") 
# producer_config = common_config | profiles["high_throughput"] | {"client.id": "finnhub-news-producer"}


# producer = FinnHubNewsProducer(
#     tickers=tickers,
#     logger=logger,
#     topic=topic,
#     api_key=api_key,
#     producer_config=producer_config,

#     poll_interval=300,
#     lookback_days=2,  
# )

# logger.info("=" * 40)   
# logger.info("FinnHub → Redpanda News Producer Starting...")
# logger.info(f"Topic     : {topic}")
# logger.info(f"Tickers   : {tickers}")
# logger.info("=" * 40)


# producer.run()
