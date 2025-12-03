import time
from kafka_producer import send_to_kafka

from reddit_producer import scrape_reddit
from twitter_tavily_producer import scrape_twitter_tavily

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

        print(f"Completed tickers batch for {ticker}.\n")
        time.sleep(2)

if __name__ == "__main__":
    print(" Unified Producer Started — Streaming to Redpanda...")
    while True:
        run_once()
        print("\n Sleeping for 3 minutes before next cycle...\n")
        time.sleep(180)   # repeat every 3 min
