import time
from src.utils.producers.socials_producer import send_to_kafka
from src.utils.websearch.reddit_socials import scrape_reddit
from src.utils.websearch.tavily_twitter_socials import scrape_twitter_tavily
from src.utils.event_envelope import create_event_envelope
import os
from dotenv import load_dotenv

load_dotenv()

EQUITY_TICKERS = os.getenv("TICKERS")

def run_once():
    for ticker in EQUITY_TICKERS:

        print(f"\n====== FETCHING DATA FOR {ticker} ======\n")

        # 1. Reddit HTML
        reddit_posts = scrape_reddit(company=ticker.lower(), subreddit="stocks", limit=10)
        for event in reddit_posts:
            data = create_event_envelope(
                payload=event,
                source="reddit",
                source_type="webscrape"
            )
            send_to_kafka(data)

        # 2. Twitter Tavily Search
        tavily_twitter_results = scrape_twitter_tavily(company=ticker.lower())
        for event in tavily_twitter_results:
            data = create_event_envelope(
                payload=event,
                source="tavily-twitter",
                source_type="websearch"
            )
            send_to_kafka(data)

        print(f"Completed tickers batch for {ticker}.\n")
        time.sleep(2)

if __name__ == "__main__":
    print(" Socials Producer Started — Streaming to Redpanda...")
    while True:
        run_once()
        print("\n Sleeping for 3 minutes before next cycle...\n")
        time.sleep(180)   # repeat every 3 min
