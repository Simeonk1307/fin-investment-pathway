import time
from src.utils.producers.socials_producer import send_to_kafka
from src.utils.websearch.tavily_news import scrape_tavily_web
import os
from dotenv import load_dotenv

load_dotenv()

EQUITY_TICKERS = ["NVDA","MSFT","AAPL","GOOGL","AMZN","META","AVGO","TSLA"]

def run_once():
    for ticker in EQUITY_TICKERS:

        print(f"\n====== FETCHING DATA FOR {ticker} ======\n")

        # 3. Tavily Web Search (News + Blogs + Analysis)
        tavily_web_results = scrape_tavily_web(company=ticker.lower())
        for event in tavily_web_results:
            send_to_kafka(event, os.getenv("REDPANDA_NEWS_TOPIC"))

        print(f"Completed tickers batch for {ticker}.\n")
        time.sleep(2)

if __name__ == "__main__":
    print(" Unified Producer Started — Streaming to Redpanda...")
    while True:
        run_once()
        print("\n Sleeping for 3 minutes before next cycle...\n")
        time.sleep(180)   # repeat every 3 min
