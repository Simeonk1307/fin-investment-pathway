from tavily import TavilyClient
import time
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def build_queries(company):
    return [
        f"{company} twitter reactions",
        f"{company} tweets news",
        f"{company} viral tweets",
        f"{company} trending sentiment",
        f"{company} latest twitter buzz",
    ]

def scrape_twitter_tavily(company):
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
                "url": item.get("url"),
                "title": item.get("title"),
                "content": item.get("content")
            })

        time.sleep(0.15)

    return results_out


