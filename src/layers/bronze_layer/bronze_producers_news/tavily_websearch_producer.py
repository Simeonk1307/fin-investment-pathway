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
