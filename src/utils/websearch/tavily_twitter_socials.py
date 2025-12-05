from tavily import TavilyClient
import time

TAVILY_KEY = "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"
client = TavilyClient(api_key=TAVILY_KEY)

CACHE = {}       # company -> (timestamp, data)
CACHE_TTL = 600  # 10 minutes


def scrape_twitter_tavily(company="tesla"):
    now = time.time()

    # ----------------------------
    # 1. Cache check
    # ----------------------------
    if company in CACHE:
        ts, data = CACHE[company]
        if now - ts < CACHE_TTL:
            print(f"[twitter_tavily] Using cached results for {company}")
            return data

    print(f"[twitter_tavily] Running Tavily search for {company}")

    # Use ONLY ONE HIGH-QUALITY QUERY
    query = f"{company} latest twitter reactions and social sentiment"

    try:
        response = client.search(
            query,
            search_depth="basic",   # very important
            max_results=8,          # enough for sentiment
        )
    except Exception as e:
        print("[Tavily error]:", e)
        return []

    items = response.get("results", [])
    out = []

    for item in items:
        out.append({
            "source": "twitter_tavily",
            "company": company,
            "query": query,
            "id": item.get("url"),
            "title": item.get("title"),
            "text": item.get("content", ""),
            "score": item.get("score"),
            "created_at": item.get("published_date", "")
        })

    # ----------------------------
    # 2. Store in cache
    # ----------------------------
    CACHE[company] = (now, out)

    return out

# from tavily import TavilyClient
# import time

# TAVILY_KEY = "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"
# client = TavilyClient(api_key=TAVILY_KEY)

# def build_queries(company):
#     return [
#         f"{company} twitter reactions",
#         f"{company} tweets news",
#         f"{company} viral tweets",
#         f"{company} trending sentiment",
#         f"{company} latest twitter buzz",
#     ]

# def scrape_twitter_tavily(company="tesla"):
#     queries = build_queries(company)
#     results_out = []

#     for q in queries:
#         try:
#             response = client.search(q, search_depth="advanced", max_results=30)
#         except Exception as e:
#             print("[Tavily error]:", e)
#             continue

#         results = response.get("results", [])
#         for item in results:
#             results_out.append({
#                 "source": "twitter_tavily",
#                 "company": company,
#                 "query": q,
#                 "id": item.get("url"),
#                 "title": item.get("title"),
#                 "text": item.get("content", ""),
#                 "score": item.get("score"),
#                 "created_at": item.get("published_date", "")
#             })

#         time.sleep(0.15)

#     return results_out
