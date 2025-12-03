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
