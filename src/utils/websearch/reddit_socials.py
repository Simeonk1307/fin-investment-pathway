import requests
from bs4 import BeautifulSoup
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}


# ----------------------------------------
# Helper function: Extract full post content + metadata
# ----------------------------------------
def extract_full_post(url):
    """Fetch full post content + author + upvotes + comments from a Reddit post."""
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        if res.status_code != 200:
            return "", "", 0, 0
    except:
        return "", "", 0, 0

    soup = BeautifulSoup(res.text, "html.parser")

    # Full post content (selftext)
    md_div = soup.find("div", class_="md")
    content = md_div.get_text(separator="\n", strip=True) if md_div else ""

    # Author
    author_tag = soup.find("a", class_="author")
    author = author_tag.text.strip() if author_tag else "unknown"

    # Upvotes
    score_tag = soup.find("div", class_="score")
    try:
        upvotes = int(score_tag["title"]) if score_tag and "title" in score_tag.attrs else 0
    except:
        upvotes = 0

    # Comment count
    comments_tag = soup.find("a", string=lambda x: x and "comment" in x.lower())
    try:
        comment_count = int(comments_tag.text.split()[0]) if comments_tag else 0
    except:
        comment_count = 0

    return content, author, upvotes, comment_count



# ----------------------------------------
# Main search scraper
# ----------------------------------------
def scrape_reddit(company="tesla", subreddit="stocks", limit=10):
    print(f"[reddit_html] Searching r/{subreddit} for '{company}'")

    search_query = company.replace(" ", "+")
    url = (
        f"https://old.reddit.com/r/{subreddit}/search/?q={search_query}"
        f"&restrict_sr=1&sort=new"
    )

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print("Blocked or error:", response.status_code)
            return []
    except Exception as e:
        print("Request failed:", e)
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    posts = soup.find_all("div", class_="search-result")

    events = []

    for p in posts[:limit]:

        # -------- Title + URL --------
        title_tag = p.find("a", class_="search-title")
        if not title_tag:
            continue

        title = title_tag.text.strip()
        link = title_tag.get("href", "")

        # -------- Timestamp --------
        time_tag = p.find("time")
        created_at = time_tag.get("datetime", "") if time_tag else ""

        # -------- Fetch full content + metadata --------
        content, author, upvotes, comment_count = extract_full_post(link)

        # -------- Text for sentiment --------
        full_text = title + "\n" + content if content else title

        # -------- Store in final event dict --------
        events.append({
            "source": "reddit_html",
            "company": company,
            "subreddit": subreddit,
            "id": link,
            "title": title,
            "text": full_text,   # <-- Sentiment input
            "content": content,
            "author": author,
            "upvotes": upvotes,
            "comments": comment_count,
            "created_at": created_at
        })

        time.sleep(0.5)  # avoid rate limits

    return events

# import requests
# from bs4 import BeautifulSoup
# import time

# HEADERS = {
#     "User-Agent": "Mozilla/5.0"
# }

# def scrape_reddit(company="tesla", subreddit="stocks", limit=20):
#     print(f"[reddit_html] Searching r/{subreddit} for '{company}'")

#     search_query = company.replace(" ", "+")
#     url = f"https://old.reddit.com/r/{subreddit}/search/?q={search_query}&restrict_sr=1&sort=new"

#     try:
#         response = requests.get(url, headers=HEADERS)
#     except Exception as e:
#         print("Request failed:", e)
#         return []

#     if response.status_code != 200:
#         print("Request blocked:", response.status_code)
#         return []

#     soup = BeautifulSoup(response.text, "html.parser")
#     posts = soup.find_all("div", class_="search-result")

#     events = []
#     count = 0

#     for p in posts:
#         title_tag = p.find("a", class_="search-title")
#         if not title_tag:
#             continue

#         title = title_tag.text.strip()
#         link = title_tag.get("href", "")

#         events.append({
#             "source": "reddit_html",
#             "company": company,
#             "subreddit": subreddit,
#             "id": link,
#             "text": title,
#             "created_at": ""
#         })

#         count += 1
#         if count >= limit:
#             break

#     return events
