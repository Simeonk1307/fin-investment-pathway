import hashlib
import re
from datetime import datetime
import emoji

# Convert emoji → words (🙂 → "smile_emoji")
def convert_emojis(text):
    return emoji.demojize(text, delimiters=("", "_emoji ")).replace(" ", "_")

# Normalize elongated words: "goooood" → "good"
def normalize_repeated_chars(text):
    return re.sub(r"(.)\1{2,}", r"\1\1", text)

# Remove URLs
def remove_urls(text):
    return re.sub(r"http\S+", "", text)

# Remove HTML tags
def remove_html(text):
    return re.sub(r"<.*?>", " ", text)

# Remove extra whitespace
def normalize_spaces(text):
    return re.sub(r"\s+", " ", text).strip()

def generate_event_id(event):
    s = f"{event.get('id','')}-{event.get('source','')}-{event.get('company','')}"
    return hashlib.sha256(s.encode()).hexdigest()

def preprocess_event(event):

    raw_text = event.get("text", "") or ""
    raw_title = event.get("title", "") or ""

    # Clean text
    text = raw_text
    text = convert_emojis(text)
    text = remove_urls(text)
    text = remove_html(text)
    text = normalize_repeated_chars(text)
    text = normalize_spaces(text)

    # Clean title
    title = raw_title
    title = convert_emojis(title)
    title = remove_urls(title)
    title = remove_html(title)
    title = normalize_repeated_chars(title)
    title = normalize_spaces(title)

    unified = {
        "event_id": generate_event_id(event),
        "source": event.get("source"),
        "company": event.get("company"),
        "title": title,
        "text": text,
        "url": event.get("id"),
        "query": event.get("query"),
        "published_at": event.get("created_at") or None,
        "ingested_at": datetime.utcnow().isoformat() + "Z",
        "raw": event  # store original for debugging
    }

    return unified
