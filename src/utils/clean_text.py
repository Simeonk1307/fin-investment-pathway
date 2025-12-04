import re
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


def clean_text(text):
    clean = convert_emojis(text)
    clean = remove_urls(text)
    clean = remove_html(text)
    clean = normalize_repeated_chars(text)
    clean = normalize_spaces(text)

    return clean
