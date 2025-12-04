import pathway as pw
import os
from dotenv import load_dotenv
from src.schemas.silver_socials_schema import SocialsSchema
from src.utils.finbert_model import get_finbert_sentiment
from src.utils.common import common_config

load_dotenv()
pw.set_license_key(os.getenv("PATHWAY_LICENSE_KEY"))

consumer_settings = common_config | {
    "group.id": "finbert-sentiment-socials",
    "auto.offset.reset": "earliest",
}

SILVER_TOPIC = os.getenv("REDPANDA_SILVER_SOCIALS_TOPIC")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


socials = pw.io.redpanda.read(
    rdkafka_settings=consumer_settings,
    topic=SILVER_TOPIC,
    schema=SocialsSchema,
    format="json",
    autocommit_duration_ms=1000
)

# TODO check if we can even do this or not or we need to use pw.apply or some thing else
# TODO check if we are getting what we need i.e in sentiment
socials = socials.with_columns(
    sentiment_tuple = get_finbert_sentiment(pw.this.text)
)

social_sentiment = socials.with_columns(
    sentiment_label = pw.this.sentiment_tuple[0],
    sentiment_score = pw.this.sentiment_tuple[1]
).without_columns(pw.this.sentiment_tuple)

pw.io.jsonlines.write(
    social_sentiment,
    os.path.join(OUTPUT_DIR, "social_sentiment.jsonl")
)

pw.run()