from kafka import KafkaConsumer
import json
from finbert_model import get_finbert_sentiment
from sentiment_producer import send_sentiment

CLEAN_TOPIC = "silver.socials"

consumer = KafkaConsumer(
    CLEAN_TOPIC,
    bootstrap_servers="d4luu1og819mdlmr61u0.any.ap-south-1.mpx.prd.cloud.redpanda.com:9092",
    security_protocol="SASL_SSL",
    sasl_mechanism="SCRAM-SHA-256",
    sasl_plain_username="forall123",
    sasl_plain_password="geIDt40rmgOZbzbmtogttp1nJkMTsr",
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    auto_offset_reset="latest",             # IMPORTANT FIX
    enable_auto_commit=True,
    group_id="sentiment-group-v3"           # NEW GROUP ID
)

print("FinBERT Sentiment Consumer Started...")

for msg in consumer:
    clean_event = msg.value

    text = clean_event.get("text", "")
    label, score = get_finbert_sentiment(text)

    enriched_event = {
        **clean_event,
        "sentiment_label": label,
        "sentiment_score": score
    }

    send_sentiment(enriched_event)

    # SAFEST POSSIBLE PRINT — NEVER CRASHES
    event_id = clean_event.get("event_id", "missing_event_id")
    print(f"✔ Sentiment for {event_id}: {label} ({score:.3f})")
