import pathway as pw
import os
from dotenv import load_dotenv
from src.schemas.silver.news_schema import FinnHubNewsSchema
from src.utils.common import common_config, profiles
from src.utils.casting import create_schema_parser, cast_to_str, cast_to_int

load_dotenv()
pw.set_license_key(os.getenv("PATHWAY_LICENSE_KEY"))

consumer_settings = common_config | {
    "group.id": "bronze-news-consumer1",  # Changed from stocks to news
    # "auto.offset.reset": "latest",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": "true",
    "auto.commit.interval.ms": "500",
}

producer_settings = common_config | profiles["high_throughput"] | {"client.id": "silver-news-producer"}

BRONZE_TOPIC = os.getenv("REDPANDA_BRONZE_NEWS_TOPIC")
SILVER_TOPIC = os.getenv("REDPANDA_SILVER_NEWS_TOPIC")
DLQ_TOPIC = os.getenv("REDPANDA_SILVER_NEWS_DLQ_TOPIC")

raw = pw.io.redpanda.read(
    rdkafka_settings=consumer_settings,
    topic=BRONZE_TOPIC,
    format="raw",
    autocommit_duration_ms=500,
)

safe_parse_news = create_schema_parser(
    FinnHubNewsSchema,
    field_mapping={"news_id": "id"}
)

parsed = raw.select(
    result=safe_parse_news(pw.this.data)
)

with_status = parsed.select(
    success=cast_to_int(pw.this.result["success"]),
    data=pw.this.result["data"],
    error=cast_to_str(pw.this.result["error"]),
    raw=cast_to_str(pw.this.result["raw"]),
)

valid = with_status.filter(pw.this.success == 1)
valid = valid.select(
    category=cast_to_str(pw.this.data["category"]),
    datetime=cast_to_int(pw.this.data["datetime"]),
    headline=cast_to_str(pw.this.data["headline"]),
    news_id=cast_to_int(pw.this.data["news_id"]),
    image=cast_to_str(pw.this.data["image"]),
    related=cast_to_str(pw.this.data["related"]),
    source=cast_to_str(pw.this.data["source"]),
    summary=cast_to_str(pw.this.data["summary"]),
    url=cast_to_str(pw.this.data["url"]),
)

failed = with_status.filter(pw.this.success == 0)
failed = failed.select(
    error=pw.this.error,
    raw_data=pw.this.raw,
)

# Deduplicate by news_id 
deduped = valid.groupby(pw.this.news_id).reduce(
    category=pw.reducers.earliest(pw.this.category),
    datetime=pw.reducers.earliest(pw.this.datetime),
    headline=pw.reducers.earliest(pw.this.headline),
    news_id=pw.reducers.earliest(pw.this.news_id),
    image=pw.reducers.earliest(pw.this.image),
    related=pw.reducers.earliest(pw.this.related),
    source=pw.reducers.earliest(pw.this.source),
    summary=pw.reducers.earliest(pw.this.summary),
    url=pw.reducers.earliest(pw.this.url),
)

# Write to Silver topic with related ticker as key for partitioning
pw.io.kafka.write(
    deduped,
    rdkafka_settings=producer_settings, 
    topic_name=SILVER_TOPIC, 
    format="json",
    key=pw.this.related
)

pw.io.kafka.write(
    failed, 
    rdkafka_settings=producer_settings, 
    topic_name=DLQ_TOPIC, 
    format="json",
)

print(f"Pipeline: {BRONZE_TOPIC} → {SILVER_TOPIC} (DLQ: {DLQ_TOPIC})")

try:
    pw.run()
except KeyboardInterrupt:
    print("KeyboardInterrupt!!")