import pathway as pw
import os
from dotenv import load_dotenv
from src.schemas.silver_news_schema import FinnHubNewsSchema
from src.utils.common import common_config, profiles
from src.utils.casting import create_schema_parser, cast_to_str, cast_to_int, unpack_from_schema, dedupe_from_schema

load_dotenv()
pw.set_license_key(os.getenv("PATHWAY_LICENSE_KEY"))

consumer_settings = common_config | {
    "group.id": "bronze-news-consumer",  # Changed from stocks to news
    # "auto.offset.reset": "latest",
    "auto.offset.reset": "latest",
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
valid = unpack_from_schema(
    table=valid, 
    schema_class=FinnHubNewsSchema, 
    source_column="data"
)

failed = with_status.filter(pw.this.success == 0)
failed = failed.select(
    error=pw.this.error,
    raw_data=pw.this.raw,
)

# Deduplicate by news_id 
deduped = dedupe_from_schema(
    table=valid, 
    schema_class=FinnHubNewsSchema, 
    dedupe_columns=["news_id"]
) # this is fine


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