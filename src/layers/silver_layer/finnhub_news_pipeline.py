import pathway as pw
import os
from dotenv import load_dotenv
from src.schemas.silver_news_schema import FinnHubNewsSchema, finnhub_news_mapping
from src.utils.common import common_config, profiles
from src.utils.casting import create_schema_parser, cast_to_str, cast_to_int, unpack_from_schema, dedupe_from_schema
from src.KnowledgeGraph.kg_updater import KGNewsUpdater, Neo4jConfig


load_dotenv()
pw.set_license_key(os.getenv("PATHWAY_LICENSE_KEY"))

neo4j_config = Neo4jConfig(
    uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    user=os.getenv("NEO4J_USER", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD", "password"),
)
kg_updater = KGNewsUpdater(neo4j_config)

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
    schema_class=FinnHubNewsSchema,
    field_mapping=finnhub_news_mapping
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

def row_to_news_dict(row) -> dict:
    return {
        "category": row.category,
        "datetime": int(row.datetime),
        "headline": row.headline,
        "news_id": int(row.news_id),
        "image": row.image,
        "related": row.related,
        "source": row.source,
        "summary": row.summary,
        "url": row.url,
    }

def kg_update_sink(row) -> None:
    news = row_to_news_dict(row)
    kg_updater.update_kg_from_news(news)

_ = deduped.select(
    _sink = pw.apply(kg_update_sink, pw.this)
)

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