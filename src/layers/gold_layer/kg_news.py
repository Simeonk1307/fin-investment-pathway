import os
import sys
import logging
import time
import pathway as pw
from dotenv import load_dotenv, find_dotenv

from src.schemas.silver_schemas import FinnHubNewsSchema
from src.utils.common import common_config, profiles
from src.KnowledgeGraph.kg_news_updater import KGNewsUpdater, Neo4jConfig

load_dotenv(find_dotenv)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger(__name__)

pw.set_license_key(os.getenv("PATHWAY_LICENSE_KEY"))

neo4j_config = Neo4jConfig(
    uri=os.getenv("NEO4J_URI"),
    user=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

try:
    kg_updater = KGNewsUpdater(neo4j_config)
    logger.info("[KG] Neo4j connected")
except Exception as e:
    logger.error(f"[KG] Neo4j connection failed: {e}")
    sys.exit(1)

consumer_settings = common_config | {
    "group.id": f"kg-news-consumer{time.time()}",
    "auto.offset.reset": "earliest",
}

SILVER_TOPIC = os.getenv("REDPANDA_SILVER_NEWS_TOPIC", "silver.news")
logger.info(f"[PIPELINE] Reading from {SILVER_TOPIC}")

news = pw.io.redpanda.read(
    rdkafka_settings=consumer_settings,
    topic=SILVER_TOPIC,
    schema=FinnHubNewsSchema,
    format="json",
    autocommit_duration_ms=1000,
)

_stats = {"processed": 0, "events": 0, "errors": 0}
_LOG_INTERVAL = 100


@pw.udf
def update_kg(
    news_id: int,
    symbol: str,
    timestamp: int,
    source: str,
    category: str,
    title: str,
    content: str,
    url: str,
    image_url: str,
) -> int:
    global _stats
    try:
        result = kg_updater.update_kg_from_news(
            {
                "news_id": news_id,
                "symbol": symbol or "",
                "timestamp": timestamp,
                "source": source or "",
                "category": category or "",
                "title": title or "",
                "content": content or "",
                "url": url or "",
                "image_url": image_url or "",
            }
        )

        _stats["processed"] += 1
        if result.get("events_detected"):
            _stats["events"] += len(result["events_detected"])

        if _stats["processed"] % _LOG_INTERVAL == 0:
            logger.info(
                f"[KG] Processed: {_stats['processed']} | "
                f"Events: {_stats['events']} | Errors: {_stats['errors']}"
            )

        return 1
    except Exception:
        _stats["errors"] += 1
        return 0


result = news.select(
    news_id=pw.this.news_id,
    symbol=pw.this.symbol,
    kg_status=update_kg(
        pw.this.news_id,
        pw.this.symbol,
        pw.this.timestamp,
        pw.this.source,
        pw.this.category,
        pw.this.title,
        pw.this.content,
        pw.this.url,
        pw.this.image_url,
    ),
)

pw.io.jsonlines.write(result, "kg_news_output.jsonl")

logger.info("[PIPELINE] Running...")

try:
    pw.run()
except KeyboardInterrupt:
    logger.info("[PIPELINE] Shutting down...")
finally:
    kg_updater.close()
    logger.info(
        f"[PIPELINE] Final stats - Processed: {_stats['processed']} | "
        f"Events: {_stats['events']} | Errors: {_stats['errors']}"
    )
