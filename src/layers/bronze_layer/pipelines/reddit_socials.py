import os
import sys
import json
import time
import uuid
import signal
import logging
from datetime import datetime
from dotenv import load_dotenv

from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient

from src.layers.bronze_layer.collectors.reddit_socials import RedditHtmlProducer
from src.utils.common import common_config, profiles
from src.config.logger_config import get_module_logger

from src.observability.helping import OTELLoggerManager, OTELMetricsManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s"
)



# -------------------- OBSERVABILITY SETUP --------------------
logger_manager = OTELLoggerManager(
    service_name="Logger",
    otlp_endpoint="http://localhost:4317",
)

metrics_manager = OTELMetricsManager(
    service_name="bronze_news_pipeline_metrics",
    otlp_endpoint="http://localhost:4317",
)
# logger = logging.getLogger("bronze.news")

ticker_count = metrics_manager.counter(
    "tickers_processed",
    "Total processed tickers",
)

ws_messages = metrics_manager.counter(
    "ws_messages_received",
    "Total WebSocket messages received from Finnhub",
)

kafka_latency = metrics_manager.histogram(
    "kafka_produce_latency_seconds",
    "Latency for producing messages to Kafka",
    unit="s",
)

restarts = metrics_manager.counter(
    "finnhub_restarts",
    "Number of times the Finnhub websocket connection  restart was attempted",
)

def record_kafka_latency(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        kafka_latency.record(time.time() - start)
        return result
    return wrapper

# --------------------------------------------------------------------


logger = logger_manager.get_logger()
# logger = get_logger = logging.getLogger("bronze.socials")


def section(title):
    logger.info("")
    logger.info("=" * 80)
    logger.info(title)
    logger.info("=" * 80)


def write_debug_file(pipeline, name, content):
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    folder = os.path.join("debug_output", pipeline)
    os.makedirs(folder, exist_ok=True)

    if isinstance(content, (dict, list)):
        content = json.dumps(content, indent=2, sort_keys=True)

    path = os.path.join(folder, f"{ts}-{name}.log")
    with open(path, "a") as f:
        f.write(str(content) + "\n")


def debug_snapshot(name, **data):
    write_debug_file("bronze/socials", name, {"ts": datetime.utcnow().isoformat(), **data})


def validate_env(logger):
    required = ["REDPANDA_BRONZE_SOCIALS_TOPIC"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        logger.error(f"Missing env vars: {missing}")
        sys.exit(1)


def load_tickers(logger):
    raw = os.getenv("TICKERS", "[]")
    try:
        tickers = json.loads(raw)
        if isinstance(tickers, list) and tickers:
            return tickers
    except:
        pass

    logger.warning("Invalid or empty TICKERS, using defaults")
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]


def validate_broker(config):
    md = AdminClient(config).list_topics(timeout=10)
    logger.info(f"Broker OK ({len(md.brokers)} brokers, {len(md.topics)} topics)")
    return md


def validate_topic(config, topic):
    md = AdminClient(config).list_topics(timeout=10)
    if topic not in md.topics:
        raise ValueError(f"Topic missing: {topic}")
    logger.info(f"Topic OK: {topic}")


def validate_producer(config):
    p = Producer(config)
    p.flush(3)
    logger.info("Producer OK")


def main():
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    load_dotenv()
    validate_env(logger)

    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    topic = os.getenv("REDPANDA_BRONZE_SOCIALS_TOPIC")
    tickers = load_tickers(logger)

    producer_config = (
        common_config
        | profiles["high_throughput"]
        | {"client.id": f"reddit-socials-{uuid.uuid4().hex[:6]}", "log_level": 0}
    )

    section("Validating Reddit Socials Bronze Pipeline")

    try:
        validate_broker(common_config)
        validate_topic(common_config, topic)
        validate_producer(producer_config)
    except Exception as e:
        logger.error("")
        logger.error("▼ VALIDATION ERROR ▼")
        logger.error(f"{type(e).__name__}: {e}")
        logger.error("▲ END ERROR ▲")
        logger.error("")
        sys.exit(1)

    if DEBUG:
        debug_snapshot(
            "startup",
            topic=topic,
            tickers=tickers,
            producer_id=producer_config.get("client.id"),
            producer_config_keys=list(producer_config.keys()),
        )

    section("Starting Reddit Socials Bronze Producer")
    logger.info(f"Mode: {'DEBUG' if DEBUG else 'PROD'}")
    logger.info(f"Tickers: {len(tickers)}")
    logger.info(f"Topic: {topic}")

    try:
        producer = RedditHtmlProducer(
            logger=logger,
            topic=topic,
            producer_config=producer_config,
            tickers=tickers,
            subreddit="stocks",
            poll_interval=300,
            limit=50,
            debug=DEBUG,
            debug_writer=write_debug_file,
        )
        producer.run()

    except SystemExit:
        logger.info("Shutdown signal received")
    except Exception as e:
        logger.error("")
        logger.error("▼ RUNTIME ERROR ▼")
        logger.error(f"{type(e).__name__}: {e}", exc_info=True)
        logger.error("▲ END ERROR ▲")
        logger.error("")
        sys.exit(1)


if __name__ == "__main__":
    main()
