import os
import sys
import json
import time
import uuid
import signal
import logging
import pathway as pw
import finnhub

from dotenv import load_dotenv
from datetime import datetime
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient

from src.layers.bronze_layer.collectors.finnhub_news_producer import FinnHubNewsProducer
from src.utils.common import common_config, profiles

from src.observability.helping import OTELLoggerManager, OTELMetricsManager


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s"
)



# -------------------- OBSERVABILITY SETUP --------------------
logger_manager = OTELLoggerManager(
    service_name="bronze_news_pipeline_logs",
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


# logger = logging.getLogger("bronze.news")


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

    logger.debug(f"Debug written: {path}")


def debug_snapshot(name, **data):
    write_debug_file("bronze/news", name, {
        "ts": datetime.utcnow().isoformat(),
        **data
    })


def validate_broker(config):
    admin = AdminClient(config)
    md = admin.list_topics(timeout=10)
    logger.info(f"Broker OK ({len(md.brokers)} brokers, {len(md.topics)} topics)")
    return md


def validate_topic(config, topic):
    md = AdminClient(config).list_topics(timeout=10)
    if topic not in md.topics:
        raise ValueError(f"Topic missing: {topic}")
    logger.info(f"Topic OK: {topic}")


def validate_producer(config):
    p = Producer(config)
    p.flush(5)
    logger.info("Producer OK")


def validate_api_key(api_key):
    client = finnhub.Client(api_key=api_key)
    client.company_news("AAPL", _from="2024-01-01", to="2024-01-01")
    logger.info("Finnhub API key OK")


def validate_tickers(api_key, tickers):
    client = finnhub.Client(api_key=api_key)
    valid = []

    for t in tickers:
        start = time.time()
        try:
            profile = client.company_profile2(symbol=t)
            latency = round((time.time() - start) * 1000, 2)

            if profile and profile.get("ticker"):
                valid.append(t)
                logger.info(f"✓ {t} ({latency} ms)")
            else:
                logger.warning(f"✗ {t} ({latency} ms)")
        except Exception as e:
            logger.warning(f"{t} error: {e}")

        time.sleep(0.05)

    if not valid:
        raise ValueError("No valid tickers")

    logger.info(f"Tickers OK ({len(valid)}/{len(tickers)})")
    return valid


def main():
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    required = [
        "PATHWAY_LICENSE_KEY",
        "REDPANDA_BRONZE_NEWS_TOPIC",
        "FINNHUB_API_KEY",
        "TICKERS",
        "REDPANDA_BROKERS",
        "REDPANDA_SECURITY_PROTOCOL",
        # "REDPANDA_SASL_MECHANISM",
        # "REDPANDA_USERNAME",
        # "REDPANDA_PASSWORD"
    ]

    missing = [v for v in required if not os.getenv(v)]
    if missing:
        logger.error(f"Missing env vars: {missing}")
        sys.exit(1)

    pw.set_license_key(os.getenv("PATHWAY_LICENSE_KEY"))

    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    SKIP_TICKER_VALIDATION = os.getenv("SKIP_TICKER_VALIDATION", "false").lower() == "true"

    try:
        tickers = json.loads(os.getenv("TICKERS"))
    except Exception:
        logger.error("Invalid TICKERS JSON")
        sys.exit(1)

    topic = os.getenv("REDPANDA_BRONZE_NEWS_TOPIC")
    api_key = os.getenv("FINNHUB_API_KEY")

    consumer_id = (
        f"bronze-news-consumer-debug-{int(time.time())}"
        if DEBUG else "bronze-news-consumer"
    )

    producer_config = (
        common_config
        | profiles["low_latency"]
        | {"client.id": f"finnhub-news-{uuid.uuid4().hex[:6]}"}
        | {"log_level": 0}
    )

    consumer_config = common_config | {
        "group.id": consumer_id,
        "auto.offset.reset": "earliest" if DEBUG else "latest",
        "enable.auto.commit": "true",
        "auto.commit.interval.ms": "500",
    }

    section("Validating Bronze Pipeline")

    try:
        validate_broker(common_config)
        validate_topic(common_config, topic)
        validate_producer(producer_config)
        validate_api_key(api_key)
    except Exception as e:
        logger.error("")
        logger.error("▼▼▼ VALIDATION ERROR ▼▼▼")
        logger.error(f"{type(e).__name__}: {e}")
        logger.error("▲▲▲ END ERROR ▲▲▲")
        logger.error("")
        sys.exit(1)

    if not SKIP_TICKER_VALIDATION:
        try:
            tickers = validate_tickers(api_key, tickers)
        except Exception as e:
            logger.error("")
            logger.error("▼▼▼ TICKER VALIDATION ERROR ▼▼▼")
            logger.error(f"{type(e).__name__}: {e}")
            logger.error("▲▲▲ END ERROR ▲▲▲")
            logger.error("")
            sys.exit(1)
    else:
        logger.info("Ticker validation skipped")

    if DEBUG:
        debug_snapshot(
            "startup",
            topic=topic,
            tickers=tickers,
            producer_id=producer_config.get("client.id"),
            consumer_id=consumer_id,
            producer_config_keys=list(producer_config.keys()),
            consumer_config_keys=list(consumer_config.keys()),
        )

    section("Starting FinnHub News Bronze Producer")
    logger.info(f"Mode: {'DEBUG' if DEBUG else 'PROD'}")
    logger.info(f"Group: {consumer_id}")
    logger.info(f"Tickers: {len(tickers)}")
    logger.info(f"Topic: {topic}")

    try:
        producer = FinnHubNewsProducer(
            tickers=tickers,
            logger=logger,
            topic=topic,
            api_key=api_key,
            producer_config=producer_config,
            poll_interval=10,
            lookback_days=30,
            debug=DEBUG,
            debug_writer=write_debug_file,
        )
        producer.run()

    except SystemExit:
        logger.info("Shutdown signal received")

    except Exception as e:
        logger.error("")
        logger.error("▼▼▼ RUNTIME ERROR ▼▼▼")
        logger.error(f"{type(e).__name__}: {e}", exc_info=True)
        logger.error("▲▲▲ END ERROR ▲▲▲")
        logger.error("")
        sys.exit(1)


if __name__ == "__main__":
    main()
