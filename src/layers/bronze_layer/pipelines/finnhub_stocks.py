import os
import sys
import json
import uuid
import signal
import logging
import pathway as pw
from dotenv import load_dotenv
from datetime import datetime
from confluent_kafka.admin import AdminClient
from confluent_kafka import Producer

from src.schemas.bronze_schemas import UnifiedSchema
from src.layers.bronze_layer.collectors.finnhub_stocks_subject import FinnhubSubject
from src.utils.common import common_config, profiles
from src.config.logger_config import get_module_logger


from src.observability.helping import OTELLoggerManager, OTELMetricsManager

import time

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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s"
)
# logger = logging.getLogger("bronze.stocks")
logger = logger_manager.get_logger()

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

    with open(os.path.join(folder, f"{ts}-{name}.log"), "a") as f:
        f.write(str(content) + "\n")


def debug_snapshot(name, **data):
    write_debug_file("bronze/stocks", name, {"ts": datetime.utcnow().isoformat(), **data})


def validate_env(logger):
    required = [
        "PATHWAY_LICENSE_KEY",
        "FINNHUB_API_KEY",
        "TICKERS",
        "REDPANDA_BRONZE_STOCKS_TOPIC",
        "REDPANDA_BROKERS",
        "REDPANDA_SECURITY_PROTOCOL",
        "REDPANDA_SASL_MECHANISM",
        "REDPANDA_USERNAME",
        "REDPANDA_PASSWORD",
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        logger.error("Missing env vars: %s", ", ".join(missing))
        sys.exit(1)


def load_tickers(logger):
    raw = os.getenv("TICKERS", "[]")
    try:
        t = json.loads(raw)
        if isinstance(t, list) and t:
            return t
    except:
        pass
    logger.error("Invalid TICKERS JSON")
    sys.exit(1)


def validate_broker(config):
    md = AdminClient(config).list_topics(timeout=10)
    logger.info(f"Broker OK ({len(md.brokers)} brokers, {len(md.topics)} topics)")


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
    load_dotenv()
    validate_env(logger)

    pw.set_license_key(os.getenv("PATHWAY_LICENSE_KEY"))

    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    tickers = load_tickers(logger)
    topic = os.getenv("REDPANDA_BRONZE_STOCKS_TOPIC")
    api_key = os.getenv("FINNHUB_API_KEY")

    producer_id = f"finnhub-stocks-{uuid.uuid4().hex[:6]}"

    producer_config = (
        common_config
        | profiles["high_throughput"]
        | {"client.id": producer_id}
        | {"log_level": "0"}
    )


    section("Validating Finnhub Stocks Bronze Pipeline")

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
            producer_id=producer_id,
            producer_config_keys=list(producer_config.keys()),
        )

    section("Starting Finnhub Stocks Bronze Producer")
    logger.info(f"Mode: {'DEBUG' if DEBUG else 'PROD'}")
    logger.info(f"Tickers: {len(tickers)}")
    logger.info(f"Topic: {topic}")

    def on_error(error_type, err, ctx=None):
        logger.error(f"WS error [{error_type}]: {err}")

    try:
        subject = FinnhubSubject(
            api_key=api_key,
            symbols=tickers,
            logger=logger,
            on_error=on_error,
            reconnect_delay=1.0,
            max_delay=60.0,
            debug=DEBUG,
            debug_writer=write_debug_file

        )

        trades = pw.io.python.read(subject, schema=UnifiedSchema)

        pw.io.kafka.write(
            trades,
            rdkafka_settings=producer_config,
            topic_name=topic,
            key=pw.this.source,

            format="json",
        )

        logger.info("Kafka output configured. Starting Pathway...")
        pw.run()

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
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    main()
