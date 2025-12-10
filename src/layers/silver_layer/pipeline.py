import os
import sys
import signal
import logging
import pathway as pw
from dotenv import load_dotenv, find_dotenv

from src.layers.silver_layer.core import create_silver_pipeline
from src.schemas.silver_schemas import (
    FinnHubStockSchema, finnhub_stocks_mapping,
    FinnHubNewsSchema, finnhub_news_mapping,
    SocialsSchema, socials_mapping,
    FinnhubFilingsSchema, finnhub_filings_mapping,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    stream=sys.stdout,
    force=True,
)


import time
from src.observability.helping import OTELLoggerManager, OTELMetricsManager

load_dotenv(find_dotenv())

# -------------------- OBSERVABILITY SETUP --------------------
logger_manager = OTELLoggerManager(
    service_name="Silver_pipeline_creation_logger",
    otlp_endpoint="http://localhost:4317",
)

metrics_manager = OTELMetricsManager(
    service_name="Silver_pipeline",
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

# logger = logging.getLogger(__name__)

logging.getLogger("librdkafka").setLevel(logging.CRITICAL)
logging.getLogger("confluent_kafka").setLevel(logging.CRITICAL)


def shutdown_handler(signum, frame):
    logger.info("[SILVER] Shutdown signal received")
    sys.stdout.flush()
    os._exit(0)


signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)


PIPELINES = {
    "STOCKS": {
        "schema": FinnHubStockSchema,
        "mapping": finnhub_stocks_mapping,
        "dedupe": ["symbol", "timestamp"],
        "key": "symbol",
    },
    "NEWS": {
        "schema": FinnHubNewsSchema,
        "mapping": finnhub_news_mapping,
        "dedupe": ["news_id"],
        "key": "symbol",
    },
    "SOCIALS": {
        "schema": SocialsSchema,
        "mapping": socials_mapping,
        "dedupe": ["symbol", "url", "title"],
        "key": "symbol",
    },
    "FILINGS": {
        "schema": FinnhubFilingsSchema,
        "mapping": finnhub_filings_mapping,
        "dedupe": ["access_number"],
        "key": "symbol",
    },
}

KAFKA_RESILIENCE_CONFIG = {
    "socket.timeout.ms": "30000",
    "socket.connection.setup.timeout.ms": "30000",
    "reconnect.backoff.ms": "500",
    "reconnect.backoff.max.ms": "30000",
    "message.timeout.ms": "60000",
    "request.timeout.ms": "60000",
    "session.timeout.ms": "60000",
    "heartbeat.interval.ms": "20000",
    "log.connection.close": "false",
    "log_level": "2",
}


def validate_env():
    pipeline = os.getenv("PIPELINE")
    
    if not pipeline:
        print("[SILVER] ERROR: PIPELINE not set", file=sys.stderr)
        sys.exit(1)
    
    if not os.getenv("PATHWAY_LICENSE_KEY"):
        print("[SILVER] ERROR: PATHWAY_LICENSE_KEY not set", file=sys.stderr)
        sys.exit(1)
    
    if not os.getenv("REDPANDA_BROKERS"):
        print("[SILVER] ERROR: REDPANDA_BROKERS not set", file=sys.stderr)
        sys.exit(1)
    
    if pipeline not in PIPELINES:
        print(f"[SILVER] ERROR: Invalid PIPELINE: {pipeline}", file=sys.stderr)
        print(f"[SILVER] Available: {list(PIPELINES.keys())}", file=sys.stderr)
        sys.exit(1)
    
    return pipeline


def main():
    print("[SILVER] Starting...", flush=True)
    load_dotenv(find_dotenv())
    pipeline = validate_env()
    config = PIPELINES[pipeline]
    debug = (os.getenv("DEBUG").lower() == "true")

    print(f"[SILVER] Pipeline: {pipeline}", flush=True)
    print(f"[SILVER] Mode: {'DEBUG' if debug else 'PRODUCTION'}", flush=True)

    try:
        create_silver_pipeline(
            name=pipeline,
            output_schema=config["schema"],
            field_mapping=config["mapping"],
            dedupe_columns=config["dedupe"],
            key_column=config["key"],
            debug=debug,
            extra_kafka_config=KAFKA_RESILIENCE_CONFIG,
        )
    except Exception as e:
        print(f"[SILVER] Pipeline creation failed: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("[SILVER] Pipeline ready", flush=True)
    
    try:
        pw.set_license_key(os.getenv("PATHWAY_LICENSE_KEY"))
    except Exception as e:
        print(f"[SILVER] License error: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    
    pw.set_monitoring_config(server_endpoint="http://localhost:4317")
    print("[SILVER] Starting Pathway runtime...", flush=True)
    
    while True:
        try:
            pw.run()
            break
        except KeyboardInterrupt:
            print("[SILVER] Interrupted", flush=True)
            break
        except Exception as e:
            error_str = str(e).lower()
            print(f"[SILVER] Error: {e}", flush=True)
            if any(kw in error_str for kw in ["resolve", "connection", "timeout", "kafka", "broker", "network"]):
                print("[SILVER] Connection error, retrying in 10s...", flush=True)
                try:
                    import time
                    time.sleep(10)
                except KeyboardInterrupt:
                    break
            else:
                print("[SILVER] Unknown error, retrying in 10s...", flush=True)
                try:
                    import time
                    time.sleep(10)
                except KeyboardInterrupt:
                    break

    print("[SILVER] Shutdown complete", flush=True)


if __name__ == "__main__":
    main()