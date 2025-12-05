import os
import sys
import json
from dotenv import load_dotenv
import pathway as pw

from src.schemas.bronze_schemas import UnifiedSchema  # adjust if needed
from src.layers.bronze_layer.collectors.finnhub_stocks_subject import FinnhubSubject
from src.utils.common import common_config, profiles
from src.config.logger_config import get_module_logger
import pathway as pw


def validate_env(logger):
    required = [
        "PATHWAY_LICENSE_KEY",
        "FINNHUB_API_KEY",
        "TICKERS",
        "REDPANDA_BRONZE_STOCKS_TOPIC",
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        logger.error("Missing env vars: %s", ", ".join(missing))
        sys.exit(1)


def load_tickers(logger):
    raw = os.getenv("TICKERS", "[]")
    try:
        tickers = json.loads(raw)
        if not isinstance(tickers, list) or not tickers:
            raise ValueError("TICKERS must be non-empty list")
        return tickers
    except Exception as e:
        logger.error("Invalid TICKERS JSON: %s", e, exc_info=True)
        sys.exit(1)


def main():
    load_dotenv()
    logger = get_module_logger("FinnhubStocksProducer")

    validate_env(logger)
    tickers = load_tickers(logger)

    pw.set_license_key(os.getenv("PATHWAY_LICENSE_KEY"))

    topic = os.getenv("REDPANDA_BRONZE_STOCKS_TOPIC")
    api_key = os.getenv("FINNHUB_API_KEY")

    producer_config = common_config | profiles["high_throughput"] | {
        "client.id": "finnhub-stocks-producer"
    }

    logger.info("=" * 70)
    logger.info("Finnhub Stocks → Redpanda Producer (Pathway) Starting...")
    logger.info("Topic   : %s", topic)
    logger.info("Tickers : %s", tickers)
    logger.info("Brokers : %s", producer_config.get("bootstrap.servers"))
    logger.info("=" * 70)

    def on_error(error_type: str, error: Exception, context: str = None):
        logger.error("WebSocket error [%s]: %s", error_type, error)
        if context:
            logger.debug("Context: %s", context)

    try:
        subject = FinnhubSubject(
            api_key=api_key,
            symbols=tickers,
            reconnect_delay=1.0,
            max_delay=60.0,
            on_error=on_error,
            logger=logger,
        )

        trades = pw.io.python.read(subject, schema=UnifiedSchema)

        pw.io.kafka.write(
            trades,
            rdkafka_settings=producer_config,
            topic_name=topic,
            format="json",
            key=pw.this.source,
        )

        logger.info("Kafka output configured. Starting Pathway runtime...")
        pw.run()

    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
        sys.exit(0)
    except Exception as e:
        logger.error("Stocks pipeline failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
