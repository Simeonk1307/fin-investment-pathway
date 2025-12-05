import os
import json
import sys
from dotenv import load_dotenv

from src.layers.bronze_layer.collectors.reddit_socials import RedditHtmlProducer
from src.config.logger_config import get_module_logger
from src.utils.common import common_config, profiles


def load_tickers(logger):
    raw = os.getenv("TICKERS", "[]")
    try:
        tickers = json.loads(raw)
        if not isinstance(tickers, list) or not tickers:
            raise ValueError("TICKERS must be non-empty list")
        return tickers
    except Exception as e:
        logger.error("Invalid TICKERS JSON: %s", e, exc_info=True)
        return ["AAPL", "MSFT", "GOOGL"]  # safe default


def validate_env(logger):
    required = ["REDPANDA_BRONZE_SOCIALS_TOPIC"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        logger.error("Missing env vars: %s", ", ".join(missing))
        sys.exit(1)


def main():
    load_dotenv()
    logger = get_module_logger("RedditSocialsProducer")

    validate_env(logger)
    tickers = load_tickers(logger)

    topic = os.getenv("REDPANDA_BRONZE_SOCIALS_TOPIC")
    producer_config = common_config | profiles["high_throughput"] | {
        "client.id": "reddit-socials-producer"
    }


    try:
        producer = RedditHtmlProducer(
            logger=logger,
            topic=topic,
            producer_config=producer_config,
            tickers=tickers,
            subreddit="stocks",
            poll_interval=300,
            limit=10,
        )
        producer.run()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error("Reddit socials producer failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
