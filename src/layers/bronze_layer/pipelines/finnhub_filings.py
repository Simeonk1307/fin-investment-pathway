import os
import json
import sys
import signal
import time
import logging
from dotenv import load_dotenv

from src.layers.bronze_layer.collectors.finnhub_filings_producer import FinnhubFilingsProducer
from src.config.logger_config import get_module_logger
from src.utils.common import common_config, profiles


# ============================================================
# GLOBAL SHUTDOWN FLAGS
# ============================================================
_shutdown_flag = False
_producer_instance = None


def _signal_handler(signum, frame):
    global _shutdown_flag, _producer_instance

    print(f"\n[SecFilingsProducer] Signal {signum} received — shutting down...")
    _shutdown_flag = True

    if _producer_instance is not None:
        try:
            _producer_instance.stop()
        except Exception:
            pass


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ============================================================
# Helpers
# ============================================================
def load_tickers(logger: logging.Logger) -> list[str]:
    """Load tickers from env or fallback."""
    raw = os.getenv("TICKERS", "[]")

    try:
        tickers = json.loads(raw)
        if not isinstance(tickers, list) or not tickers:
            raise ValueError("TICKERS must be a non-empty list")
        return tickers

    except Exception:
        logger.exception("Invalid TICKERS env. Using safe defaults.")
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]


def validate_env(logger: logging.Logger):
    required = ["FINNHUB_API_KEY", "REDPANDA_BRONZE_FILINGS_TOPIC"]
    missing = [k for k in required if not os.getenv(k)]

    if missing:
        logger.error("Missing required env vars: %s", ", ".join(missing))
        sys.exit(1)


# ============================================================
# MAIN
# ============================================================
def main():
    global _producer_instance

    load_dotenv()
    logger = get_module_logger("SecFilingsProducer")

    validate_env(logger)

    tickers = load_tickers(logger)
    topic = os.getenv("REDPANDA_BRONZE_FILINGS_TOPIC")
    api_key = os.getenv("FINNHUB_API_KEY")

    # Combine Kafka/Redpanda configs
    producer_config = {
        **common_config,
        **profiles["high_throughput"],
        "client.id": "finnhub-filings-producer",
    }

    # ------------------------------
    # Startup Banner
    # ------------------------------
    logger.info("============================================================")
    logger.info("               Finnhub Filings → Redpanda Producer           ")
    logger.info("============================================================")
    logger.info("Topic           : %s", topic)
    logger.info("Tickers         : %s", ", ".join(tickers))
    logger.info("Poll Interval   : %s seconds", 600)
    logger.info("Lookback Window : %s days", 90)
    logger.info("Kafka Profile   : high_throughput")
    logger.info("============================================================")

    restart_delay = 2  # starts small, backoff increases

    while not _shutdown_flag:
        try:
            _producer_instance = FinnhubFilingsProducer(
                logger=logger,
                topic=topic,
                producer_config=producer_config,
                tickers=tickers,
                api_key=api_key,
                poll_interval=600,
                lookback_days=90,      # WORKS with Finnhub
                max_retries=5,
            )

            logger.info("[Launcher] Producer initialized.")

            # Blocking run loop
            _producer_instance.run()

        except KeyboardInterrupt:
            logger.info("[Launcher] KeyboardInterrupt — exiting...")
            break

        except Exception:
            logger.exception("[Launcher] Producer crashed unexpectedly.")

            if _shutdown_flag:
                break

            logger.warning("[Launcher] Restarting producer in %.1f seconds...", restart_delay)
            time.sleep(restart_delay)
            restart_delay = min(restart_delay * 2, 60)

        else:
            logger.info("[Launcher] Producer stopped cleanly.")
            break

    logger.info("[Launcher] Shutdown complete.")


# ============================================================
# ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    main()
