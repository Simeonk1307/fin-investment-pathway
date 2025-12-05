import os, json
from src.layers.bronze_layer.collectors.finnhub_news_producer import FinnHubNewsProducer
from dotenv import load_dotenv
from src.config.logger_config import get_module_logger
from src.utils.common import common_config, profiles
import pathway as pw
import sys, signal

load_dotenv()

def main():
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    
    required = ["PATHWAY_LICENSE_KEY", "TICKERS", "REDPANDA_BRONZE_NEWS_TOPIC", "FINNHUB_API_KEY"]
    missing = [var for var in required if not os.getenv(var)]
    if missing:
        print(f"Missing: {missing}")
        sys.exit(1)
    
    try:
        pw.set_license_key(os.getenv("PATHWAY_LICENSE_KEY"))
        
        tickers = json.loads(os.getenv("TICKERS"))
        if not tickers:
            print("No tickers")
            sys.exit(1)
        
        logger = get_module_logger("FinnHubNewsProducer")
        topic = os.getenv("REDPANDA_BRONZE_NEWS_TOPIC")
        api_key = os.getenv("FINNHUB_API_KEY")
        
        config = common_config | profiles["high_throughput"] | {"client.id": "finnhub-news"}
        
        logger.info(f"Starting: {topic}, {len(tickers)} tickers")
        
        producer = FinnHubNewsProducer(
            tickers=tickers,
            logger=logger,
            topic=topic,
            api_key=api_key,
            producer_config=config,
            poll_interval=5,
            lookback_days=7,
        )
        
        producer.run()
        
    except SystemExit:
        pass  # Immediate shutdown, no cleanup needed
    except json.JSONDecodeError as e:
        print(f"JSON error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()