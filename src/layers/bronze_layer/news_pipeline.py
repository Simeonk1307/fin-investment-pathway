


import os, json, sys
from src.utils.producers.news_producer import FinnHubNewsProducer
from dotenv import load_dotenv
from src.config.logger_config import get_module_logger
from src.utils.common import common_config, profiles
import pathway as pw

# Extend this to settings in src.config.settings import Config
load_dotenv()
pw.set_license_key(os.getenv("PATHWAY_LICENSE_KEY")) 

tickers = json.loads(os.getenv("TICKERS"))
logger = get_module_logger("FinnHubNewsProducer")
topic = os.getenv("REDPANDA_BRONZE_NEWS_TOPIC")
api_key = os.getenv("FINNHUB_API_KEY") 
producer_config = common_config | profiles["high_throughput"] | {"client.id": "finnhub-news-producer"}


producer = FinnHubNewsProducer(
    tickers=tickers,
    logger=logger,
    topic=topic,
    api_key=api_key,
    producer_config=producer_config,

    poll_interval=300,
    lookback_days=2,  
)

logger.info("=" * 40)   
logger.info("FinnHub → Redpanda News Producer Starting...")
logger.info(f"Topic     : {topic}")
logger.info(f"Tickers   : {tickers}")
logger.info("=" * 40)


producer.run()
