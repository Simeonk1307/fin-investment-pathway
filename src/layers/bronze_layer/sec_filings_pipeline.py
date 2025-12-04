import os, json, sys
from src.utils.producers.sec_filings_producer import SecFilingsProducer
from dotenv import load_dotenv
from src.config.logger_config import get_module_logger
from src.utils.common import common_config, profiles
import pathway as pw

# Extend this to settings in src.config.settings import Config
load_dotenv()
pw.set_license_key(os.getenv("PATHWAY_LICENSE_KEY")) 

tickers = json.loads(os.getenv("TICKERS"))
logger = get_module_logger("SecFilingsProducer")
topic = os.getenv("REDPANDA_BRONZE_FILINGS_TOPIC")
producer_config = common_config | profiles["high_throughput"] | {"client.id": "finnhub-news-producer"}
my_user_agent = os.getenv("SEC_USER_AGENT", "MyStudentProject contact@example.edu")

producer = SecFilingsProducer(
    logger=logger,
    topic=topic,
    producer_config=producer_config,
    
    user_agent=my_user_agent,
    poll_interval=60,
)

logger.info("=" * 40)
logger.info("SecFilings → Redpanda Filings Producer Starting...")
logger.info(f"Topic     : {topic}")
logger.info(f"Tickers   : {tickers}")
logger.info("=" * 40)

producer.run()