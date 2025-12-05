import os, json
from dotenv import load_dotenv
from src.config.logger_config import get_module_logger
from src.layers.bronze_layer.collectors.yfinance_stocks import YFinanceStocksProducer
from src.utils.common import common_config, profiles
import pathway as pw

# Extend this to settings in src.config.settings import Config
load_dotenv()
pw.set_license_key(os.getenv("PATHWAY_LICENSE_KEY"))

tickers = json.loads(os.getenv("TICKERS"))
logger = get_module_logger("YFinanceProducer")
topic=os.getenv("REDPANDA_BRONZE_STOCKS_TOPIC")
producer_config = common_config | profiles["high_throughput"] | {"client.id": "yfinance-stocks-producer"}

producer = YFinanceStocksProducer(
    tickers=tickers,
    logger=logger,
    topic=topic,
    producer_config=producer_config
)

logger.info("=" * 40)
logger.info("YFinance → Redpanda Stocks Producer Starting...")
logger.info(f"Topic     : {topic}")
logger.info(f"Tickers   : {tickers}")
logger.info("=" * 40)

producer.run()
