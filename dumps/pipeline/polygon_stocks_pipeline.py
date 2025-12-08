import os
import json
from dotenv import load_dotenv
import pathway as pw
from src.schemas.bronze_schemas import BronzeSchema
from src.config.logger_config import get_module_logger
from src.utils.common import common_config, profiles
from src.layers.bronze_layer.collectors.polygon_stocks import PolygonSubject


# DID NOT IMPLEMENT THIS BECAUSE OF PRICING OF THE API

load_dotenv()
pw.set_license_key(os.getenv("PATHWAY_LICENSE_KEY"))


api_key = os.getenv("POLYGON_API_KEY")
if not api_key:
    raise ValueError("POLYGON_API_KEY environment variable is required")


tickers = json.loads(os.getenv("TICKERS"))
logger = get_module_logger("POLYGONStocksProducer")
topic = os.getenv("REDPANDA_BRONZE_STOCKS_TOPIC") 
producer_config = common_config | profiles["low_latency"] | {"client.id": "polygon-stocks-producer"}


logger.info("=" * 60)
logger.info("Polygon → Redpanda Stocks Producer Starting...")
logger.info(f"Topic     : {topic}")
logger.info(f"Tickers   : {tickers}")
logger.info(f"Brokers   : {producer_config.get('bootstrap.servers')}")
logger.info("=" * 60)


subject = PolygonSubject(
    api_key=api_key,
    symbols=tickers,
    reconnect_delay=1.0,
    max_delay=60.0,
    on_error=lambda type, err, ctx: logger.error(f"WebSocket {type}: {err}"),
)


trades = pw.io.python.read(
    subject, 
    schema=BronzeSchema
)


pw.io.kafka.write(
    trades,
    rdkafka_settings=producer_config,
    topic_name=topic,
    format="json",
    key=pw.this.source,
)


try:
    logger.info("Starting Pathway pipeline...")
    pw.run()
except KeyboardInterrupt:
    logger.info("Shutdown signal received. Closing...")
except Exception as e:
    logger.error(f"Pipeline error: {e}")
    raise