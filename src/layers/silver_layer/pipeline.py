import os
import sys
import logging
from src.layers.silver_layer.core import create_silver_pipeline, run
from src.schemas.silver_schemas import (
    FinnHubStockSchema, finnhub_stocks_mapping,
    FinnHubNewsSchema, finnhub_news_mapping,
    SocialsSchema, socials_mapping,
    FinnhubFilingsSchema, finnhub_filings_mapping,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger(__name__)

PIPELINES = {
    "STOCKS": (FinnHubStockSchema, finnhub_stocks_mapping, ["symbol", "timestamp"], "symbol"),
    "NEWS": (FinnHubNewsSchema, finnhub_news_mapping, ["news_id"], "symbol"),
    "SOCIALS": (SocialsSchema, socials_mapping, ["symbol", "url", "title"], "symbol"),
    "FILINGS": (FinnhubFilingsSchema, finnhub_filings_mapping, ["symbol", "timestamp"], "symbol"),
}

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
PIPELINE = os.getenv("PIPELINE")


if __name__ == "__main__":
    if not PIPELINE:
        logger.error("PIPELINE environment variable not set")
        logger.info(f"Available pipelines: {list(PIPELINES.keys())}")
        sys.exit(1)
    
    if PIPELINE not in PIPELINES:
        logger.error(f"Invalid PIPELINE: {PIPELINE}")
        logger.info(f"Available pipelines: {list(PIPELINES.keys())}")
        sys.exit(1)

    logger.info(f"Pipeline: {PIPELINE}")
    logger.info(f"Mode: {'DEBUG' if DEBUG else 'PRODUCTION'}")
    
    schema, mapping, dedupe, key = PIPELINES[PIPELINE]
    
    create_silver_pipeline(
        name=PIPELINE,
        output_schema=schema,
        field_mapping=mapping,
        dedupe_columns=dedupe,
        key_column=key,
        debug=DEBUG,
    )
    
    run()