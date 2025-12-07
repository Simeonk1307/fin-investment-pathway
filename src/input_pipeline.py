import os
import sys
import time
import json
import signal
import logging
import pathway as pw
from dotenv import load_dotenv
from src.schemas.silver_schemas import FinnHubNewsSchema, SocialsSchema, FinnhubFilingsSchema, FinnHubStockSchema
from src.utils.common import common_config, profiles
from src.agents.finbert import FinBertSentimentAnalyzer
from src.utils.reducers import WeightedSentimentScoreAccumulator, SimpleSentimentScoreAccumulator
from src.utils.minio_storage import MinioStorage
from src.utils.filings_summarizer import FilingsSummarizer
from src.agents.llm_factory import get_llm
from src.utils.reducers import StockAccumulator

LLM = get_llm('perplexity')

def debug_statement(*args):
    import time
    logger.info(f"[DEBUG " + " ".join(str(a) for a in args))
    time.sleep(5)



DEBUG = os.getenv("DEBUG", "false").lower() == "true"
finbert_analyzer = FinBertSentimentAnalyzer()
load_dotenv()

_minio = MinioStorage()
_summarizer = FilingsSummarizer()


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("librdkafka").setLevel(logging.CRITICAL)
logging.getLogger("confluent_kafka").setLevel(logging.CRITICAL)

KAFKA_RESILIENCE = {
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

def shutdown_handler(signum, frame):
    print("[GOLD:NEWS] Shutdown signal received", flush=True)
    os._exit(0)


signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)



def _clean_json_string(text) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            return str(text)
        except Exception:
            return ""
    if not text:
        return ""
    try:
        original = text
        for _ in range(5):
            if not (text.startswith('"') and text.endswith('"')):
                break
            try:
                text = json.loads(text)
                if not isinstance(text, str):
                    return str(text) if text else ""
            except json.JSONDecodeError:
                break
        if isinstance(text, str):
            text = text.replace("\\u2018", "'").replace("\\u2019", "'")
            text = text.replace("\\u201c", '"').replace("\\u201d", '"')
            text = text.replace("\\n", " ").replace("\\t", " ")
            text = text.replace("\n", " ").replace("\t", " ")
            return text.strip()
        return str(text)
    except Exception:
        try:
            return str(text).strip().strip('"')
        except Exception:
            return ""


def _merge_texts(title, content) -> str:
    try:
        t = _clean_json_string(title)
        c = _clean_json_string(content)
        # logger.info(f"Merging texts. Title: {t}, Content: {c}")
        if t and c:
            return f"{t}. {c}"
        return t or c or ""
    except Exception:
        return ""

def _get_sentiment_score(title, content) -> list[float]:
    try:
        t = _clean_json_string(title)
        c = _clean_json_string(content)
        # logger.info(f"Getting sentiment score. Title: {t}, Content: {c}")
        if not t and not c:
            return [0.0, 0.0, 0.0]
        result = finbert_analyzer.analyze_sentiment(f"{t}\n{c}")
        if result is None:
            return [0.0, 0.0, 0.0]
        return result
    except Exception as e:
        logger.error(f"Unexpected error encountered while getting sentiment score: {e}")
        return [0.0, 0.0, 0.0]


@pw.udf
def merge_texts(title: str, content: str) -> str:
    return _merge_texts(title, content)


@pw.udf
def get_sentiment_score(title: str, content: str) -> list[float]:
    return _get_sentiment_score(title, content)

@pw.udf
def get_weight_timestamp(timestamp: int) -> float:
    """
    Calculate weight for stock news with appropriate decay.
    For stocks, use 10-day half-life (news from 10 days ago = 50% weight)
    Args:
        timestamp: Unix timestamp (e.g., 1734453535)
    Returns:
        Weight between 0.0001 and 1.0
    """
    
    current_time = time.time()
    age_seconds = current_time - timestamp
    age_days = age_seconds / 86400  # Convert seconds to days
    # logger.info(f"Calculating weight for timestamp {timestamp}, age in days: {age_days}")
    # Day 0 (today): weight = 1.0
    # Day 10: weight = 0.5
    # Day 30: weight = 0.125

    half_life_days = 10.0
    weight = 2 ** (-age_days / half_life_days)
    # logger.info(f"Calculated weight: {weight} for age in days: {age_days}")
    # time.sleep(3)
    
    return max(weight, 0.0001)  # Minimum weight

def news_input_pipeline()->pw.Table:
    TEMP = "NEWS"
    SCHEMA = FinnHubNewsSchema
    SILVER_TOPIC = os.getenv(f"REDPANDA_SILVER_{TEMP}_TOPIC")
    

    suffix = f"-{int(time.time())}" if DEBUG else ""
    consumer = common_config | KAFKA_RESILIENCE | {
        "group.id": f"finbert-sentiment-{TEMP.lower()}{suffix}",
        "auto.offset.reset": "earliest",
    }

    if SILVER_TOPIC is None:
        logger.error(f"[GOLD:{TEMP}] REDPANDA_SILVER_{TEMP}_TOPIC environment variable not set.")
        sys.exit(1)

    logger.info(f"[GOLD:{TEMP}] Reading from {SILVER_TOPIC}")

    silver = pw.io.redpanda.read(
        rdkafka_settings=consumer,
        topic=SILVER_TOPIC,
        schema=SCHEMA,
        format="json",
        autocommit_duration_ms=1000,
    )
    enriched = silver.select(
        timestamp=pw.this.timestamp,
        symbol=pw.this.symbol,
        merged=merge_texts(pw.this.title, pw.this.content),
        sentiment=get_sentiment_score(pw.this.title, pw.this.content),
        weight=get_weight_timestamp(pw.this.timestamp)
    )

    logger.info(f"[GOLD:{TEMP}] {TEMP} Data enriched")

    final = enriched.groupby(pw.this.symbol).reduce(
        symbol=pw.this.symbol,
        news_articles=pw.reducers.tuple(pw.this.merged),
        news_sentiment_scores=pw.reducers.udf_reducer(WeightedSentimentScoreAccumulator)(pw.this.sentiment, pw.this.weight)
    )

    logger.info(f"[GOLD:{TEMP}] {TEMP} Data analysed")


    return final

def social_input_pipeline()->pw.Table:
    TEMP = "SOCIALS"
    SCHEMA = SocialsSchema
    SILVER_TOPIC = os.getenv(f"REDPANDA_SILVER_{TEMP}_TOPIC")
    

    suffix = f"-{int(time.time())}" if DEBUG else ""
    consumer = common_config | KAFKA_RESILIENCE | {
        "group.id": f"finbert-sentiment-{TEMP.lower()}{suffix}",
        "auto.offset.reset": "earliest",
    }
    

    if SILVER_TOPIC is None:
        logger.error(f"[GOLD:{TEMP}] REDPANDA_SILVER_{TEMP}_TOPIC environment variable not set.")
        sys.exit(1)

    logger.info(f"[GOLD:{TEMP}] Reading from {SILVER_TOPIC}")

    silver = pw.io.redpanda.read(
        rdkafka_settings=consumer,
        topic=SILVER_TOPIC,
        schema=SCHEMA,
        format="json",
        autocommit_duration_ms=1000,
    )
    enriched = silver.select(
        # timestamp=pw.this.timestamp,
        symbol=pw.this.symbol,
        merged=merge_texts(pw.this.title, pw.this.content),
        sentiment=get_sentiment_score(pw.this.title, pw.this.content),
        # weight=get_weight_timestamp(pw.this.timestamp)
    )

    logger.info(f"[GOLD:{TEMP}] {TEMP} Data enriched")

    final = enriched.groupby(pw.this.symbol).reduce(
        symbol=pw.this.symbol,
        socials_articles=pw.reducers.tuple(pw.this.merged),
        socials_sentiment_scores=pw.reducers.udf_reducer(SimpleSentimentScoreAccumulator)(pw.this.sentiment)
    )

    logger.info(f"[GOLD:{TEMP}] {TEMP} Data analysed")

    return final

@pw.udf
def summarize_filing(storage_url: str) -> str:
    """Fetch filing from MinIO and summarize to 20 points"""
    try:
        if not storage_url:
            return ""
        
        # Clean the URL
        url = storage_url.strip().strip('"').strip("'").replace('\\"', '')
        
        logger.info(f"Processing filing: {url}")
        
        # Fetch content
        content = _minio.read_filing(url)
        
        if not content or len(content.strip()) < 100:
            logger.warning(f"Empty or too short content for {url}")
            return ""
        
        logger.info(f"Content length: {len(content)} chars")
        
        # Summarize
        summary = _summarizer.summarize(content)
        
        if not summary:
            logger.warning(f"Empty summary for {url}")
            return ""
        
        logger.info(f"Summary generated: {len(summary)}")
        return summary
        
    except Exception as e:
        logger.error(f"Error processing filing {storage_url}: {e}", exc_info=True)
        return ""

def filings_input_pipeline() -> pw.Table:
    SILVER_TOPIC = os.getenv("REDPANDA_SILVER_FILINGS_TOPIC")
    
    consumer = common_config | KAFKA_RESILIENCE | {
        "group.id": f"gold-filings-{int(time.time())}" if DEBUG else "gold-filings",
        "auto.offset.reset": "earliest",
    }
    
    silver = pw.io.redpanda.read(
        rdkafka_settings=consumer,
        topic=SILVER_TOPIC,
        schema=FinnhubFilingsSchema,
        format="json",
        autocommit_duration_ms=1000,
    )
    
    # Group and get sorted URLs
    grouped = silver.groupby(pw.this.symbol).reduce(
        symbol=pw.this.symbol,
        all_urls=pw.reducers.sorted_tuple(pw.this.storage_url)
    )
    
    # Take last 5
    top5 = grouped.select(
        symbol=pw.this.symbol,
        top5_urls=pw.apply(lambda urls: urls[-5:], pw.this.all_urls)
    )
    
    # Flatten the tuple back into rows
    flattened = top5.flatten(pw.this.top5_urls).select(
        symbol=pw.this.symbol,
        storage_url=pw.this.top5_urls
    )
    
    # Now summarize each row
    with_summary = flattened.select(
        symbol=pw.this.symbol,
        filing_summary=summarize_filing(pw.this.storage_url)
        # filing_summary=pw.this.symbol
    )
    # pw.io.csv.write(with_summary,'debug_output/filings_summary.csv')
    # @pw.udf
    # def debug(**args):
    #     logger.info(args)
    #     return args
    # # Group back
    # final = with_summary.groupby(pw.this.symbol).reduce(
    #     symbol=pw.this.symbol,
    #     debug = debug(pw.this.symbol),
    #     filing_summaries=pw.reducers.tuple(pw.this.filing_summary)
    # )
    
    return with_summary

# src/input_pipeline.py

def stock_analysis_pipeline() -> pw.Table:
    SILVER_TOPIC = os.getenv("REDPANDA_SILVER_STOCK_TOPIC")
    
    consumer = common_config | KAFKA_RESILIENCE | {
        "group.id": f"stock-{int(time.time())}" if DEBUG else "stock",
        "auto.offset.reset": "earliest",
    }
    
    silver = pw.io.redpanda.read(
        rdkafka_settings=consumer,
        topic=SILVER_TOPIC,
        schema=FinnHubStockSchema,
        format="json",
        autocommit_duration_ms=1000,
    )
    
    # 5min window
    w5 = silver.windowby(
        pw.this.timestamp,
        window=pw.temporal.sliding(hop=pw.Duration("5m"), duration=pw.Duration("5m")),
        instance=pw.this.symbol
    ).reduce(
        symbol=pw.this.symbol,
        time=pw.this._pw_window_end,
        m5=pw.reducers.udf_reducer(StockAccumulator)(pw.this.price, pw.this.volume)
    )
    
    # 15min window
    w15 = silver.windowby(
        pw.this.timestamp,
        window=pw.temporal.sliding(hop=pw.Duration("15m"), duration=pw.Duration("15m")),
        instance=pw.this.symbol
    ).reduce(
        symbol=pw.this.symbol,
        time=pw.this._pw_window_end,
        m15=pw.reducers.udf_reducer(StockAccumulator)(pw.this.price, pw.this.volume)
    )
    
    # Join
    final = w5.join(w15, pw.left.symbol == pw.right.symbol, pw.left.time == pw.right.time).select(
        symbol=pw.left.symbol,
        timestamp=pw.left.time,
        analysis_5min=pw.left.m5,
        analysis_15min=pw.right.m15
    )
    
    return final




# def input_pipeline() -> list[pw.Table]: 
#     return [news_input_pipeline(), social_input_pipeline()]


if __name__ == "__main__":
    output_path = "debug_output/inputs"
    os.makedirs(output_path, exist_ok=True)
    news_table= news_input_pipeline()
    socials_table = social_input_pipeline()
    filings_table = filings_input_pipeline()

    pw.io.csv.write(news_table, f"{output_path}/news_sentiment.csv")
    pw.io.csv.write(socials_table, f"{output_path}/socials_sentiment.csv")
    pw.io.csv.write(filings_table, f"{output_path}/filings_sentiment.csv")
    # pw.io.jsonlines.write(news_table, "debug_output/news_sentiment.jsonl")
    pw.run()