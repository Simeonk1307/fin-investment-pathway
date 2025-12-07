import os
import sys
import time
import json
import signal
import logging
import pathway as pw
from dotenv import load_dotenv
from src.schemas.silver_schemas import FinnHubNewsSchema, SocialsSchema, FinnhubFilingsSchema
from src.utils.common import common_config, profiles
from src.agents.finbert import FinBertSentimentAnalyzer
from src.utils.reducers import WeightedSentimentScoreAccumulator, SimpleSentimentScoreAccumulator
from src.utils.minio_storage import MinioStorage
from src.agents.llm_factory import get_llm
LLM = get_llm('perplexity')

def debug_statement(*args):
    import time
    logger.info(f"[DEBUG " + " ".join(str(a) for a in args))
    time.sleep(5)



DEBUG = os.getenv("DEBUG", "false").lower() == "true"
finbert_analyzer = FinBertSentimentAnalyzer()
load_dotenv()

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

_minio = None

def get_minio():
    global _minio
    if _minio is None:
        _minio = MinioStorage()
    return _minio


@pw.udf
def udf_extract_filing(path: str) -> str:
    return get_minio().read_filing_extract(path, max_total=2000)


def filings_input_pipeline() -> pw.Table:
    TEMP = "FILINGS"
    SILVER_TOPIC = os.getenv(f"REDPANDA_SILVER_{TEMP}_TOPIC")

    if not SILVER_TOPIC:
        logger.error(f"[GOLD:{TEMP}] Topic not set")
        sys.exit(1)

    suffix = f"-{int(time.time())}" if DEBUG else ""
    consumer = common_config | KAFKA_RESILIENCE | {
        "group.id": f"gold-filings{suffix}",
        "auto.offset.reset": "earliest",
    }

    logger.info(f"[GOLD:{TEMP}] Reading from {SILVER_TOPIC}")

    silver = pw.io.redpanda.read(
        rdkafka_settings=consumer,
        topic=SILVER_TOPIC,
        schema=FinnhubFilingsSchema,
        format="json",
        autocommit_duration_ms=1000,
    )

    enriched = silver.select(
        symbol=pw.this.symbol,
        filings_summary=udf_extract_filing(pw.this.storage_url),
    ).groupby(pw.this.symbol).reduce(
        symbol=pw.this.symbol,
        filings_summary=pw.reducers.tuple(pw.this.filings_summary),
    )

    return enriched




# def input_pipeline() -> list[pw.Table]: 
#     return [news_input_pipeline(), social_input_pipeline()]


if __name__ == "__main__":
    os.makedirs("debug_output", exist_ok=True)
    # news_table= news_input_pipeline()
    # socials_table = social_input_pipeline()
    filings_table = filings_input_pipeline()

    # pw.io.csv.write(news_table, "debug_output/news_sentiment.csv")
    # pw.io.csv.write(socials_table, "debug_output/socials_sentiment.csv")
    pw.io.csv.write(filings_table, "debug_output/socials_sentiment.csv")
    # pw.io.jsonlines.write(news_table, "debug_output/news_sentiment.jsonl")
    pw.run()