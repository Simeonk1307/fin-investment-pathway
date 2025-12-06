import os
import sys
import time
import json
import signal
import logging
import pathway as pw
from dotenv import load_dotenv
from src.schemas.silver_schemas import FinnHubNewsSchema
from src.utils.common import common_config, profiles
from src.agents.finbert import FinBertSentimentAnalyzer
def debug_statement(*args):
    import time
    logger.info(f"[DEBUG " + " ".join(str(a) for a in args))
    time.sleep(5)



DEBUG = os.getenv("DEBUG", "false").lower() == "true"
SILVER_TOPIC = os.getenv("REDPANDA_SILVER_NEWS_TOPIC")
GOLD_TOPIC = os.getenv("REDPANDA_GOLD_NEWS_TOPIC", "gold.news.sentiment")
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


def news_input_pipeline()->pw.Table:
    suffix = f"-{int(time.time())}" if DEBUG else ""
    consumer = common_config | KAFKA_RESILIENCE | {
        "group.id": f"finbert-sentiment-news{suffix}",
        "auto.offset.reset": "earliest",
    }
    

    if SILVER_TOPIC is None:
        logger.error("[GOLD:NEWS] REDPANDA_SILVER_NEWS_TOPIC environment variable not set.")
        sys.exit(1)

    logger.info(f"[GOLD:NEWS] Reading from {SILVER_TOPIC}")

    news = pw.io.redpanda.read(
        rdkafka_settings=consumer,
        topic=SILVER_TOPIC,
        schema=FinnHubNewsSchema,
        format="json",
        autocommit_duration_ms=1000,
    )
    enriched = news.select(
        symbol=pw.this.symbol,
        merged=merge_texts(pw.this.title, pw.this.content),
        sentiment=get_sentiment_score(pw.this.title, pw.this.content),
    )

    news_table = enriched.groupby(pw.this.symbol).reduce(
        symbol=pw.this.symbol,
        news_articles=pw.reducers.tuple(pw.this.merged),
        news_sentiment_scores=pw.reducers.tuple(pw.this.sentiment),
    )


    return news_table

def social_input_pipeline()->pw.Table:
    suffix = f"-{int(time.time())}" if DEBUG else ""
    consumer = common_config | KAFKA_RESILIENCE | {
        "group.id": f"finbert-sentiment-socials{suffix}",
        "auto.offset.reset": "earliest",
    }
    

    if SILVER_TOPIC is None:
        logger.error("[GOLD:NEWS] REDPANDA_SILVER_NEWS_TOPIC environment variable not set.")
        sys.exit(1)

    logger.info(f"[GOLD:NEWS] Reading from {SILVER_TOPIC}")

    news = pw.io.redpanda.read(
        rdkafka_settings=consumer,
        topic=SILVER_TOPIC,
        schema=FinnHubNewsSchema,
        format="json",
        autocommit_duration_ms=1000,
    )

    enriched = news.select(
        symbol=pw.this.symbol,
        merged=merge_texts(pw.this.title, pw.this.content),
        sentiment=get_sentiment_score(pw.this.title, pw.this.content),
    )

    news_table = enriched.groupby(pw.this.symbol).reduce(
        symbol=pw.this.symbol,
        news_articles=pw.reducers.tuple(pw.this.merged),
        news_sentiment_scores=pw.reducers.tuple(pw.this.sentiment),
    )


    return news_table



# def input_pipeline() -> list[pw.Table]: 
#     return [news_input_pipeline(), social_input_pipeline()]


if __name__ == "__main__":
    os.makedirs("debug_output", exist_ok=True)
    news_table= news_input_pipeline()
    pw.io.csv.write(news_table, "debug_output/news_sentiment.csv")
    # pw.io.jsonlines.write(news_table, "debug_output/news_sentiment.jsonl")
    pw.run()