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

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("librdkafka").setLevel(logging.CRITICAL)
logging.getLogger("confluent_kafka").setLevel(logging.CRITICAL)

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
SILVER_TOPIC = os.getenv("REDPANDA_SILVER_NEWS_TOPIC")
GOLD_TOPIC = os.getenv("REDPANDA_GOLD_NEWS_TOPIC", "gold.news.sentiment")

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
        if t and c:
            return f"{t}. {c}"
        return t or c or ""
    except Exception:
        return ""


def _get_sentiment_score(title, content) -> float:
    try:
        from src.layers.gold_layer.finbert import get_sentiment
        t = _clean_json_string(title)
        c = _clean_json_string(content)
        if not t and not c:
            return 0.0
        result = get_sentiment(t, c)
        if result is None:
            return 0.0
        return float(result)
    except ImportError:
        return 0.0
    except ValueError:
        return 0.0
    except TypeError:
        return 0.0
    except AttributeError:
        return 0.0
    except RuntimeError:
        return 0.0
    except Exception:
        return 0.0


@pw.udf
def merge_texts(title: str, content: str) -> str:
    return _merge_texts(title, content)


@pw.udf
def get_sentiment_score(title: str, content: str) -> float:
    return _get_sentiment_score(title, content)


def validate_env():
    missing = []
    if not SILVER_TOPIC:
        missing.append("REDPANDA_SILVER_NEWS_TOPIC")
    if not os.getenv("PATHWAY_LICENSE_KEY"):
        missing.append("PATHWAY_LICENSE_KEY")
    if not os.getenv("REDPANDA_BROKERS"):
        missing.append("REDPANDA_BROKERS")
    if missing:
        print(f"[GOLD:NEWS] ERROR: Missing env vars: {missing}", flush=True)
        sys.exit(1)


def validate_broker(timeout: float = 10.0) -> bool:
    try:
        from confluent_kafka.admin import AdminClient
        config = common_config | {"socket.timeout.ms": "5000", "log_level": "2"}
        admin = AdminClient(config)
        metadata = admin.list_topics(timeout=timeout)
        return metadata is not None
    except Exception:
        return False


def wait_for_broker():
    attempt = 0
    while True:
        attempt += 1
        if validate_broker():
            print("[GOLD:NEWS] Broker connected", flush=True)
            return
        wait_time = min(30, 5 * attempt)
        print(f"[GOLD:NEWS] Waiting for broker... (attempt {attempt}, next in {wait_time}s)", flush=True)
        time.sleep(wait_time)


def create_pipeline():
    suffix = f"-{int(time.time())}" if DEBUG else ""
    
    consumer = common_config | KAFKA_RESILIENCE | {
        "group.id": f"finbert-sentiment-news{suffix}",
        "auto.offset.reset": "earliest",
    }
    
    producer = common_config | KAFKA_RESILIENCE | profiles["low_latency"]

    print(f"[GOLD:NEWS] Mode: {'DEBUG' if DEBUG else 'PROD'}", flush=True)
    print(f"[GOLD:NEWS] Input: {SILVER_TOPIC}", flush=True)
    print(f"[GOLD:NEWS] Output: {GOLD_TOPIC}", flush=True)
    print(f"[GOLD:NEWS] Consumer: {consumer['group.id']}", flush=True)

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

    final = enriched.groupby(pw.this.symbol).reduce(
        symbol=pw.this.symbol,
        news_articles=pw.reducers.tuple(pw.this.merged),
        news_sentiment_scores=pw.reducers.tuple(pw.this.sentiment),
    )

    if DEBUG:
        os.makedirs("debug_output/gold", exist_ok=True)
        pw.io.jsonlines.write(final, "debug_output/gold/news_sentiment.jsonl")
    else:
        pw.io.kafka.write(
            final,
            rdkafka_settings=producer,
            topic_name=GOLD_TOPIC,
            key=pw.this.symbol,
            format="json",
        )

    print("[GOLD:NEWS] Pipeline ready", flush=True)


def run_pipeline() -> str:
    try:
        pw.run()
        return "success"
    except KeyboardInterrupt:
        return "interrupted"
    except SystemExit:
        return "exit"
    except Exception as e:
        error_str = str(e).lower()
        if any(kw in error_str for kw in ["resolve", "connection", "timeout", "kafka", "broker", "network", "host"]):
            print(f"[GOLD:NEWS] Connection error: {e}", flush=True)
            return "connection"
        print(f"[GOLD:NEWS] Runtime error: {e}", flush=True)
        return "error"


def main():
    print("[GOLD:NEWS] Starting...", flush=True)
    
    validate_env()
    
    try:
        pw.set_license_key(os.getenv("PATHWAY_LICENSE_KEY"))
    except Exception as e:
        print(f"[GOLD:NEWS] License error: {e}", flush=True)
        sys.exit(1)

    while True:
        wait_for_broker()
        
        try:
            create_pipeline()
        except Exception as e:
            print(f"[GOLD:NEWS] Pipeline creation failed: {e}", flush=True)
            time.sleep(10)
            continue
        
        result = run_pipeline()
        
        if result == "interrupted" or result == "exit":
            break
        
        if result == "connection":
            print("[GOLD:NEWS] Connection lost. Reconnecting...", flush=True)
            time.sleep(5)
            continue
        
        if result == "error":
            print("[GOLD:NEWS] Restarting in 10s...", flush=True)
            time.sleep(10)
            continue
        
        if result == "success":
            break

    print("[GOLD:NEWS] Shutdown complete", flush=True)


if __name__ == "__main__":
    main()