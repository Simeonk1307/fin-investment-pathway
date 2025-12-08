import os
import sys
import time
import json
import signal
import logging
import pathway as pw
from dotenv import load_dotenv
from src.schemas.silver_schemas import FinnhubFilingsSchema
from src.utils.common import common_config, profiles

# --- SETUP & LOGGING ---
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger(__name__)

# Reduce noise from Kafka libs
logging.getLogger("librdkafka").setLevel(logging.CRITICAL)
logging.getLogger("confluent_kafka").setLevel(logging.CRITICAL)

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Topics
SILVER_TOPIC = os.getenv("REDPANDA_SILVER_FILINGS_TOPIC", "silver.filings")
GOLD_TOPIC = os.getenv("REDPANDA_GOLD_FILINGS_TOPIC", "gold.filings.sentiment")

# Kafka Settings
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

# --- SHUTDOWN HANDLER ---
def shutdown_handler(signum, frame):
    print("[GOLD:FILINGS] Shutdown signal received", flush=True)
    os._exit(0)

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

# --- UDFs & HELPERS ---

def _clean_json_string(text) -> str:
    """Helper to clean up potentially messy text from JSON"""
    if not text: return ""
    try:
        if isinstance(text, str):
            text = text.replace("\\n", " ").replace("\\t", " ")
            text = text.replace("\n", " ").replace("\t", " ")
            # Limit length for FinBERT (it truncates anyway, but saves processing)
            return text[:2000].strip()
        return str(text)[:2000]
    except Exception:
        return ""

def _get_sentiment_score(headline, content) -> float:
    """Runs FinBERT on the filing headline + content summary"""
    try:
        # Import inside function to avoid heavy load if not used
        from src.agents.finbert import FinBertSentimentAnalyzer
        
        # Initialize (singleton pattern handled by class usually, or re-init per batch)
        # Note: In production, better to init once globally or use a dedicated service.
        # For Pathway UDF, we'll instantiate here.
        analyzer = FinBertSentimentAnalyzer()
        
        text = f"{headline}. {content}"
        # Analyze returns (neg, neu, pos) -> we want a single float score (-1 to 1)
        # Assuming your finbert.py returns a tuple of probabilities
        probs = analyzer.analyze_sentiment(text[:512]) # FinBERT max length
        
        # Convert probabilities (neg, neu, pos) to a single scalar score
        # Score = P(pos) - P(neg)
        # probs[0]=neg, probs[1]=neu, probs[2]=pos
        sentiment_score = float(probs[2] - probs[0])
        
        return sentiment_score
        
    except ImportError:
        logger.error("FinBERT module missing")
        return 0.0
    except Exception as e:
        # logger.error(f"Sentiment error: {e}")
        return 0.0

@pw.udf
def get_filing_sentiment(headline: str, content: str) -> float:
    # Wrap helper in UDF
    return _get_sentiment_score(headline, content)

# --- PIPELINE LOGIC ---

def validate_env():
    missing = []
    if not SILVER_TOPIC: missing.append("REDPANDA_SILVER_FILINGS_TOPIC")
    if not os.getenv("PATHWAY_LICENSE_KEY"): missing.append("PATHWAY_LICENSE_KEY")
    if not os.getenv("REDPANDA_BROKERS"): missing.append("REDPANDA_BROKERS")
    
    if missing:
        print(f"[GOLD:FILINGS] ERROR: Missing env vars: {missing}", flush=True)
        sys.exit(1)

def wait_for_broker():
    # Simple wait logic
    attempt = 0
    while attempt < 5:
        try:
            from confluent_kafka.admin import AdminClient
            config = common_config | {"socket.timeout.ms": "5000"}
            admin = AdminClient(config)
            if admin.list_topics(timeout=5):
                print("[GOLD:FILINGS] Broker connected", flush=True)
                return
        except:
            pass
        attempt += 1
        print(f"[GOLD:FILINGS] Waiting for broker... ({attempt})", flush=True)
        time.sleep(5)

def create_pipeline():
    suffix = f"-{int(time.time())}" if DEBUG else ""
    
    consumer = common_config | KAFKA_RESILIENCE | {
        "group.id": f"finbert-sentiment-filings{suffix}",
        "auto.offset.reset": "earliest",
    }
    
    producer = common_config | KAFKA_RESILIENCE | profiles["low_latency"]

    print(f"[GOLD:FILINGS] Input: {SILVER_TOPIC}", flush=True)
    print(f"[GOLD:FILINGS] Output: {GOLD_TOPIC}", flush=True)

    # 1. READ SILVER DATA
    filings = pw.io.redpanda.read(
        rdkafka_settings=consumer,
        topic=SILVER_TOPIC,
        schema=FinnhubFilingsSchema,
        format="json",
        autocommit_duration_ms=1000,
    )

    # 2. ENRICH WITH SENTIMENT
    # We calculate sentiment on the headline + scraped content
    enriched = filings.select(
        ticker=pw.this.ticker,
        company=pw.this.company,
        form_type=pw.this.form_type,
        date=pw.this.date,
        link=pw.this.link,
        
        # Combine text for the Agent later
        combined_text=pw.apply(lambda h, c: f"{h}: {c}", pw.this.headline, pw.this.content),
        
        # Calculate FinBERT Score (-1 to 1)
        sentiment_score=get_filing_sentiment(pw.this.headline, pw.this.content),
    )

    # 3. AGGREGATE (Optional but good for Agent batches)
    # We group by ticker to provide the Agent with "Recent Filings context"
    # For filings, we often want instant alerts, but grouping helps if multiple 8-Ks drop at once.
    final_output = enriched.groupby(pw.this.ticker).reduce(
        ticker=pw.this.ticker,
        company=pw.reducers.earliest(pw.this.company),
        
        # List of recent filings
        recent_filings=pw.reducers.tuple(pw.this.combined_text),
        
        # List of scores
        filing_sentiment_scores=pw.reducers.tuple(pw.this.sentiment_score),
        
        # Metadata
        last_filing_date=pw.reducers.max(pw.this.date)
    )

    # 4. OUTPUT
    if DEBUG:
        os.makedirs("debug_output/gold", exist_ok=True)
        pw.io.jsonlines.write(final_output, "debug_output/gold/filings_sentiment.jsonl")
    else:
        pw.io.kafka.write(
            final_output,
            rdkafka_settings=producer,
            topic_name=GOLD_TOPIC,
            key=pw.this.ticker,
            format="json",
        )

    print("[GOLD:FILINGS] Pipeline built successfully", flush=True)

def main():
    print("[GOLD:FILINGS] Starting...", flush=True)
    validate_env()
    
    try:
        pw.set_license_key(os.getenv("PATHWAY_LICENSE_KEY"))
    except Exception as e:
        print(f"[GOLD:FILINGS] License error: {e}", flush=True)
        sys.exit(1)

    wait_for_broker()
    create_pipeline()
    
    try:
        pw.run()
    except KeyboardInterrupt:
        print("[GOLD:FILINGS] Stopping...", flush=True)
    except Exception as e:
        print(f"[GOLD:FILINGS] Runtime error: {e}", flush=True)

if __name__ == "__main__":
    main()

