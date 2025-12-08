import pathway as pw
import os
from dotenv import load_dotenv, find_dotenv
from src.schemas.silver_schemas import FinnHubEquitySchema
from src.utils.reducers import stddev

load_dotenv(find_dotenv)
pw.set_license_key(os.getenv("PATHWAY_LICENSE_KEY"))

consumer_settings = {
    "bootstrap.servers": os.getenv("REDPANDA_BROKERS"),
    "security.protocol": "SASL_SSL",
    "sasl.mechanism": "SCRAM-SHA-256",
    "sasl.username": os.getenv("REDPANDA_USERNAME"),
    "sasl.password": os.getenv("REDPANDA_PASSWORD"),
    "group.id": "financial-analyst-pro2",
    "auto.offset.reset": "earliest",
}

SILVER_TOPIC = os.getenv("REDPANDA_SILVER_STOCKS_TOPIC")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 1. DATA INGESTION
# ==============================================================================

stocks = pw.io.redpanda.read(
    rdkafka_settings=consumer_settings,
    topic=SILVER_TOPIC,
    schema=FinnHubEquitySchema,
    format="json",
    autocommit_duration_ms=1000
)

# ==============================================================================
# 2. CONVERT TIMESTAMP TO DATETIME
# ==============================================================================

stocks = stocks.select(
    pw.this.symbol,
    pw.this.price,
    pw.this.volume,
    timestamp_raw=pw.this.timestamp,
    timestamp=pw.this.timestamp.dt.from_timestamp(unit="ms")
)

# ==============================================================================
# 3. WINDOW DEFINITIONS
# ==============================================================================

SHORT_WINDOW = pw.temporal.sliding(
    hop=pw.Duration("1m"),
    duration=pw.Duration("5m")
)

MEDIUM_WINDOW = pw.temporal.sliding(
    hop=pw.Duration("1m"),
    duration=pw.Duration("20m")
)

LONG_WINDOW = pw.temporal.sliding(
    hop=pw.Duration("1m"),
    duration=pw.Duration("50m")
)

VWAP_WINDOW = pw.temporal.tumbling(
    duration=pw.Duration("1h")
)

# ==============================================================================
# 4. BASIC INDICATORS - MOVING AVERAGES
# ==============================================================================

sma_short = stocks.windowby(
    stocks.timestamp,
    window=SHORT_WINDOW,
    instance=stocks.symbol
).reduce(
    symbol=pw.reducers.any(pw.this.symbol),
    sma_5=pw.reducers.avg(pw.this.price),
    window_end=pw.reducers.max(pw.this.timestamp),
    data_points=pw.reducers.count()
)

sma_medium = stocks.windowby(
    stocks.timestamp,
    window=MEDIUM_WINDOW,
    instance=stocks.symbol
).reduce(
    symbol=pw.reducers.any(pw.this.symbol),
    sma_20=pw.reducers.avg(pw.this.price),
    window_end=pw.reducers.max(pw.this.timestamp)
)

sma_long = stocks.windowby(
    stocks.timestamp,
    window=LONG_WINDOW,
    instance=stocks.symbol
).reduce(
    symbol=pw.reducers.any(pw.this.symbol),
    sma_50=pw.reducers.avg(pw.this.price),
    window_end=pw.reducers.max(pw.this.timestamp)
)

# ==============================================================================
# 5. PRICE STATISTICS & VOLATILITY
# ==============================================================================

price_stats = stocks.windowby(
    stocks.timestamp,
    window=MEDIUM_WINDOW,
    instance=stocks.symbol
).reduce(
    symbol=pw.reducers.any(pw.this.symbol),
    window_end=pw.reducers.max(pw.this.timestamp),
    current_price=pw.reducers.latest(pw.this.price),
    open_price=pw.reducers.earliest(pw.this.price),
    high_price=pw.reducers.max(pw.this.price),
    low_price=pw.reducers.min(pw.this.price),
    price_stddev=stddev(pw.this.price),
    total_volume=pw.reducers.sum(pw.this.volume),
    avg_volume=pw.reducers.avg(pw.this.volume),
)

price_stats_enriched = price_stats.select(
    symbol=pw.this.symbol,
    window_end=pw.this.window_end,
    current_price=pw.this.current_price,
    open_price=pw.this.open_price,
    high_price=pw.this.high_price,
    low_price=pw.this.low_price,
    price_range=pw.this.high_price - pw.this.low_price,
    price_change=pw.this.current_price - pw.this.open_price,
    price_change_pct=((pw.this.current_price - pw.this.open_price) / pw.this.open_price) * 100,
    volatility=pw.this.price_stddev,
    volatility_pct=(pw.this.price_stddev / pw.this.current_price) * 100,
    total_volume=pw.this.total_volume,
    avg_volume=pw.this.avg_volume,
)

# ==============================================================================
# 6. VWAP
# ==============================================================================

vwap_calc = stocks.windowby(
    stocks.timestamp,
    window=VWAP_WINDOW,
    instance=stocks.symbol
).reduce(
    symbol=pw.reducers.any(pw.this.symbol),
    window_end=pw.reducers.max(pw.this.timestamp),
    price_volume_sum=pw.reducers.sum(pw.this.price * pw.this.volume),
    volume_sum=pw.reducers.sum(pw.this.volume),
    current_price=pw.reducers.latest(pw.this.price),
)

vwap = vwap_calc.select(
    symbol=pw.this.symbol,
    window_end=pw.this.window_end,
    current_price=pw.this.current_price,
    vwap=pw.this.price_volume_sum / (pw.this.volume_sum + 0.0001),
    total_volume=pw.this.volume_sum,
).select(
    symbol=pw.this.symbol,
    window_end=pw.this.window_end,
    current_price=pw.this.current_price,
    vwap=pw.this.vwap,
    total_volume=pw.this.total_volume,
    vwap_deviation=pw.this.current_price - pw.this.vwap,
    vwap_deviation_pct=((pw.this.current_price - pw.this.vwap) / (pw.this.vwap + 0.0001)) * 100,
)

# ==============================================================================
# 7. MOMENTUM INDICATORS
# ==============================================================================

momentum = stocks.windowby(
    stocks.timestamp,
    window=MEDIUM_WINDOW,
    instance=stocks.symbol
).reduce(
    symbol=pw.reducers.any(pw.this.symbol),
    window_end=pw.reducers.max(pw.this.timestamp),
    current_price=pw.reducers.latest(pw.this.price),
    oldest_price=pw.reducers.earliest(pw.this.price),
    high_price=pw.reducers.max(pw.this.price),
    low_price=pw.reducers.min(pw.this.price),
)

momentum_indicators = momentum.select(
    symbol=pw.this.symbol,
    window_end=pw.this.window_end,
    current_price=pw.this.current_price,
    roc=((pw.this.current_price - pw.this.oldest_price) / (pw.this.oldest_price + 0.0001)) * 100,
    williams_r=((pw.this.high_price - pw.this.current_price) / 
                (pw.this.high_price - pw.this.low_price + 0.0001)) * -100,
    price_position=((pw.this.current_price - pw.this.low_price) / 
                    (pw.this.high_price - pw.this.low_price + 0.0001)) * 100,
)

# ==============================================================================
# 8. COMBINED TRADING SIGNALS
# ==============================================================================

indicators_combined = sma_short.join(
    sma_medium,
    pw.left.symbol == pw.right.symbol,
).select(
    symbol=pw.left.symbol,
    window_end=pw.left.window_end,
    sma_5=pw.left.sma_5,
    sma_20=pw.right.sma_20,
    data_points=pw.left.data_points,
)

indicators_combined = indicators_combined.join(
    sma_long,
    pw.left.symbol == pw.right.symbol,
).select(
    symbol=pw.left.symbol,
    window_end=pw.left.window_end,
    sma_5=pw.left.sma_5,
    sma_20=pw.left.sma_20,
    sma_50=pw.right.sma_50,
    data_points=pw.left.data_points,
)

indicators_combined = indicators_combined.join(
    price_stats_enriched,
    pw.left.symbol == pw.right.symbol,
).select(
    symbol=pw.left.symbol,
    window_end=pw.left.window_end,
    current_price=pw.right.current_price,
    price_change_pct=pw.right.price_change_pct,
    sma_5=pw.left.sma_5,
    sma_20=pw.left.sma_20,
    sma_50=pw.left.sma_50,
    volatility_pct=pw.right.volatility_pct,
    total_volume=pw.right.total_volume,
    high_price=pw.right.high_price,
    low_price=pw.right.low_price,
)

trading_signals = indicators_combined.select(
    pw.this.symbol,
    pw.this.window_end,
    pw.this.current_price,
    pw.this.sma_5,
    pw.this.sma_20,
    pw.this.sma_50,
    pw.this.price_change_pct,
    pw.this.volatility_pct,
    pw.this.total_volume,
    short_term_trend=pw.if_else(
        pw.this.current_price > pw.this.sma_5,
        "BULLISH",
        "BEARISH"
    ),
    medium_term_trend=pw.if_else(
        pw.this.sma_5 > pw.this.sma_20,
        "BULLISH",
        "BEARISH"
    ),
    long_term_trend=pw.if_else(
        pw.this.sma_20 > pw.this.sma_50,
        "BULLISH",
        "BEARISH"
    ),
    above_sma_5=pw.this.current_price > pw.this.sma_5,
    above_sma_20=pw.this.current_price > pw.this.sma_20,
    above_sma_50=pw.this.current_price > pw.this.sma_50,
)

# ==============================================================================
# 9. OUTPUT - JSON LINES (FASTER THAN CSV)
# ==============================================================================

pw.io.jsonlines.write(
    price_stats_enriched,
    os.path.join(OUTPUT_DIR, "price_statistics.jsonl")
)

pw.io.jsonlines.write(
    vwap,
    os.path.join(OUTPUT_DIR, "vwap_indicators.jsonl")
)

pw.io.jsonlines.write(
    momentum_indicators,
    os.path.join(OUTPUT_DIR, "momentum_indicators.jsonl")
)

pw.io.jsonlines.write(
    trading_signals,
    os.path.join(OUTPUT_DIR, "trading_signals.jsonl")
)

# ==============================================================================
# 10. RUN THE PIPELINE
# ==============================================================================

if __name__ == "__main__":
    print("Starting Technical Indicators Pipeline...")
    print(f"Reading from topic: {SILVER_TOPIC}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Output format: JSON Lines (.jsonl)")
    pw.run()