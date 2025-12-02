import pathway as pw
import os
from dotenv import load_dotenv
from src.schemas.silver.stocks_schema import YFinanceEquitySchema
from src.utils.common import common_config, profiles
from src.utils.reducers import stddev, range_calc

OUTPUT_DIR = "outputs"

# Create directory if missing
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

load_dotenv()
pw.set_license_key(os.getenv("PATHWAY_LICENSE_KEY"))

consumer_settings = common_config | {
    "group.id": "silver-stocks-consumer2",
    # "auto.offset.reset": "latest",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": "true",
    "auto.commit.interval.ms": "500",
}

# TODO FOR FAZIL: CHANGE ALL DOTENV KEYS TO SETTINGS

SILVER_TOPIC = os.getenv("REDPANDA_SILVER_STOCKS_TOPIC")
# GOLD_TOPIC = os.getenv("REDPANDA_GOLD_STOCKS_TOPIC")

stocks = pw.io.redpanda.read(
    rdkafka_settings=consumer_settings,
    topic=os.getenv("REDPANDA_SILVER_STOCKS_TOPIC"),
    schema=YFinanceEquitySchema,
    format="json",
    autocommit_duration_ms=1000
)

stocks = stocks.with_columns(
    timestamp=pw.declare_type(
        pw.DateTimeNaive,
        pw.apply_with_type(
            lambda ms: pw.DateTimeNaive.fromtimestamp(ms / 1000.0),
            pw.DateTimeNaive,
            pw.this.time
        )
    )
)

current_stats = stocks.groupby(pw.this.ticker).reduce(
    ticker=pw.this.ticker,
    latest_update_time=pw.reducers.latest(pw.this.time),
    latest_price=pw.reducers.latest(pw.this.price),
    latest_change_percent=pw.reducers.latest(pw.this.change_percent),
    
    max_price=pw.reducers.max(pw.this.price),
    min_price=pw.reducers.min(pw.this.price),
    avg_price=pw.reducers.avg(pw.this.price),
    std_price=stddev(pw.this.price),
    range_price=range_calc(pw.this.price),

    total_volume=pw.reducers.sum(pw.this.day_volume), 
)

pw.io.csv.write(
    current_stats, 
    os.path.join(OUTPUT_DIR, "current_stock_stats.csv")
)

# =====================================================
# 4. TIME WINDOWS (5 MIN)
# =====================================================
stats_5min = stocks.windowby(
    pw.this.timestamp,
    window=pw.temporal.sliding(
        hop=pw.Duration("10s"),
        duration=pw.Duration("5m")
    ),
    behavior=pw.temporal.common_behavior()
).groupby(pw.this.ticker).reduce(
    ticker=pw.this.ticker,
    ma_5min=pw.reducers.avg(pw.this.price),
    high_5min=pw.reducers.max(pw.this.price),
    low_5min=pw.reducers.min(pw.this.price),
    volume_5min=pw.reducers.sum(pw.this.day_volume),
    volatility_5min=stddev(pw.this.price)
)

# =====================================================
# 5. TIME WINDOWS (15 MIN)
# =====================================================
stats_15min = stocks.windowby(
    pw.this.timestamp,
    window=pw.temporal.sliding(
        hop=pw.Duration("30s"),
        duration=pw.Duration("15m")
    ),
    behavior=pw.temporal.common_behavior()
).groupby(pw.this.ticker).reduce(
    ticker=pw.this.ticker,
    ma_15min=pw.reducers.avg(pw.this.price),
    high_15min=pw.reducers.max(pw.this.price),
    low_15min=pw.reducers.min(pw.this.price),
    volume_15min=pw.reducers.sum(pw.this.day_volume),
    volatility_15min=stddev(pw.this.price),
)

# =====================================================
# 6. JOIN TIMEFRAMES
# =====================================================

indicators = current_stats.join_left(
    stats_5min, pw.left.ticker == pw.right.ticker
).select(
    *pw.left,
    ma_5min=pw.coalesce(pw.right.ma_5min, pw.left.latest_price),
    high_5min=pw.coalesce(pw.right.high_5min, pw.left.latest_price),
    low_5min=pw.coalesce(pw.right.low_5min, pw.left.latest_price),
    volume_5min=pw.coalesce(pw.right.volume_5min, 0),
    volatility_5min=pw.coalesce(pw.right.volatility_5min, 0.0),
)

indicators = indicators.join_left(
    stats_15min, pw.left.ticker == pw.right.ticker
).select(
    *pw.left,
    ma_15min=pw.coalesce(pw.right.ma_15min, pw.left.latest_price),
    high_15min=pw.coalesce(pw.right.high_15min, pw.left.latest_price),
    low_15min=pw.coalesce(pw.right.low_15min, pw.left.latest_price),
    volume_15min=pw.coalesce(pw.right.volume_15min, 0),
    volatility_15min=pw.coalesce(pw.right.volatility_15min, 0.0),
)

# =====================================================
# 7. CALCULATE FINAL INDICATORS
# =====================================================
final = indicators.with_columns(
    # MACD
    macd=pw.this.ma_5min - pw.this.ma_15min,
    
    # Bollinger Bands
    bb_upper=pw.this.ma_5min + (2 * pw.this.volatility_5min),
    bb_lower=pw.this.ma_5min - (2 * pw.this.volatility_5min),
    
    # Price vs MAs
    price_vs_ma5=pw.apply(
        lambda p, ma: ((p - ma) / ma * 100) if ma > 0 else 0.0,
        pw.this.latest_price, pw.this.ma_5min
    ),
    
    # Volatility Coefficient
    volatility_coef=pw.apply(
        lambda std, avg: (std / avg * 100) if avg > 0 else 0.0,
        pw.this.std_price, pw.this.avg_price
    ),
)

# RSI Approximation
final = final.with_columns(
    rsi=pw.apply(
        lambda pos_pct, change: min(100.0, max(0.0, 50.0 + (pos_pct - 50.0) * 0.5 + change)),
        pw.apply(
            lambda p, h, l: ((p - l) / (h - l) * 100.0) if h != l else 50.0,
            pw.this.latest_price, pw.this.max_price, pw.this.min_price
        ),
        pw.this.latest_change_percent
    )
)

# Signals
final = final.with_columns(
    signal=pw.apply(
        lambda rsi, macd, price, bb_lower, bb_upper:
        "STRONG_BUY" if (rsi < 30 or price <= bb_lower) and macd > 0
        else "BUY" if rsi < 45
        else "STRONG_SELL" if (rsi > 70 or price >= bb_upper) and macd < 0
        else "SELL" if rsi > 55
        else "HOLD",
        pw.this.rsi, pw.this.macd, pw.this.latest_price,
        pw.this.bb_lower, pw.this.bb_upper
    ),
    
    risk_level=pw.apply(
        lambda vol_coef: 
        "HIGH" if vol_coef > 3.0
        else "MEDIUM" if vol_coef > 1.5
        else "LOW",
        pw.this.volatility_coef
    )
)

# =====================================================
# 8. OUTPUT
# =====================================================
output = final.select(
    ticker=pw.this.ticker,
    latest_update_time=pw.this.latest_update_time,
    latest_price=pw.this.latest_price,
    
    ma_5min=pw.this.ma_5min,
    ma_15min=pw.this.ma_15min,
    
    bb_upper=pw.this.bb_upper,
    bb_lower=pw.this.bb_lower,
    
    macd=pw.this.macd,
    rsi=pw.this.rsi,
    volatility=pw.this.volatility_coef,
    
    simple_signal=pw.this.signal,
    simple_risk_level=pw.this.risk_level
)

pw.io.csv.write(
    output, 
    os.path.join(OUTPUT_DIR, "stock_indicators.csv")
)

print("📊 Pipeline Running Successfully")
print("--------------------------------")

pw.run()