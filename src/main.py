import pathway as pw
from dotenv import load_dotenv
from src.schemas.stock_schema import YFinanceSchema
import os
import time
import math

load_dotenv()

rdkafka_settings = {
    "bootstrap.servers": os.getenv("REDPANDA_BROKER"),
    "security.protocol": os.getenv("REDPANDA_SECURITY_PROTOCOL"),
    "sasl.mechanism": os.getenv("REDPANDA_SASL_MECHANISM"),
    "group.id": f"stock_consumer_group_{int(time.time())}",
    "sasl.username": os.getenv("REDPANDA_USERNAME"),
    "sasl.password": os.getenv("REDPANDA_PASSWORD"),
    "auto.offset.reset": "earliest",
}

# =====================================================
# 1. HELPER FUNCTION
# =====================================================
def calculate_stddev(prices_list) -> float:
    """Calculate standard deviation from a list of prices."""
    if not prices_list or len(prices_list) < 2:
        return 0.0
    
    mean = sum(prices_list) / len(prices_list)
    variance = sum((x - mean) ** 2 for x in prices_list) / len(prices_list)
    return math.sqrt(variance)

# =====================================================
# 2. READ & PREPARE DATA
# =====================================================
stocks = pw.io.redpanda.read(
    rdkafka_settings=rdkafka_settings,
    topic=os.getenv("REDPANDA_STOCK_TOPIC"),
    schema=YFinanceSchema,
    format="json",
    autocommit_duration_ms=1000
)

stocks_with_timestamp = stocks.select(
    timestamp=pw.declare_type(
        pw.DateTimeNaive,
        pw.apply_with_type(
            lambda ms: pw.DateTimeNaive.fromtimestamp(ms / 1000.0),
            pw.DateTimeNaive,
            pw.this.timestamp_ms
        )
    ),
    timestamp_ms=pw.this.timestamp_ms,
    symbol=pw.this.symbol,
    price=pw.this.price,
    change_percent=pw.this.change_percent,
    volume=pw.this.volume,
    update_time=pw.this.update_time
)

# =====================================================
# 3. CURRENT STATS (Global Accumulation)
# =====================================================
current_stats = stocks_with_timestamp.groupby(pw.this.symbol).reduce(
    symbol=pw.this.symbol,
    latest_price=pw.reducers.latest(pw.this.price),
    latest_change_percent=pw.reducers.latest(pw.this.change_percent),
    last_update=pw.reducers.latest(pw.this.update_time),
    
    session_high=pw.reducers.max(pw.this.price),
    session_low=pw.reducers.min(pw.this.price),
    session_avg=pw.reducers.avg(pw.this.price),
    
    # Collect prices for manual stddev
    price_list=pw.reducers.tuple(pw.this.price),
    
    total_volume=pw.reducers.sum(pw.this.volume),
)

current_stats = current_stats.with_columns(
    # ✅ FIX 1: Use apply_with_type
    session_stddev=pw.apply_with_type(
        calculate_stddev, float, pw.this.price_list
    ),
    price_range=pw.this.session_high - pw.this.session_low,
)

# =====================================================
# 4. TIME WINDOWS (5 MIN)
# =====================================================
stats_5min = stocks_with_timestamp.windowby(
    pw.this.timestamp,
    window=pw.temporal.sliding(
        hop=pw.Duration("10s"),
        duration=pw.Duration("5m")
    ),
    behavior=pw.temporal.common_behavior()
).groupby(pw.this.symbol).reduce(
    symbol=pw.this.symbol,
    ma_5min=pw.reducers.avg(pw.this.price),
    high_5min=pw.reducers.max(pw.this.price),
    low_5min=pw.reducers.min(pw.this.price),
    volume_5min=pw.reducers.sum(pw.this.volume),
    prices_5min=pw.reducers.tuple(pw.this.price), # For stddev
)

stats_5min = stats_5min.with_columns(
    # ✅ FIX 1: Use apply_with_type
    volatility_5min=pw.apply_with_type(
        calculate_stddev, float, pw.this.prices_5min
    )
)

# =====================================================
# 5. TIME WINDOWS (15 MIN)
# =====================================================
stats_15min = stocks_with_timestamp.windowby(
    pw.this.timestamp,
    window=pw.temporal.sliding(
        hop=pw.Duration("30s"),
        duration=pw.Duration("15m")
    ),
    behavior=pw.temporal.common_behavior()
).groupby(pw.this.symbol).reduce(
    symbol=pw.this.symbol,
    ma_15min=pw.reducers.avg(pw.this.price),
    prices_15min=pw.reducers.tuple(pw.this.price),
)

stats_15min = stats_15min.with_columns(
    # ✅ FIX 1: Use apply_with_type
    volatility_15min=pw.apply_with_type(
        calculate_stddev, float, pw.this.prices_15min
    )
)

# =====================================================
# 6. JOIN TIMEFRAMES
# =====================================================

# Join 5min
indicators = current_stats.join_left(
    stats_5min, pw.left.symbol == pw.right.symbol
).select(
    *pw.left,
    ma_5min=pw.coalesce(pw.right.ma_5min, pw.left.latest_price),
    high_5min=pw.coalesce(pw.right.high_5min, pw.left.latest_price),
    low_5min=pw.coalesce(pw.right.low_5min, pw.left.latest_price),
    # ✅ FIX 2: Use 0.0 (float) for float columns
    volatility_5min=pw.coalesce(pw.right.volatility_5min, 0.0),
    # ✅ FIX 3: Use 0 (int) for int columns
    volume_5min=pw.coalesce(pw.right.volume_5min, 0),
)

# Join 15min
indicators = indicators.join_left(
    stats_15min, pw.left.symbol == pw.right.symbol
).select(
    *pw.left,
    ma_15min=pw.coalesce(pw.right.ma_15min, pw.left.latest_price),
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
        pw.this.session_stddev, pw.this.session_avg
    ),
)

# RSI Approximation
final = final.with_columns(
    rsi=pw.apply(
        lambda pos_pct, change: min(100.0, max(0.0, 50.0 + (pos_pct - 50.0) * 0.5 + change)),
        pw.apply(
            lambda p, h, l: ((p - l) / (h - l) * 100.0) if h != l else 50.0,
            pw.this.latest_price, pw.this.session_high, pw.this.session_low
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
    symbol=pw.this.symbol,
    time=pw.this.last_update,
    price=pw.this.latest_price,
    
    ma_5min=pw.this.ma_5min,
    ma_15min=pw.this.ma_15min,
    
    bb_upper=pw.this.bb_upper,
    bb_lower=pw.this.bb_lower,
    
    macd=pw.this.macd,
    rsi=pw.this.rsi,
    volatility=pw.this.volatility_coef,
    
    signal=pw.this.signal,
    risk_level=pw.this.risk_level
)

pw.io.jsonlines.write(output, "outputs/stock_indicators.jsonl")
pw.io.csv.write(output, "outputs/stock_indicators.csv")

print("📊 Pipeline Running Successfully")
print("--------------------------------")
print("1. Manual StdDev using apply_with_type")
print("2. Type-safe coalescing (float/float)")
print("3. Full Indicator Set")

pw.run()