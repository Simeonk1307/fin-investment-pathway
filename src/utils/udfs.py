import pathway as pw

@pw.udf(deterministic=True, return_type=pw.DateTimeNaive)
def ms_to_datetime(ms: int) -> pw.DateTimeNaive:
    """Convert milliseconds to datetime."""
    return pw.DateTimeNaive.from_timestamp(ms / 1000.0)

@pw.udf(deterministic=True, return_type=float)
def calc_price_vs_ma(price: float, ma: float) -> float:
    """Calculate price vs moving average percentage."""
    if ma is None or ma == 0:
        return 0.0
    return ((price - ma) / ma) * 100.0

@pw.udf(deterministic=True, return_type=float)
def calc_volatility_coef(std: float, avg: float) -> float:
    """Calculate volatility coefficient."""
    if avg is None or avg == 0:
        return 0.0
    return (std / avg) * 100.0

@pw.udf(deterministic=True, return_type=float)
def calc_price_position(price: float, high: float, low: float) -> float:
    """
    Calculate where price sits in the range (0-100).
    This is a simple RSI proxy.
    """
    if high == low or high is None or low is None:
        return 50.0
    return ((price - low) / (high - low)) * 100.0

@pw.udf(deterministic=True, return_type=str)
def calc_signal(rsi: float, macd: float, price: float, bb_lower: float, bb_upper: float) -> str:
    """Generate trading signal based on indicators."""
    if rsi is None:
        rsi = 50.0
    if macd is None:
        macd = 0.0
    
    # Oversold conditions
    if rsi < 20 and macd > 0:
        return "STRONG_BUY"
    elif rsi < 35 or price <= bb_lower:
        return "BUY"
    
    # Overbought conditions
    elif rsi > 80 and macd < 0:
        return "STRONG_SELL"
    elif rsi > 65 or price >= bb_upper:
        return "SELL"
    
    # Neutral
    else:
        return "HOLD"

@pw.udf(deterministic=True, return_type=str)
def calc_risk_level(volatility_coef: float) -> str:
    """Assess risk based on volatility."""
    if volatility_coef is None:
        return "UNKNOWN"
    elif volatility_coef > 3.0:
        return "HIGH"
    elif volatility_coef > 1.5:
        return "MEDIUM"
    else:
        return "LOW"