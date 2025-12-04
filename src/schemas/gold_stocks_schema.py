import pathway as pw

class GoldStockIndicatorsSchema(pw.Schema):
    ticker: str
    latest_update_time: int
    latest_price: float

    ma_5min: float
    ma_15min: float

    bb_upper: float
    bb_lower: float

    macd: float
    rsi: float
    volatility: float

    simple_signal: str
    simple_risk_level: str


# class 
