import pathway as pw

class YFinanceEquitySchema(pw.Schema):
    price: float
    symbol: str # from id    
    timestamp: int
    # exchange: str
    # quote_type: int
    # market_hours: int
    # price_hint: int

class FinnHubStockSchema(pw.Schema):
    price: int
    symbol: str
    timestamp: int
    volume: int

finnhub_stocks_mapping = {
    "price": "p",
    "symbol": "s",
    "timestamp": "t",
    "volume": "v"
}

yfinance_equity_mapping = {
    "symbol": "id"
}