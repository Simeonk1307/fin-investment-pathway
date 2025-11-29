import pathway as pw

class YFinanceSchema(pw.Schema):
    timestamp_ms: int
    date: str
    update_time:str
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int

# class PolygonSchema(pw.Schema):
#     update_time:str
#     timestamp: pw.DateTimeNaive
#     date: str
#     symbol: str
#     price: float
#     change: float
#     change_percent: float
#     volume: int

# so on for other stock data schemas