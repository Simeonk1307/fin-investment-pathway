import pathway as pw

class StockSchema(pw.Schema):
    update_time:str
    timestamp: pw.DateTimeNaive
    date: str
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int