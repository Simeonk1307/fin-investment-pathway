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

class NewsSchema(pw.Schema):
        article_id: int
        headline: str
        description: str
        url: str
        source_name: str
        published_at: str
        category: str
        company: str


class YFinanceSchema(pw.Schema):
    update_time:str
    timestamp: pw.DateTimeNaive
    date: str
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int


