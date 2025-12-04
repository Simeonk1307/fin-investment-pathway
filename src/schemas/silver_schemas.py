import pathway as pw

# STOCKS
class YFinanceEquitySchema(pw.Schema):
    price: float
    symbol: str 
    timestamp: int

class FinnHubStockSchema(pw.Schema):
    price: float
    symbol: str
    timestamp: int
    volume: float

finnhub_stocks_mapping = { 
    "price": "p",
    "symbol": "s",
    "timestamp": "t",
    "volume": "v"
}

yfinance_equity_mapping = {
    "symbol": "id"
}


# NEWS
class FinnHubNewsSchema(pw.Schema):
    news_id: int
    symbol: str
    timestamp: int
    source: str
    category: str
    title: str
    content: str
    url: str
    image_url: str
        
finnhub_news_mapping = {
    "news_id": "id",
    "symbol": "related",
    "timestamp": "datetime",
    "title": "headline",
    "content": "summary",
    "image_url": "image",
}

# SOCIALS
class SocialsSchema(pw.Schema):
    symbol: str
    source: str
    url: str
    title: str
    content: str
        

socials_mapping = {
    "symbol" : "company"
}

# FILINGS
class SecFilingsSchema(pw.Schema):
    symbol: str
    timestamp: int
    source: str
    company: str
    form_type: str
    title: str
    content:str
    url: str
    
    date: str

sec_filings_mapping = {
    "symbol" : "ticker",
    "title": "headline",
    "url": "link",
    "timestamp": "time_ms"
}

