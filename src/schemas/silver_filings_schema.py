import pathway as pw

class SecFilingsSchema(pw.Schema):
    source: str
    ticker: str
    company: str
    form_type: str
    headline: str
    content:str
    link: str
    time_ms: int
    date: str

sec_filings_mapping = {}