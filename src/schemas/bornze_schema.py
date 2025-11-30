import pathway as pw
from datetime import datetime

class BronzeEventSchema(pw.Schema):
    event_id: str
    source_type: str
    ticker: str
    event_ts: datetime
    ingest_ts: datetime
    raw_payload: str
    schema_version: int