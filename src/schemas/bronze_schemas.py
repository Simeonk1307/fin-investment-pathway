import pathway as pw

class UnifiedSchema(pw.Schema):
    event_id: str
    source: str
    source_type: str
    received_at: int
    payload: pw.Json
    schema_version: int


