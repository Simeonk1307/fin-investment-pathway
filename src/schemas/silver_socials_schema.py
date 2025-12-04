import pathway as pw

class SocialsSchema(pw.Schema):
        source: str
        company: str
        title: str
        text: str
        url: str
        query: str
        timestamp: str

# TODO i removed some fields as they were unnecessary like event_id, raw, ingestion_time
# {
#         "source": event.get("source"),
#         "company": event.get("company"),
#         "title": title,
#         "text": text,
#         "url": event.get("id"),
#         "query": event.get("query"),
#         "timestamp": event.get("created_at") or current time,
# }

socials_mapping = {}
