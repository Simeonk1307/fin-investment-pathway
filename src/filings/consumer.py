import pathway as pw
import requests
from bs4 import BeautifulSoup

class InputSchema(pw.Schema):
    title: str
    link: str
    type: str
    updated: str

@pw.udf
def smart_extract(url: str, filing_type: str) -> str:
    headers = {'User-Agent': 'RedpandaPipeline contact@example.com'}
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code != 200: return f"Error: HTTP {resp.status_code}"
        soup = BeautifulSoup(resp.content, 'html.parser')
        text = soup.get_text(" ", strip=True)

        if "10-K" in filing_type:
            idx = text.find("Risk Factors")
            return "RISK: " + text[idx:idx+300] if idx != -1 else "No Risk Found"
        elif "4" in filing_type:
            idx = text.find("Relationship of Reporting Person")
            return "INSIDER: " + text[idx:idx+300] if idx != -1 else "Insider Trade"
        elif "13G" in filing_type:
            idx = text.find("NAME OF REPORTING PERSON")
            return "WHALE: " + text[idx:idx+100] if idx != -1 else "Whale Move"
        else:
            return "Preview: " + text[:100]
    except:
        return "Error"

# Pathway connects to Redpanda using "kafka" settings
# This is standard practice.
filings = pw.io.kafka.read(
    rdkafka_settings={
        "bootstrap.servers": "localhost:9092",
        "group.id": "redpanda_group",
        "auto.offset.reset": "earliest"
    },
    topic="sec_filings",
    format="json",
    schema=InputSchema,
    autocommit_duration_ms=1000
)

processed = filings.select(
    **pw.this,
    info_extract=smart_extract(pw.this.link, pw.this.type)
)

pw.io.jsonlines.write(processed, "output_redpanda.jsonl")
pw.run()