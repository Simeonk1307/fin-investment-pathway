import time
import json
import feedparser
# NOTE: We import 'kafka' because Redpanda uses the Kafka protocol.
# There is no 'redpanda' python library. This IS the correct way to connect.
from kafka import KafkaProducer

TOPIC = "sec_filings"
# Redpanda defaults to port 9092, just like Kafka
SERVER = "localhost:9092" 
RSS_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=&company=&dateb=&owner=include&start=0&count=40&output=atom"
HEADERS = {'User-Agent': 'RedpandaPipeline contact@example.com'}

# This connects to your Redpanda instance
producer = KafkaProducer(
    bootstrap_servers=SERVER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

seen_links = set()
print(f"Connecting to Redpanda at {SERVER}...")

while True:
    try:
        feed = feedparser.parse(RSS_URL, request_headers=HEADERS)
        for entry in reversed(feed.entries):
            if entry.link not in seen_links:
                full_title = entry.title
                # Safe logic to get filing type
                filing_type = full_title.split(" - ")[0] if " - " in full_title else "Unknown"

                payload = {
                    "title": full_title,
                    "link": entry.link,
                    "type": filing_type,
                    "updated": entry.updated
                }
                
                # Sends data to Redpanda
                producer.send(TOPIC, payload)
                print(f"Sent to Redpanda: {filing_type}")
                seen_links.add(entry.link)
        time.sleep(60)
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(10)