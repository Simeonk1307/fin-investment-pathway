from kafka import KafkaConsumer
import json
from datetime import datetime
from cleaner_producer import send_clean_event
from preprocess_utils import preprocess_event

RAW_TOPIC = "bronze.social"

consumer = KafkaConsumer(
    RAW_TOPIC,
    bootstrap_servers="d4luu1og819mdlmr61u0.any.ap-south-1.mpx.prd.cloud.redpanda.com:9092",
    security_protocol="SASL_SSL",
    sasl_mechanism="SCRAM-SHA-256",
    sasl_plain_username="forall123",
    sasl_plain_password="geIDt40rmgOZbzbmtogttp1nJkMTsr",
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="cleaner-group"
)

print("🚀 Cleaning Consumer is running and listening to RAW events...")

for msg in consumer:
    raw_event = msg.value

    # Clean + unify schema
    clean_event = preprocess_event(raw_event)

    # Store to clean-events
    send_clean_event(clean_event)

    print("✔ CLEANED EVENT SENT:", clean_event["event_id"])
