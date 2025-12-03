from kafka import KafkaProducer
import json

CLEAN_TOPIC = "silver.socials"

producer_clean = KafkaProducer(
    bootstrap_servers="d4luu1og819mdlmr61u0.any.ap-south-1.mpx.prd.cloud.redpanda.com:9092",
    security_protocol="SASL_SSL",
    sasl_mechanism="SCRAM-SHA-256",
    sasl_plain_username="forall123",
    sasl_plain_password="geIDt40rmgOZbzbmtogttp1nJkMTsr",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

def send_clean_event(cleaned_event: dict):
    key_bytes = cleaned_event["source"].encode()
    producer_clean.send(
        CLEAN_TOPIC,
        key=key_bytes,
        value=cleaned_event
    )
    producer_clean.flush()
