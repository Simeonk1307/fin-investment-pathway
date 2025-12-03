from kafka import KafkaProducer
import json

SENTIMENT_TOPIC = "gold.news"

producer_senti = KafkaProducer(
    bootstrap_servers="d4luu1og819mdlmr61u0.any.ap-south-1.mpx.prd.cloud.redpanda.com:9092",
    security_protocol="SASL_SSL",
    sasl_mechanism="SCRAM-SHA-256",
    sasl_plain_username="forall123",
    sasl_plain_password="geIDt40rmgOZbzbmtogttp1nJkMTsr",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

def send_sentiment(event):
    producer_senti.send(
        SENTIMENT_TOPIC,
        key=event["source"].encode(),  # keep source-based partitioning
        value=event
    )
    producer_senti.flush()
