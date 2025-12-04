from kafka import KafkaProducer
import json
import os
from dotenv import load_dotenv

load_dotenv()


producer = KafkaProducer(
    bootstrap_servers=os.getenv("REDPANDA_BROKERS"),
    security_protocol=os.getenv("REDPANDA_SECURITY_PROTOCOL"),
    sasl_mechanism=os.getenv("REDPANDA_SASL_MECHANISM"),
    sasl_plain_username=os.getenv("REDPANDA_USERNAME"),
    sasl_plain_password=os.getenv("REDPANDA_PASSWORD"),
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

def send_to_kafka(event: dict, topic: str):
    try:
        # Use the original source field directly
        key_str = event.get("source", "unknown")

        producer.send(
            topic=os.getenv(topic),
            key=key_str.encode("utf-8"),   # KEY = original source
            value=event
        )

        producer.flush()
        print(f" Sent [{key_str}]:", event.get("id"))

    except Exception as e:
        print(" Kafka send error:", e)

