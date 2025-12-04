from kafka import KafkaProducer
import json

REDPANDA_BROKER = "d4luu1og819mdlmr61u0.any.ap-south-1.mpx.prd.cloud.redpanda.com:9092"
REDPANDA_USERNAME = "forall123"
REDPANDA_PASSWORD = "geIDt40rmgOZbzbmtogttp1nJkMTsr"
REDPANDA_TOPIC = "bronze.social"


producer = KafkaProducer(
    bootstrap_servers=REDPANDA_BROKER,
    security_protocol="SASL_SSL",
    sasl_mechanism="SCRAM-SHA-256",
    sasl_plain_username=REDPANDA_USERNAME,
    sasl_plain_password=REDPANDA_PASSWORD,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

def send_to_kafka(event: dict):
    try:
        # Use the original source field directly
        key_str = event.get("source", "unknown")

        producer.send(
            REDPANDA_TOPIC,
            key=key_str.encode("utf-8"),   # KEY = original source
            value=event
        )

        producer.flush()
        print(f" Sent [{key_str}]:", event.get("id"))

    except Exception as e:
        print(" Kafka send error:", e)

