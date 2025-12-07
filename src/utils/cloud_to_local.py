


import pathway as pw
from src.schemas.silver_schemas import FinnHubStockSchema
from src.schemas.bronze_schemas import UnifiedSchema

rdkafka_settings = {
    "bootstrap.servers": "d4luu1og819mdlmr61u0.any.ap-south-1.mpx.prd.cloud.redpanda.com:9092",
    "security.protocol": "SASL_SSL",
    "sasl.mechanism": "SCRAM-SHA-256",
    "sasl.username": "forall123",
    "sasl.password": "geIDt40rmgOZbzbmtogttp1nJkMTsr",
    "auto.offset.reset": "earliest",
    "group.id": "sksj",
    "log_level": "0",

}

table = pw.io.redpanda.read(
    rdkafka_settings=rdkafka_settings,
    topic="bronze.stocks",
    schema=UnifiedSchema,
    format="json",
)

rdkafka_settings = {
    "bootstrap.servers": "localhost:19092",
    "security.protocol": "PLAINTEXT",
    "sasl.mechanism": "SCRAM-SHA-256",
    "log_level": "0",
}

pw.io.kafka.write(
    table,
    rdkafka_settings=rdkafka_settings,
    topic_name="bronze.stocks",
    format="json",
    key=pw.this.source,
)

pw.run()