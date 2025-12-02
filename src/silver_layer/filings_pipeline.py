import pathway as pw
import os
from dotenv import load_dotenv
from src.schemas.silver.filings_schema import SecFilingsSchema
from src.utils.common import common_config, profiles
from src.utils.casting import create_schema_parser, cast_to_str, cast_to_int, cast_to_float

load_dotenv()
pw.set_license_key(os.getenv("PATHWAY_LICENSE_KEY"))

consumer_settings = common_config | {
    "group.id": "bronze-filings-consumer1",
    # "auto.offset.reset": "latest",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": "true",
    "auto.commit.interval.ms": "500",
}

producer_settings = common_config | profiles["high_throughput"] | {"client.id": "silver-filings-producer"}

BRONZE_TOPIC = os.getenv("REDPANDA_BRONZE_FILINGS_TOPIC")
SILVER_TOPIC = os.getenv("REDPANDA_SILVER_FILINGS_TOPIC")
DLQ_TOPIC = os.getenv("REDPANDA_SILVER_FILINGS_DLQ_TOPIC")

raw = pw.io.redpanda.read(
    rdkafka_settings=consumer_settings,
    topic=BRONZE_TOPIC,
    format="raw",
    autocommit_duration_ms=500,
)

safe_parse_filings = create_schema_parser(
    SecFilingsSchema
)

parsed = raw.select(
    result=safe_parse_filings(pw.this.data)
)

with_status = parsed.select(
    success=cast_to_int(pw.this.result["success"]),
    data=pw.this.result["data"],
    error=cast_to_str(pw.this.result["error"]),
    raw=cast_to_str(pw.this.result["raw"]),
)

valid = with_status.filter(pw.this.success == 1)
valid = valid.select(
    source    = pw.this.data["source"],
    ticker    = pw.this.data["ticker"],
    company   = pw.this.data["company"],
    form_type = pw.this.data["form_type"],
    headline  = pw.this.data["headline"],
    link      = pw.this.data["link"],
    time_ms   = pw.cast_to_int(pw.this.data["time_ms"]),
    date      = pw.this.data["date"],
)


failed = with_status.filter(pw.this.success == 0)
failed = failed.select(
    error=pw.this.error,
    raw_data=pw.this.raw,
)

deduped = valid.groupby(pw.this.ticker, pw.this.time_ms).reduce(
    source      = pw.reducers.earliest(pw.this.source),
    ticker      = pw.reducers.earliest(pw.this.ticker),
    company     = pw.reducers.earliest(pw.this.company),
    form_type   = pw.reducers.earliest(pw.this.form_type),
    headline    = pw.reducers.earliest(pw.this.headline),
    link        = pw.reducers.earliest(pw.this.link),
    time_ms     = pw.reducers.earliest(pw.this.time_ms),
    date        = pw.reducers.earliest(pw.this.date),
)



pw.io.kafka.write(
    deduped,
    rdkafka_settings=producer_settings, 
    topic_name=SILVER_TOPIC, 
    format="json",
    key=pw.this.ticker
)

pw.io.kafka.write(
    failed, 
    rdkafka_settings=producer_settings, 
    topic_name=DLQ_TOPIC, 
    format="json",
)

print(f"Pipeline: {BRONZE_TOPIC} → {SILVER_TOPIC} (DLQ: {DLQ_TOPIC})")

try:
    pw.run()
except KeyboardInterrupt:
    print("KeyboardInterrupt!!")