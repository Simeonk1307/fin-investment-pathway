import pathway as pw
import os
from dotenv import load_dotenv
from src.schemas.silver.stocks_schema import YFinanceEquitySchema
from src.utils.common import common_config, profiles
from src.utils.casting import create_schema_parser, cast_to_str, cast_to_int, cast_to_float

load_dotenv()
pw.set_license_key(os.getenv("PATHWAY_LICENSE_KEY"))

consumer_settings = common_config | {
    "group.id": "bronze-stocks-consumer1",
    # "auto.offset.reset": "latest",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": "true",
    "auto.commit.interval.ms": "500",
}

producer_settings = common_config | profiles["high_throughput"] | {"client.id": "silver-stocks-producer"}

BRONZE_TOPIC = os.getenv("REDPANDA_BRONZE_STOCKS_TOPIC")
SILVER_TOPIC = os.getenv("REDPANDA_SILVER_STOCKS_TOPIC")
DLQ_TOPIC = os.getenv("REDPANDA_SILVER_STOCKS_DLQ_TOPIC")

raw = pw.io.redpanda.read(
    rdkafka_settings=consumer_settings,
    topic=BRONZE_TOPIC,
    format="raw",
    autocommit_duration_ms=500,
)

safe_parse_stock = create_schema_parser(
    YFinanceEquitySchema,
    field_mapping={"ticker": "id"}
)

parsed = raw.select(
    result=safe_parse_stock(pw.this.data)
)

with_status = parsed.select(
    success=cast_to_int(pw.this.result["success"]),
    data=pw.this.result["data"],
    error=cast_to_str(pw.this.result["error"]),
    raw=cast_to_str(pw.this.result["raw"]),
)

valid = with_status.filter(pw.this.success == 1)
valid = valid.select(
    ticker=cast_to_str(pw.this.data["ticker"]),
    price=cast_to_float(pw.this.data["price"]),
    time=cast_to_int(pw.this.data["time"]),
    exchange=cast_to_str(pw.this.data["exchange"]),
    quote_type=cast_to_int(pw.this.data["quote_type"]),
    market_hours=cast_to_int(pw.this.data["market_hours"]),
    change_percent=cast_to_float(pw.this.data["change_percent"]),
    day_volume=cast_to_int(pw.this.data["day_volume"]),
    change=cast_to_float(pw.this.data["change"]),
    price_hint=cast_to_int(pw.this.data["price_hint"]),
)

failed = with_status.filter(pw.this.success == 0)
failed = failed.select(
    error=pw.this.error,
    raw_data=pw.this.raw,
)

deduped = valid.groupby(pw.this.ticker, pw.this.time).reduce(
    ticker=pw.reducers.earliest(pw.this.ticker),
    price=pw.reducers.earliest(pw.this.price),
    time=pw.reducers.earliest(pw.this.time),
    exchange=pw.reducers.earliest(pw.this.exchange),
    quote_type=pw.reducers.earliest(pw.this.quote_type),
    market_hours=pw.reducers.earliest(pw.this.market_hours),
    change_percent=pw.reducers.earliest(pw.this.change_percent),
    day_volume=pw.reducers.earliest(pw.this.day_volume),
    change=pw.reducers.earliest(pw.this.change),
    price_hint=pw.reducers.earliest(pw.this.price_hint),
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