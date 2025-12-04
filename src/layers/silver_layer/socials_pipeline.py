import pathway as pw
import os
from dotenv import load_dotenv
from src.schemas.bronze_schema import BronzeSchema
from src.schemas.silver_socials_schema import SocialsSchema, socials_mapping
from src.utils.common import common_config, profiles
from src.utils.casting import create_schema_parser, cast_to_str, cast_to_int,  unpack_from_schema, dedupe_from_schema
from src.utils.clean_text import clean_text

load_dotenv()
pw.set_license_key(os.getenv("PATHWAY_LICENSE_KEY"))

consumer_settings = common_config | {
    "group.id": "bronze-socials-consumer",
    "auto.offset.reset": "latest", #"latest" replace later
    "enable.auto.commit": "true",
    "auto.commit.interval.ms": "500",
}

producer_settings = common_config | profiles["high_throughput"] | {"client.id": "silver-socials-producer"}

BRONZE_TOPIC = os.getenv("REDPANDA_BRONZE_SOCIALS_TOPIC")
SILVER_TOPIC = os.getenv("REDPANDA_SILVER_SOCIALS_TOPIC")
DLQ_TOPIC = os.getenv("REDPANDA_SILVER_SOCIALS_DLQ_TOPIC")


#TODO CHECK IF EVRYTHING WORKS FINE
raw = pw.io.redpanda.read(
    rdkafka_settings=consumer_settings,
    topic=BRONZE_TOPIC,
    schema=BronzeSchema,
    format="json",
    autocommit_duration_ms=500,
)

safe_parse_stock = create_schema_parser(
    schema_class=SocialsSchema,
    field_mapping=socials_mapping
)

parsed = raw.select(
    result=safe_parse_stock(pw.this.payload)
)

with_status = parsed.select(
    success=cast_to_int(pw.this.result["success"]),
    data=pw.this.result["data"],
    error=cast_to_str(pw.this.result["error"]),
    raw=cast_to_str(pw.this.result["raw"]),
)

valid = with_status.filter(pw.this.success == 1)
valid = unpack_from_schema(
    table=valid, 
    schema_class=SocialsSchema, 
    source_column="data"
)

failed = with_status.filter(pw.this.success == 0)
failed = failed.select(
    error=pw.this.error,
    raw_data=pw.this.raw,
)

# TODO
deduped = dedupe_from_schema(
    table=valid, 
    schema_class=SocialsSchema, 
    dedupe_columns=["source","timestamp", "company", "title"] ##
) # look into this - ["url"] enough look into it


# TODO shd we remove anything and check if this even works
#Apply some cleaning here
cleaned = deduped.with_columns(
    title = pw.apply(clean_text, pw.this.title),
    text = pw.apply(clean_text, pw.this.text),
)


# TODO
pw.io.kafka.write(
    deduped,
    rdkafka_settings=producer_settings, 
    topic_name=SILVER_TOPIC, 
    format="json",
    key=pw.this.company ## is source a better idea
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