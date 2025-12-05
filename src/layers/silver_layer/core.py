import os
import time
import logging
import threading
import pathway as pw
from dotenv import load_dotenv
from src.schemas.bronze_schemas import UnifiedSchema
from src.utils.common import common_config, profiles
from src.layers.silver_layer.clean_text import clean_text
from src.utils.casting import (
    create_schema_parser,
    cast_to_str,
    cast_to_int,
    unpack_from_schema,
    dedupe_from_schema,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("librdkafka").setLevel(logging.CRITICAL)
logging.getLogger("confluent_kafka").setLevel(logging.CRITICAL)

KAFKA_RESILIENCE = {
    "socket.timeout.ms": "30000",
    "socket.connection.setup.timeout.ms": "30000",
    "reconnect.backoff.ms": "500",
    "reconnect.backoff.max.ms": "30000",
    "message.timeout.ms": "60000",
    "request.timeout.ms": "60000",
    "session.timeout.ms": "60000",
    "heartbeat.interval.ms": "20000",
    "log.connection.close": "false",
    "log_level": "2",
}

_shutdown_event = threading.Event()


def request_shutdown():
    _shutdown_event.set()


def is_shutdown_requested():
    return _shutdown_event.is_set()


def reset_shutdown():
    _shutdown_event.clear()


def create_silver_pipeline(
    name: str,
    output_schema: type[pw.Schema],
    field_mapping: dict,
    dedupe_columns: list[str],
    key_column: str = None,
    debug: bool = False,
    extra_kafka_config: dict = None,
) -> None:
    extra = extra_kafka_config or {}
    
    bronze = os.getenv(f"REDPANDA_BRONZE_{name}_TOPIC")
    silver = os.getenv(f"REDPANDA_SILVER_{name}_TOPIC")
    dlq = os.getenv(f"REDPANDA_SILVER_{name}_DLQ_TOPIC")

    if not all([bronze, silver, dlq]):
        raise ValueError(f"[{name}] Missing topic configuration")

    logger.info(f"[{name}] Bronze: {bronze} | Silver: {silver} | DLQ: {dlq}")

    suffix = f"-{int(time.time())}" if debug else ""
    
    consumer = common_config | KAFKA_RESILIENCE | extra | {
        "group.id": f"bronze-{name.lower()}-consumer{suffix}",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": "true",
        "auto.commit.interval.ms": "500",
    }
    
    producer = common_config | KAFKA_RESILIENCE | extra | profiles["high_throughput"] | {
        "client.id": f"silver-{name.lower()}-producer"
    }

    logger.info(f"[{name}] Consumer: {consumer['group.id']}")

    raw = pw.io.redpanda.read(
        rdkafka_settings=consumer,
        topic=bronze,
        schema=UnifiedSchema,
        format="json",
        autocommit_duration_ms=500,
    )

    parsed = raw.select(
        result=create_schema_parser(output_schema, field_mapping)(pw.this.payload)
    )

    with_status = parsed.select(
        success=cast_to_int(pw.this.result["success"]),
        data=pw.this.result["data"],
        error=cast_to_str(pw.this.result["error"]),
        raw=cast_to_str(pw.this.result["raw"]),
    )

    valid = unpack_from_schema(
        with_status.filter(pw.this.success == 1),
        output_schema,
        "data",
    )

    failed = with_status.filter(pw.this.success == 0).select(
        error=pw.this.error,
        raw_data=pw.this.raw,
    )

    deduped = dedupe_from_schema(valid, output_schema, dedupe_columns)

    if name in ["SOCIALS", "NEWS"]:
        deduped = deduped.with_columns(
            title=pw.apply(clean_text, pw.this.title),
            content=pw.apply(clean_text, pw.this.content),
        )

    if debug:
        os.makedirs("debug_output", exist_ok=True)
        pw.io.jsonlines.write(deduped, f"debug_output/{name.lower()}_valid.jsonl")
        pw.io.jsonlines.write(failed, f"debug_output/{name.lower()}_failed.jsonl")
    else:
        pw.io.kafka.write(
            deduped,
            rdkafka_settings=producer,
            topic_name=silver,
            format="json",
            **({"key": getattr(pw.this, key_column)} if key_column else {}),
        )
        pw.io.kafka.write(
            failed,
            rdkafka_settings=producer,
            topic_name=dlq,
            format="json",
        )

    logger.info(f"[{name}] Pipeline ready ✓")


def run_with_shutdown() -> str:
    pw.set_license_key(os.getenv("PATHWAY_LICENSE_KEY"))
    reset_shutdown()
    
    result = {"status": "success", "error": None}
    start = time.time()
    
    def pathway_thread():
        try:
            pw.run()
        except Exception as e:
            result["error"] = e
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["resolve", "connection", "timeout", "kafka", "broker", "network", "host"]):
                result["status"] = "connection"
            else:
                result["status"] = "error"
    
    thread = threading.Thread(target=pathway_thread, daemon=True)
    thread.start()
    
    while thread.is_alive():
        if is_shutdown_requested():
            logger.info("[CORE] Shutdown requested, stopping...")
            break
        thread.join(timeout=0.5)
    
    elapsed = time.time() - start
    logger.info(f"[CORE] Stopped after {elapsed:.1f}s")
    
    if is_shutdown_requested():
        return "interrupted"
    
    if result["error"]:
        if result["status"] == "connection":
            logger.warning(f"[CORE] Connection error: {result['error']}")
        else:
            logger.error(f"[CORE] Runtime error: {result['error']}")
    
    return result["status"]