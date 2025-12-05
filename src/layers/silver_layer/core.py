import os
import sys
import time
import logging
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


def create_silver_pipeline(
    name: str,
    output_schema: type[pw.Schema],
    field_mapping: dict,
    dedupe_columns: list[str],
    key_column: str = None,
    debug: bool = False,
) -> None:
    
    logger.info(f"[{name}] Starting ({'DEBUG' if debug else 'PROD'})")
    
    bronze = os.getenv(f"REDPANDA_BRONZE_{name}_TOPIC")
    silver = os.getenv(f"REDPANDA_SILVER_{name}_TOPIC")
    dlq = os.getenv(f"REDPANDA_SILVER_{name}_DLQ_TOPIC")

    if not all([bronze, silver, dlq]):
        logger.error(f"[{name}] Missing topic configuration")
        sys.exit(1)

    logger.info(f"[{name}] Bronze: {bronze}")
    logger.info(f"[{name}] Silver: {silver}")
    logger.info(f"[{name}] DLQ: {dlq}")

    suffix = f"-{int(time.time())}" if debug else ""
    
    consumer = common_config | {
        "group.id": f"bronze-{name.lower()}-consumer{suffix}",
        "auto.offset.reset": "earliest" if debug else "latest",
        "enable.auto.commit": "true",
        "auto.commit.interval.ms": "500",
    }
    
    producer = common_config | profiles["high_throughput"] | {
        "client.id": f"silver-{name.lower()}-producer"
    }

    logger.info(f"[{name}] Consumer group: {consumer['group.id']}")

    try:
        raw = pw.io.redpanda.read(
            rdkafka_settings=consumer,
            topic=bronze,
            schema=UnifiedSchema,
            format="json",
            autocommit_duration_ms=500,
        )

        unified_errors = raw._errors if hasattr(raw, '_errors') else pw.Table.empty()
        unified_valid = raw

        

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
        logger.info(f"[{name}] Dedupe columns: {dedupe_columns}")

        if name == "SOCIALS":
            deduped = deduped.with_columns(
                title=pw.apply(clean_text, pw.this.title),
                content=pw.apply(clean_text, pw.this.content),
            )
            logger.info(f"[{name}] Text cleaning enabled")

        if debug:
            os.makedirs("debug_output", exist_ok=True)
            pw.io.jsonlines.write(valid, f"debug_output/{name.lower()}_valid.jsonl")
            pw.io.jsonlines.write(failed, f"debug_output/{name.lower()}_failed.jsonl")
            logger.info(f"[{name}] Output → debug_output/*.jsonl")
        else:
            pw.io.kafka.write(
                parsed,
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
            logger.info(f"[{name}] Output configured")

        logger.info(f"[{name}] Pipeline ready ✓")

    except Exception as e:
        logger.error(f"[{name}] Failed: {type(e).__name__}: {e}", exc_info=True)
        sys.exit(1)


def run() -> None:
    license_key = os.getenv("PATHWAY_LICENSE_KEY")
    if not license_key:
        logger.error("PATHWAY_LICENSE_KEY not set")
        sys.exit(1)
    
    try:
        pw.set_license_key(license_key)
        logger.info("License validated")
    except Exception as e:
        logger.error(f"Invalid license: {e}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("Starting Pathway runtime")
    logger.info("=" * 60)
    
    start = time.time()
    
    try:
        pw.run()
    except KeyboardInterrupt:
        logger.info("Shutdown initiated")
    except Exception as e:
        logger.error(f"Runtime error: {type(e).__name__}: {e}", exc_info=True)
        sys.exit(1)
    finally:
        elapsed = time.time() - start
        logger.info(f"Stopped after {elapsed:.1f}s")