import uuid
import datetime
from typing import Any, Literal


def create_event_envelope(
    payload: dict[str, Any],
    source: str,
    source_type: str,
    schema_version: int = 1,
    event_id: str | None = None,
) -> dict[str, Any]:
    return {
        "event_id": event_id or str(uuid.uuid4()),
        "source": source,
        "source_type": source_type,
        "received_at": int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000),
        "payload": payload,
        "schema_version": schema_version,
    }