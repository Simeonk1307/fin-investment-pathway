import os
from dotenv import load_dotenv

load_dotenv()

common_config = {
    "bootstrap.servers": os.getenv("REDPANDA_BROKERS"),
    "security.protocol": os.getenv("REDPANDA_SECURITY_PROTOCOL"),
    "sasl.mechanism": os.getenv("REDPANDA_SASL_MECHANISM"),
    "sasl.username": os.getenv("REDPANDA_USERNAME"),
    "sasl.password": os.getenv("REDPANDA_PASSWORD"),
    "log_level": 0
}

profiles = {
    "low_latency": {
        "acks": "all",
        "linger.ms": "0",
        "batch.num.messages": "1",
        "compression.type": "snappy",
        "enable.idempotence": "true",
        "max.in.flight.requests.per.connection": "1",
        "request.timeout.ms": "3000",
        "delivery.timeout.ms": "3000",
        "socket.timeout.ms": "3000",
    },
    "balanced": {
        "acks": "all",
        "linger.ms": "10",
        "batch.num.messages": "1000",
        "compression.type": "snappy",
        "enable.idempotence": "true",
        "max.in.flight.requests.per.connection": "1",
        "request.timeout.ms": "10000",
        "delivery.timeout.ms": "10000",
        "socket.timeout.ms": "10000",
    },
    "high_throughput": {
        "acks": "1",
        "linger.ms": "100",
        "batch.num.messages": "10000",
        "compression.type": "snappy",
        "enable.idempotence": "false",
        "max.in.flight.requests.per.connection": "5",
        "request.timeout.ms": "30000",
        "delivery.timeout.ms": "30000",
        "socket.timeout.ms": "30000",
    }
}