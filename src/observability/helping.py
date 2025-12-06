import logging
from typing import Dict, Callable
import time
# --------------------
# OTEL LOGGING
# --------------------
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

# --------------------
# OTEL METRICS
# --------------------
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry import metrics


class OTELLoggerManager:
    """
    Reusable OpenTelemetry Logger wrapper.
    """

    def __init__(
        self,
        service_name: str = "default-service",
        otlp_endpoint: str = "http://localhost:4317",
        log_level=logging.INFO,
    ):
        resource = Resource.create({"service.name": service_name})

        provider = LoggerProvider(resource=resource)
        provider.add_log_record_processor(
            BatchLogRecordProcessor(
                OTLPLogExporter(endpoint=otlp_endpoint)
            )
        )

        handler = LoggingHandler(logger_provider=provider)

        logger = logging.getLogger(service_name)
        logger.setLevel(log_level)
        logger.addHandler(handler)
        # logging.getLogger("librdkafka").setLevel(logging.CRITICAL)
        # logging.getLogger("confluent_kafka").setLevel(logging.CRITICAL)
        # Also print logs to console (optional)
        # console = logging.StreamHandler()
        # console.setLevel(log_level)
        # logger.addHandler(console)

        self.logger = logger

    def get_logger(self):
        return self.logger


class OTELMetricsManager:
    """
    Reusable OpenTelemetry Metrics wrapper (Counters, Gauges, Histograms).
    """

    def __init__(
        self,
        service_name: str = "default-service",
        otlp_endpoint: str = "http://localhost:4317",
        export_interval_ms: int = 5000,
    ):
        resource = Resource.create({"service.name": service_name})

        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(
                endpoint=otlp_endpoint,
                insecure=True,
            ),
            export_interval_millis=export_interval_ms,
        )

        provider = MeterProvider(
            resource=resource,
            metric_readers=[reader],
        )

        metrics.set_meter_provider(provider)
        self.meter = metrics.get_meter(service_name)
        self._metrics: Dict[str, object] = {}

    # Counter metric
    def counter(self, name: str, description: str = "", unit: str = "1"):
        if name not in self._metrics:
            self._metrics[name] = self.meter.create_counter(
                name=name,
                description=description,
                unit=unit,
            )
        return self._metrics[name]

    # Histogram metric
    def histogram(self, name: str, description: str = "", unit: str = "s"):
        if name not in self._metrics:
            self._metrics[name] = self.meter.create_histogram(
                name=name,
                description=description,
                unit=unit,
            )
        return self._metrics[name]

    # Gauge metric
    def gauge(
        self,
        name: str,
        callback: Callable,
        description: str = "",
        unit: str = "1",
    ):
        if name not in self._metrics:
            self._metrics[name] = self.meter.create_observable_gauge(
                name=name,
                callbacks=[callback],
                description=description,
                unit=unit,
            )
        return self._metrics[name]