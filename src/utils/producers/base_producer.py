from typing import Dict
from confluent_kafka import Producer
import threading
import logging
import time
import sys

class BaseProducer:
    def __init__(self, logger: logging.Logger, topic: str, producer_config: Dict):
        self.logger = logger
        self.topic = topic

        self._running = False
        self._lock = threading.Lock()
        self.initialisation_errors = 0
        self.sent = 0
        self.errors = 0
        self.delivery_errors = 0
        self.misc_errors = 0
        self.statistics = {}

        try:
            self.producer = Producer(producer_config)
            self.logger.info("Kafka Producer Initialisation SUCCESS")
        except Exception as e:
            with self._lock:
                self.errors += 1
                self.initialisation_errors += 1
            self.logger.error(f"Kafka Producer Initialisation FAILURE: {e}")
            raise
    
    def _count_error(self, counter_attr):
        with self._lock:
            setattr(self, counter_attr.__name__, counter_attr + 1)
            self.errors += 1

    def _delivery(self, err, msg):
        with self._lock:
            if err:
                self.errors += 1
                self.delivery_errors += 1
                self.logger.error(f"Delivery FAILURE: {self.errors} {err}")
            else:
                self.sent += 1
                self.logger.info(f"Delivery SUCCESS {self.sent} {msg}")

    def stats(self):
        with self._lock:
            stat = {
                    "delivery_success": (self.sent / max(self.sent + self.delivery_errors, 1)) * 100,
                    "sent": self.sent, 
                    "errors": self.errors, 
                    "initialisation_errors": self.initialisation_errors, 
                    "delivery_errors": self.delivery_errors, 
                    "misc_errors": self.misc_errors
                }
            self.logger.info(f"STATISTICS: {stat}")
            return stat
    
    def run(self):
        ws_thread = threading.Thread(target=self._run_loop, daemon=True)
        ws_thread.start()

        try:
            while self._running:
                time.sleep(0.01)
        except KeyboardInterrupt:
            self.logger.warning("Stopping on KeyboardInterrupt...")
            self._running = False

        self.producer.flush()
        self.logger.info("Successfully flushed the producer...")
        self.stats()
        sys.exit(-1)
