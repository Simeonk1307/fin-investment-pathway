import logging
import json
from confluent_kafka import Producer
from threading import Lock


class BaseProducer:
    """Unified base class for all Bronze producers.
    
    Handles:
        - Kafka initialization
        - Delivery callbacks
        - Metrics
        - Controlled run loop and graceful shutdown
    """

    def __init__(self, logger: logging.Logger, topic: str, producer_config: dict):
        self.logger = logger
        self.topic = topic
        self._running = True

        self.lock = Lock()
        self.sent = 0
        self.delivery_errors = 0
        self.misc_errors = 0
        self.initialisation_errors = 0

        try:
            self.producer = Producer(producer_config)
            self.logger.info("[PRODUCER INIT] Success. Topic=%s", topic)
        except Exception as e:
            self.initialisation_errors += 1
            self.logger.error("[PRODUCER INIT] FAILED: %s", e, exc_info=True)
            raise

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    def _record_delivery(self, err, msg):
        with self.lock:
            if err:
                self.delivery_errors += 1
                self.logger.error("[KAFKA DELIVERY FAILED] %s", err)
            else:
                self.sent += 1
                self.logger.debug("[KAFKA DELIVERY OK] offset=%s", msg.offset())

    def send(self, payload: dict, key: str = "default"):
        """Safe send wrapper with unified error logging."""
        try:
            self.producer.produce(
                topic=self.topic,
                key=str(key).encode(),
                value=json.dumps(payload).encode(),
                callback=self._record_delivery
            )
            self.producer.poll(0)
            return True
        except Exception as e:
            self.misc_errors += 1
            self.logger.error("[KAFKA PRODUCE ERROR] %s", e, exc_info=True)
            return False

    # -------------------------------------------------------------------------
    # Graceful run loop
    # -------------------------------------------------------------------------
    def run(self):
        """Main lifecycle: start producer loop until stop() is called."""
        self.logger.info("[PRODUCER] Starting run loop...")

        try:
            self._run_loop()  # <-- NO produce_fn NEEDED
        except KeyboardInterrupt:
            self.logger.warning("[PRODUCER] KeyboardInterrupt received.")
        except Exception as e:
            self.logger.error("[PRODUCER ERROR] %s", e, exc_info=True)
        finally:
            self.logger.info("[PRODUCER] Flushing and shutting down...")
            self._running = False
            self.producer.flush()
            self._log_stats()

    # -------------------------------------------------------------------------
    def _log_stats(self):
        total = self.sent + self.delivery_errors
        success_rate = (self.sent / total * 100) if total > 0 else 0

        self.logger.info("========== PRODUCER SUMMARY ==========")
        self.logger.info("Messages Sent         : %s", self.sent)
        self.logger.info("Delivery Errors       : %s", self.delivery_errors)
        self.logger.info("Other Errors          : %s", self.misc_errors)
        self.logger.info("Success Rate          : %.2f%%", success_rate)
        self.logger.info("======================================")

    # -------------------------------------------------------------------------
    def stop(self):
        """Signal the loop to stop."""
        self.logger.info("[PRODUCER] Stop requested.")
        self._running = False

    # -------------------------------------------------------------------------
    def _run_loop(self):
        """Must be implemented by child class."""
        raise NotImplementedError("Child producer must implement _run_loop().")
