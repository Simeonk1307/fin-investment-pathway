import logging, json
from confluent_kafka import Producer
from threading import Lock

class BaseProducer:
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
            self.logger.info(f"Producer ready (topic={topic})")
        except Exception as e:
            self.initialisation_errors += 1
            self.logger.error(f"Producer init failed: {e}", exc_info=True)
            raise

    def _record_delivery(self, err, msg):
        with self.lock:
            if err:
                self.delivery_errors += 1
                self.logger.error(f"Delivery error: {err}")
            else:
                self.sent += 1

    def send(self, payload: dict, key: str = "default"):
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
            self.logger.error(f"Send failed: {e}", exc_info=True)
            return False

    def run(self):
        self.logger.info("Producer loop started.")
        try:
            self._run_loop()
        except KeyboardInterrupt:
            self.logger.warning("Interrupted, stopping producer...")
        except Exception as e:
            self.logger.error(f"Producer crashed: {e}", exc_info=True)
        finally:
            self.logger.info("Cleaning up producer...")
            self._running = False
            self.producer.flush()
            self._log_stats()

    def _log_stats(self):
        total = self.sent + self.delivery_errors
        rate = (self.sent / total * 100) if total else 0
        self.logger.info(
            f"Summary: sent={self.sent}, delivery_errors={self.delivery_errors}, "
            f"other_errors={self.misc_errors}, success_rate={rate:.2f}%"
        )

    def stop(self):
        self._running = False
        self.logger.info("Stop requested for producer.")

    def _run_loop(self):
        raise NotImplementedError
