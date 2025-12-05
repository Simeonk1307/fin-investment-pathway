import json, time, logging
from typing import List, Dict
import yfinance as yf
from src.layers.bronze_layer.base.base_producer import BaseProducer
from src.layers.bronze_layer.event_envelope import create_event_envelope


class YFinanceStocksProducer(BaseProducer):
    def __init__(self, logger: logging.Logger, topic: str, producer_config: Dict, tickers: List[str]):
        super().__init__(logger=logger, topic=topic, producer_config=producer_config)
        self.tickers = tickers
        self._running = True

    def _message_handler(self, message: Dict):
        if not self._running:
            return

        try:
            data = create_event_envelope(payload= message, source="yfinance", source_type="rest") 
            self.producer.produce(
                topic=self.topic,
                key="yfinance",
                value=json.dumps(data).encode(),
                callback=self._delivery
            )
            self.producer.poll(0)

        except Exception as e:
            self.logger.error(f"FAILURE in _message_handler: {e}")
            self._count_error(self.misc_errors)

    def _run_loop(self):
        retries = 0
        while self._running and retries < 5:
            try:
                ws = yf.WebSocket()
                ws.subscribe(self.tickers)
                self.logger.info(f"Subscribed to YFinance: {', '.join(self.tickers)}...")
                retries = 0
                ws.listen(self._message_handler)
                self.logger.info(f"Currently Listening...")

            except Exception as e:
                retries += 1
                self.logger.error(f"Reconnecting ({retries}/5): {e}")
                time.sleep(min(5 * retries, 30))
