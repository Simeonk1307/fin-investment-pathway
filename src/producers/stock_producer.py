import json
import time
import threading
import datetime
from typing import List, Dict, Any
import pathway as pw
from confluent_kafka import Producer
from ..logger_config import get_module_logger
import yfinance as yf
from dotenv import load_dotenv
import os


class YFinanceProducer:
    """YFinance to Redpanda producer."""
    
    def __init__(self, logger, topic: str, producer_config: Dict[str, Any], tickers: List[str]):
        self.logger = logger
        self.topic = topic
        self.tickers = tickers
        
        try:
            self.producer = Producer(producer_config)
            self.logger.info("✓ Initialization: Redpanda producer created")
        except Exception as e:
            self.logger.error(f"✗ Failed Initialization of producer: {e}")
            raise
        
        # Thread-safe state
        self._running = threading.Event()
        self._running.set()
        self._stats_lock = threading.Lock()
        self.stats = {"sent": 0, "errors": 0}
    
    def _delivery_callback(self, err, msg):
        """Thread-safe delivery report handler."""
        with self._stats_lock:
            if err:
                self.stats["errors"] += 1
                self.logger.error(f"Delivery failed: {err}")
            else:
                self.stats["sent"] += 1
    
    def get_stats(self) -> Dict[str, int]:
        """Thread-safe stats access."""
        with self._stats_lock:
            return self.stats.copy()
        
    def handle_message(self, msg: Dict):
        """WebSocket message handler."""
        if not self._running.is_set():
            return
            
        try:
            timestamp_ms = int(msg.get("time", datetime.datetime.now().timestamp() * 1000))
            dt = datetime.datetime.fromtimestamp(timestamp_ms / 1000)

            data = {
                "timestamp_ms": timestamp_ms,
                "date": dt.strftime("%d-%m-%Y"),
                "update_time": dt.strftime("%H:%M:%S"),
                "symbol": msg.get("id", ""),
                "volume": int(msg.get("day_volume", 0)),
                "price": float(msg.get("price", 0.0)),
                "change": float(msg.get("change", 0.0)),
                "change_percent": float(msg.get("change_percent", 0.0)),
            }
            
            self.producer.produce(
                topic=self.topic,
                key=data["symbol"].encode(),
                value=json.dumps(data).encode(),
                callback=self._delivery_callback
            )
            self.producer.poll(0)
            
            self.logger.info(
                f"✓ {data['symbol']}: ${data['price']:.2f} "
                f"({data['change_percent']:+.2f}%)"
            )
            
        except Exception as e:
            self.logger.error(f"✗ Error in handle_message: {e}")
            with self._stats_lock:
                self.stats["errors"] += 1
    
    def run(self):
        """Main runner with clean shutdown."""
        ws_thread = threading.Thread(target=self._websocket_loop, daemon=True)
        ws_thread.start()
        
        try:
            while self._running.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.warning("!! Stopping on KeyboardInterrupt... !!")
            self._running.clear()
            time.sleep(2)
        
        # Final stats
        stats = self.get_stats()
        self.logger.info(f"Sent: {stats['sent']}, Errors: {stats['errors']}")
        self.producer.flush(10)
        self.logger.info("✓ Done")
    
    def _websocket_loop(self):
        """WebSocket loop with reconnection."""
        retries = 0
        
        while self._running.is_set() and retries < 5:
            try:
                ws = yf.WebSocket()
                ws.subscribe(self.tickers)
                self.logger.info(f"Connected: {', '.join(self.tickers)}")
                retries = 0  # Reset on successful connection
                ws.listen(self.handle_message)
                
            except Exception as e:
                retries += 1
                self.logger.error(f"Reconnecting ({retries}/5): {e}")
                time.sleep(min(5 * retries, 30))


if __name__ == "__main__":
    load_dotenv()
    
    tickers = ["NVDA","MSFT","AAPL","GOOGL","AMZN","META","AVGO","TSLA"]

    logger = get_module_logger("YFinanceProducer")
    producer = YFinanceProducer(
        tickers=tickers,
        logger=logger,
        topic=os.getenv("REDPANDA_STOCK_TOPIC"),
        config={
            "bootstrap.servers": os.getenv("REDPANDA_BROKER"),
            "security.protocol": os.getenv("REDPANDA_SECURITY_PROTOCOL"),
            "sasl.mechanism": os.getenv("REDPANDA_SASL_MECHANISM"),
            "sasl.username": os.getenv("REDPANDA_USERNAME"),
            "sasl.password": os.getenv("REDPANDA_PASSWORD"),
        }
    )
    
    logger.info("YFinance → Redpanda Producer")
    logger.info("=" * 40)
    logger.info("Press Ctrl+C to stop...")
    logger.info("=" * 40)
    
    producer.run()