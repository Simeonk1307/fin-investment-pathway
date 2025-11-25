import json
import time
import threading
import datetime
from typing import List, Dict, Any, Optional
from confluent_kafka import Producer
from ..logger_config import get_module_logger
import finnhub
import os
from dotenv import load_dotenv

class FinnHubNewsProducer:
    def __init__(self, logger, topic: str, producer_config: Dict[str, Any], 
                 tickers: List[str], poll_interval: int = 120, lookback_days: int = 1,
                 api_key: Optional[str] = None):
        
        self.logger = logger
        self.topic = topic
        self.tickers = tickers
        self.poll_interval = poll_interval
        self.lookback_days = lookback_days
        
        try:
            self.producer = Producer(producer_config)
            self.logger.info("✓ Redpanda producer created")
        except Exception as e:
            self.logger.error(f"✗ Failed to create Redpanda producer: {e}")
            raise
        
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            raise ValueError("FINNHUB API key not found.")
        try:
            self.finnhub_client = finnhub.Client(api_key=self.api_key)
            self.logger.info(f"✓ Finnhub client initialized for tickers: {', '.join(tickers)}")
        except Exception as e:
            self.logger.error(f"✗ Failed to initialize Finnhub client: {e}")
            raise
        self.seen_ids: set[int] = set()
        self.last_sent_timestamp_unix_sec: int = 0
        
        self._running = threading.Event()
        self._running.set()
        self._stats_lock = threading.Lock()
        self.stats = {"sent": 0, "errors": 0}
    
    def _delivery_callback(self, err, msg):
        with self._stats_lock:
            self.stats["errors" if err else "sent"] += 1
    
    def get_stats(self) -> Dict[str, int]:
        with self._stats_lock:
            return self.stats.copy()
        
    def _fetch_articles_from_finnhub(self) -> List[Dict[str, Any]]:
        current_date = datetime.date.today()
        from_date = current_date - datetime.timedelta(days=self.lookback_days)
        all_articles = []

        try:
            for ticker in self.tickers:
                articles = self.finnhub_client.company_news(
                    ticker,
                    _from=from_date.strftime("%Y-%m-%d"),
                    to=current_date.strftime("%Y-%m-%d")
                )
                all_articles.extend(articles)
                time.sleep(0.1) 
            return all_articles
        except Exception as e:
            self.logger.error(f"✗ Error fetching articles from Finnhub: {e}", exc_info=True)
            return []
    
    @staticmethod
    def _parse_finnhub_article(article: Dict[str, Any]) -> Dict[str, Any]:
        timestamp_unix_sec = int(article.get('datetime', 0))
        published_dt = datetime.datetime.fromtimestamp(timestamp_unix_sec)

        return {
            "id": int(article.get('id', 0)),
            'headline': article.get('headline', ''),
            'summary': article.get('summary', ''),
            'url': article.get('url', ''),
            'source': article.get('source', 'Unknown'),
            'published_at': published_dt.isoformat(timespec="seconds"),
            'category': article.get('category', ''),
            'ticker': article.get('related', ''),
            'timestamp': timestamp_unix_sec,
        }

    def run(self):
        news_fetching_thread = threading.Thread(target=self._news_fetching_loop, daemon=True)
        news_fetching_thread.start()
        
        try:
            while self._running.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.warning("!! Stopping on KeyboardInterrupt... !!")
            self._running.clear()
            time.sleep(3)
            
        self.producer.flush(10)
        stats = self.get_stats()
        self.logger.info(f"✓ Producer finished. Sent: {stats['sent']}, Errors: {stats['errors']}")
    
    def _news_fetching_loop(self):
        retries = 0
        max_retries = 5
        
        while self._running.is_set():
            try:
                raw_articles = self._fetch_articles_from_finnhub()
                new_articles_count = 0
                raw_articles.sort(key=lambda x: x.get('datetime', 0))

                for raw_article in raw_articles:
                    article_id = int(raw_article.get('id', 0))
                    article_timestamp = int(raw_article.get('datetime', 0))
                    
                    if article_id == 0 or article_timestamp == 0 or \
                       article_id in self.seen_ids or \
                       article_timestamp <= self.last_sent_timestamp_unix_sec:
                        continue

                    parsed_article = self._parse_finnhub_article(raw_article)
                    
                    try:
                        key = parsed_article["ticker"].encode('utf-8') if parsed_article["ticker"] else str(parsed_article["id"]).encode('utf-8')
                        self.producer.produce(
                            topic=self.topic,
                            key=key,
                            value=json.dumps(parsed_article).encode('utf-8'),
                            callback=self._delivery_callback
                        )
                        self.producer.poll(0)
                        self.logger.info(f"✓ Sent: [{parsed_article['ticker']}] {parsed_article['headline'][:50]}... (ID: {article_id})")
                        
                        self.seen_ids.add(article_id)
                        self.last_sent_timestamp_unix_sec = max(self.last_sent_timestamp_unix_sec, article_timestamp)
                        new_articles_count += 1
                        
                    except Exception as e:
                        self.logger.error(f"✗ Failed to send news (ID: {article_id}): {e}", exc_info=True)
                        with self._stats_lock:
                            self.stats["errors"] += 1
                
                self.logger.info(f"Processed {new_articles_count} new articles. Next fetch in {self.poll_interval}s.")
                retries = 0
                
            except Exception as e:
                retries += 1
                if retries <= max_retries and self._running.is_set():
                    self.logger.error(f"✗ Fetch error (retry {retries}/{max_retries}): {e}. Retrying...", exc_info=True)
                    time.sleep(min(5 * retries, 60))
                else:
                    self.logger.error(f"✗ Max retries reached or shutdown. Exiting fetch loop.", exc_info=True)
                    self._running.clear()
                    break
            
            if self._running.is_set():
                time.sleep(self.poll_interval)


if __name__ == "__main__":
    load_dotenv()
    
    tickers = ["NVDA","MSFT","AAPL","GOOGL","AMZN","META","AVGO","TSLA"]
    logger = get_module_logger("FinnHubNewsProducer")

    redpanda_producer_config = {
        "bootstrap.servers": os.getenv("REDPANDA_BROKER"),
        "security.protocol": os.getenv("REDPANDA_SECURITY_PROTOCOL"),
        "sasl.mechanism": os.getenv("REDPANDA_SASL_MECHANISM"),
        "sasl.username": os.getenv("REDPANDA_USERNAME"),
        "sasl.password": os.getenv("REDPANDA_PASSWORD"),
        "acks": "all",
        "retries": 5,
        "linger.ms": 10,
        "batch.num.messages": 1000,
        "compression.type": "snappy",
    }
    
    finnhub_api_key = os.getenv("FINNHUB_API_KEY") 

    producer = FinnHubNewsProducer(
        logger=logger,
        topic=os.getenv("REDPANDA_NEWS_TOPIC", "finnhub_news"),
        producer_config=redpanda_producer_config,
        tickers=tickers,
        poll_interval=300,
        lookback_days=2,
        api_key=finnhub_api_key
    )
    
    logger.info("FinnHub News → Redpanda Producer starting...")
    logger.info("=" * 40)
    logger.info("Press Ctrl+C to stop.")
    logger.info("=" * 40)
    
    producer.run()