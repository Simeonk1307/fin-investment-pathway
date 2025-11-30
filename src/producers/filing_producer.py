import json
import time
import threading
import datetime
import re
from typing import List, Dict, Any, Optional
from confluent_kafka import Producer
import feedparser
from ..logger_config import get_module_logger
import os
from dotenv import load_dotenv

# TODO FOR FAZIL: CHANGE ALL DOTENV KEYS TO SETTINGS

class SecFilingsProducer:
    """
    SEC EDGAR RSS to Redpanda producer.
    Follows Unified Log architecture: Writes to specific partition (Default: 2).
    """
    
    def __init__(self, logger, topic: str, producer_config: Dict[str, Any], 
                 target_partition: int = 2, poll_interval: int = 60, 
                 user_agent: Optional[str] = None):
        
        self.logger = logger
        self.topic = topic
        self.target_partition = target_partition
        self.poll_interval = poll_interval
        
        # CRITICAL: SEC requires a valid User-Agent with email
        self.headers = {'User-Agent': user_agent or os.getenv("SEC_USER_AGENT", "StudentProject contact@example.com")}
        
        try:
            self.producer = Producer(producer_config)
            self.logger.info(f"Redpanda producer created (Target Partition: {self.target_partition})")
        except Exception as e:
            self.logger.error(f"Failed to create Redpanda producer: {e}")
            raise

        self.rss_url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=&company=&dateb=&owner=include&start=0&count=40&output=atom"
        self.seen_links: set[str] = set()
        
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

    def _extract_ticker(self, title: str) -> str:
        """Extracts ticker from SEC title format: 'Company Name [TICKER]'"""
        match = re.search(r'\[([A-Z]+)\]', title)
        return match.group(1) if match else "UNKNOWN"

    def _parse_entry(self, entry: Any) -> Dict[str, Any]:
        """Parses a single RSS entry."""
        full_title = entry.title
        # Logic to split "8-K - Apple Inc. [AAPL]"
        if " - " in full_title:
            parts = full_title.split(" - ", 1)
            filing_type = parts[0]
            company_info = parts[1] if len(parts) > 1 else "Unknown"
        else:
            filing_type = "Unknown"
            company_info = full_title

        ticker = self._extract_ticker(company_info)
        
        # Convert struct_time to timestamp
        try:
            published_dt = datetime.datetime(*entry.updated_parsed[:6])
            timestamp_ms = int(published_dt.timestamp() * 1000)
        except:
            timestamp_ms = int(time.time() * 1000)

        return {
            "source": "sec_edgar",
            "ticker": ticker,
            "company": company_info.replace(f"[{ticker}]", "").strip(),
            "form_type": filing_type,
            "headline": full_title,
            "link": entry.link,
            "timestamp_ms": timestamp_ms,
            "date": datetime.datetime.fromtimestamp(timestamp_ms/1000).strftime("%Y-%m-%d"),
        }

    def run(self):
        """Main runner with clean shutdown."""
        fetch_thread = threading.Thread(target=self._feed_loop, daemon=True)
        fetch_thread.start()
        
        try:
            while self._running.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.warning("!! Stopping on KeyboardInterrupt... !!")
            self._running.clear()
            time.sleep(2)
        
        # Final stats
        self.producer.flush(10)
        stats = self.get_stats()
        self.logger.info(f"Producer finished. Sent: {stats['sent']}, Errors: {stats['errors']}")

    def _feed_loop(self):
        """RSS Fetching loop."""
        retries = 0
        
        while self._running.is_set():
            try:
                self.logger.debug(f"Fetching SEC Feed...")
                feed = feedparser.parse(self.rss_url, request_headers=self.headers)
                
                new_count = 0
                
                # Process oldest first to maintain timeline natural order
                for entry in reversed(feed.entries):
                    if not self._running.is_set(): break
                    
                    if entry.link not in self.seen_links:
                        data = self._parse_entry(entry)
                        
                        # Produce to Redpanda
                        # KEY CHANGE: explicitly using partition=self.target_partition
                        self.producer.produce(
                            topic=self.topic,
                            key=data["ticker"].encode('utf-8'),
                            value=json.dumps(data).encode('utf-8'),
                            partition=self.target_partition, 
                            callback=self._delivery_callback
                        )
                        self.producer.poll(0)
                        
                        self.logger.info(f"✓ Sent: [{data['ticker']}] {data['form_type']}")
                        self.seen_links.add(entry.link)
                        new_count += 1
                
                if new_count > 0:
                    self.logger.info(f"Processed {new_count} new filings.")
                
                retries = 0
                
            except Exception as e:
                retries += 1
                self.logger.error(f"Fetch error: {e}")
                time.sleep(min(10 * retries, 60))

            # Wait for next poll (SEC limits are strict, default 60s is safe)
            if self._running.is_set():
                time.sleep(self.poll_interval)

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv(os.path.join("src", ".env"))
    
    logger = get_module_logger("SecFilingsProducer")

    sec_protocol = os.getenv("REDPANDA_SECURITY_PROTOCOL") or "PLAINTEXT"

    redpanda_producer_config = {
        "bootstrap.servers": os.getenv("REDPANDA_BROKER", "localhost:9092"),
        "security.protocol": sec_protocol,
        "acks": "all",
        "retries": 5
    }

    if sec_protocol != "PLAINTEXT":
        redpanda_producer_config.update({
            "sasl.mechanism": os.getenv("REDPANDA_SASL_MECHANISM", "SCRAM-SHA-256"),
            "sasl.username": os.getenv("REDPANDA_USERNAME"),
            "sasl.password": os.getenv("REDPANDA_PASSWORD"),
        })

    # Unified Log Topic Name
    topic_name = os.getenv("REDPANDA_UNIFIED_TOPIC", "raw_events")
    
    # User Agent
    my_user_agent = os.getenv("SEC_USER_AGENT", "MyStudentProject contact@example.edu")

    producer = SecFilingsProducer(
        logger=logger,
        topic=topic_name,
        producer_config=redpanda_producer_config,
        target_partition=2,  # Filings go to Partition 2
        poll_interval=60,
        user_agent=my_user_agent
    )
    
    logger.info("SEC Filings -> Redpanda Producer starting...")
    logger.info(f"Target: Topic '{topic_name}' @ Partition 2")
    logger.info("=" * 40)
    
    producer.run()