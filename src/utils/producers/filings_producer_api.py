import json
import time
import datetime
import os
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List, Optional
import logging
import finnhub

from src.utils.producers.base_producer import BaseProducer
from src.config.logger_config import get_module_logger
from dotenv import load_dotenv

class FinnhubFilingsProducer(BaseProducer):
    def __init__(self, logger: logging.Logger, topic: str, producer_config: Dict,
                 tickers: List[str], api_key: Optional[str] = None,
                 target_partition: int = 2, poll_interval: int = 600,
                 lookback_days: int = 2):
        
        super().__init__(logger=logger, topic=topic, producer_config=producer_config)
        
        self.tickers = tickers
        self.target_partition = target_partition
        self.poll_interval = poll_interval
        self.lookback_days = lookback_days
        self._running = True
        
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            raise ValueError("FINNHUB_API_KEY is missing! Check your .env file.")
            
        try:
            self.finnhub_client = finnhub.Client(api_key=self.api_key)
            self.logger.info("Finnhub client initialized successfully")
        except Exception as e:
            self.logger.error(f"Finnhub init failed: {e}")
            raise

        self.seen_accession_numbers = set()
        
        self.headers = {
            'User-Agent': os.getenv("SEC_USER_AGENT", "StudentProject contact@example.com")
        }

    def _scrape_filing_text(self, url: str) -> str:
        if not url: return ""
        try:
            time.sleep(0.2) 
            resp = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(resp.content, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            return text[:5000] + "..." 
        except Exception as e:
            self.logger.warning(f"⚠ Could not scrape {url}: {e}")
            return "Content Unavailable"

    def _parse_finnhub_entry(self, entry: Dict, ticker: str) -> Dict[str, Any]:
        form_type = entry.get('form', 'Unknown')
        report_url = entry.get('reportUrl', '')
        filed_date = entry.get('filedDate', datetime.datetime.now().strftime("%Y-%m-%d"))
        
        try:
            dt = datetime.datetime.strptime(filed_date, "%Y-%m-%d")
            time_ms = int(dt.timestamp() * 1000)
        except:
            time_ms = int(time.time() * 1000)

        content_text = self._scrape_filing_text(report_url)

        return {
            "source": "finnhub_sec",
            "ticker": ticker,
            "company": ticker, 
            "form_type": form_type,
            "headline": f"{form_type} Filing for {ticker}",
            "content": content_text,
            "link": report_url,
            "time_ms": time_ms,
            "date": filed_date
        }

    def _fetch_filings_for_ticker(self, ticker: str):
        today = datetime.date.today()
        from_date = today - datetime.timedelta(days=self.lookback_days)
        
        try:
            filings = self.finnhub_client.filings(
                symbol=ticker, 
                _from=from_date.strftime("%Y-%m-%d"), 
                to=today.strftime("%Y-%m-%d")
            )
            return filings
        except Exception as e:
            self.logger.error(f"Finnhub fetch error for {ticker}: {e}")
            return []

    def _run_loop(self):
        while self._running:
            total_new = 0
            
            for ticker in self.tickers:
                if not self._running: break
                
                filings = self._fetch_filings_for_ticker(ticker)
                
                if filings:
                    filings.reverse() 

                    for entry in filings:
                        acc_num = entry.get('accessionNumber')
                        
                        if acc_num in self.seen_accession_numbers:
                            continue
                            
                        schema_data = self._parse_finnhub_entry(entry, ticker)
                        
                        try:
                            # Use partition logic OR default to None (if using specific topics)
                            self.producer.produce(
                                topic=self.topic,
                                key=ticker.encode('utf-8'),
                                value=json.dumps(schema_data).encode('utf-8'),
                                partition=self.target_partition if self.target_partition is not None else -1,
                                callback=self._delivery
                            )
                            self.producer.poll(0)
                            
                            self.logger.info(f"✓ Sent: [{ticker}] {schema_data['form_type']}")
                            self.seen_accession_numbers.add(acc_num)
                            total_new += 1
                            
                        except Exception as e:
                            self.logger.error(f"Send error: {e}")
                
                time.sleep(1) 

            if total_new > 0:
                self.logger.info(f"Processed {total_new} new filings from Finnhub.")
            
            time.sleep(self.poll_interval)

if __name__ == "__main__":
    # --- FIXED ENV LOADING ---
    # Since you run from root, just load .env directly
    if os.path.exists(".env"):
        load_dotenv(".env")
        print("Loaded .env from root directory")
    else:
        print("Warning: No .env file found in root directory!")

    logger = get_module_logger("FinnhubFilingsProducer")

    # Load Tickers
    tickers_env = os.getenv("TICKERS", "[]")
    try:
        TICKERS = json.loads(tickers_env)
        if not TICKERS:
            logger.warning("TICKERS list is empty in .env! Using default list.")
            TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]
    except json.JSONDecodeError:
        logger.error("Failed to parse TICKERS from .env! Using default list.")
        TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]

    redpanda_producer_config = {
        "bootstrap.servers": os.getenv("REDPANDA_BROKERS", "localhost:9092"),
        "security.protocol": os.getenv("REDPANDA_SECURITY_PROTOCOL", "PLAINTEXT"),
        "sasl.mechanism": os.getenv("REDPANDA_SASL_MECHANISM", "SCRAM-SHA-256"),
        "sasl.username": os.getenv("REDPANDA_USERNAME"),
        "sasl.password": os.getenv("REDPANDA_PASSWORD"),
        "acks": "all",
        "retries": 5
    }
    redpanda_producer_config = {k: v for k, v in redpanda_producer_config.items() if v}

    topic_name = os.getenv("REDPANDA_BRONZE_FILINGS_TOPIC", "bronze.filings")
    
    producer = FinnhubFilingsProducer(
        logger=logger,
        topic=topic_name,
        producer_config=redpanda_producer_config,
        tickers=TICKERS,
        target_partition=None, # Let Redpanda decide
        poll_interval=600 
    )
    
    logger.info("Finnhub Filings -> Redpanda Producer starting...")
    producer.run()