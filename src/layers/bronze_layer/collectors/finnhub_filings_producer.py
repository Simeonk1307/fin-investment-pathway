import os
import time
import datetime
import logging
import requests
import re
import finnhub
from bs4 import BeautifulSoup
from typing import Dict, List
from dotenv import load_dotenv

from src.layers.bronze_layer.base.base_producer import BaseProducer
from src.layers.bronze_layer.event_envelope import create_event_envelope
from src.utils.minio_storage import MinioStorage

# Load environment variables
load_dotenv()

_RETRY = (5, 10, 20)

class FinnhubFilingsProducer(BaseProducer):
    MAX_DEDUPE_CACHE = 50000
    MAX_CONSECUTIVE_ERRORS = 5
    BACKOFF = 30

    def __init__(
        self,
        logger,
        topic,
        producer_config,
        tickers=None,
        api_key=None,
        user_email="admin@example.com",
        poll_interval=600,
        lookback_days=2,
        max_retries=5,
        debug=False,
        debug_writer=None,
    ):
        super().__init__(logger, topic, producer_config)

        # 1. Ticker Loading Logic
        if tickers:
            self.tickers = tickers
        else:
            env_str = os.getenv("TICKERS", "")
            self.tickers = [t.strip() for t in env_str.split(",") if t.strip()]
        
        if not self.tickers:
            logger.critical("No tickers found in args or .env")
            raise ValueError("No tickers provided! Add 'TICKERS=AAPL,MSFT' to your .env file.")

        try:
            self.storage = MinioStorage()
        except Exception as e:
            logger.critical("Minio Storage initialisation failed")
            raise Exception("Minio Storage initialisation failed")


        self.poll_interval = poll_interval
        self.lookback_days = lookback_days
        self.max_retries = max_retries
        self.debug = debug
        self.debug_writer = debug_writer or (lambda *_: None)

        self.consecutive_errors = 0
        self.last_successful_fetch = time.time()

        # Finnhub Setup
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            logger.critical("Missing FINNHUB_API_KEY")
            raise ValueError("FINNHUB_API_KEY required")

        self.client = finnhub.Client(api_key=self.api_key)
        self.seen = set()

        self.sec_headers = {
            "User-Agent": f"FinnhubFilingBot/1.0 ({user_email})",
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov"
        }

        if debug:
            self.debug_writer("filings", "startup", {
                "topic": topic,
                "tickers": self.tickers,
                "poll_interval": poll_interval,
            })

        logger.info(f"Filings ready (topic={topic}, tickers={len(self.tickers)})")

    def _fetch(self, ticker):
        """Fetches metadata list from Finnhub (Lightweight)"""
        if self.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
            self.logger.error(f"Too many errors, pausing {self.BACKOFF}s")
            time.sleep(self.BACKOFF)
            self.consecutive_errors = 0

        today = datetime.date.today()
        start = today - datetime.timedelta(days=self.lookback_days)

        for r in range(self.max_retries):
            try:
                if r:
                    to = 10 + r * 5
                    self.client._session.timeout = (to, to + 10)

                fs = self.client.filings(
                    symbol=ticker,
                    _from=start.strftime("%Y-%m-%d"),
                    to=today.strftime("%Y-%m-%d"),
                )
                
                self.consecutive_errors = 0
                time.sleep(0.1) 
                return fs or []

            except Exception as e:
                self.consecutive_errors += 1
                if r < self.max_retries - 1:
                    time.sleep(_RETRY[min(r, 2)])
                    continue
                return []
        return []

    def _crawl_and_extract(self, url):
        """Visits SEC URL, strips HTML tags, cleans repetition."""
        if not url: return None
        
        try:
            with requests.Session() as s:
                resp = s.get(url, headers=self.sec_headers, timeout=10)
            
            if resp.status_code == 403:
                self.logger.error("SEC blocked request (403). Check User-Agent.")
                return None
            
            if resp.status_code != 200:
                return None

            soup = BeautifulSoup(resp.content, "html.parser")

            # 1. Clean useless tags
            for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "xbrl"]):
                tag.decompose()

            # 2. Extract text preserving structure (newlines)
            # This prevents "HeaderContent" merging into "Header Content"
            text_block = soup.get_text(separator='\n', strip=True)
            
            # 3. Deduplicate Lines
            lines = text_block.split('\n')
            cleaned_lines = []
            last_line = ""

            for line in lines:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                    
                # Skip Page Numbers (digits only)
                if line.isdigit():
                    continue

                # Skip consecutive duplicates (The main fix)
                if line == last_line:
                    continue
                
                cleaned_lines.append(line)
                last_line = line

            # 4. Join back into a single paragraph
            clean_text = ' '.join(cleaned_lines)
            
            # 5. Remove extra whitespace
            clean_text = re.sub(r'\s+', ' ', clean_text)

            return clean_text

        except Exception as e:
            self.logger.error(f"Failed to crawl {url}: {e}")
            return None

    def _parse(self, entry, ticker):
        acc = entry.get("accessNumber")
        form = entry.get("form")
        date = entry.get("filedDate")
        url = entry.get("reportUrl")

        if not acc or not form or not date or not url:
            return None

        if acc in self.seen:
            return None

        try:
            ts = int(datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
        except:
            ts = int(time.time() * 1000)

        # # Crawl for summary, but discard full content
        full_text = self._crawl_and_extract(url)
        # summary = (full_text[:2000] + "...") if full_text else "No content extracted"
        
        filename = f"{ticker}/{date}__{acc}__{form.replace('/', '_')}.txt"

        try:
            storage_url = self.storage.save_text(filename, full_text)
        except Exception as e:
            self.logger.error("Filings could not be stored")
            return {}

        return {
            "symbol": ticker,
            "timestamp": ts,
            "access_number": acc, 
            "form_type": form,
            "headline": f"{form} Filing for {ticker}",
            "url": url,
            "date": date,
            "storage_url": storage_url,
            "source": "SEC",
            "source_type": "filings",
        }

    def _publish(self, f):
        acc = f.get("access_number")
        
        if not acc or acc in self.seen:
            return False

        try:
            ev = create_event_envelope(f, source="finnhub", source_type="rest")
            ok = self.send(ev, key="finnhub")

            if ok:
                self.seen.add(acc)
                if len(self.seen) > self.MAX_DEDUPE_CACHE:
                    self.seen.pop()
                self.logger.info(f"[Sent] {f['headline']}")
                return True
            return False

        except Exception:
            return False

    def _run_loop(self):
        self.logger.info(f"Starting Loop for tickers: {self.tickers}")
        while self._running:
            try:
                for t in self.tickers:
                    if not self._running: break

                    filings = self._fetch(t)
                    filings.sort(key=lambda x: x.get("filedDate", ""))

                    for e in filings:
                        p = self._parse(e, t)
                        if p:
                            self._publish(p)
                            time.sleep(0.5) 

                time.sleep(self.poll_interval)

            except Exception as e:
                self.logger.error(f"Loop error: {e}")
                time.sleep(5)

    def stop(self):
        self._running = False
        self.logger.info("Stop")

if __name__ == "__main__":
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger("FinnhubCrawler")
    
    print("--- STARTING CRAWLER ---")

    try:
        producer = FinnhubFilingsProducer(
            logger=logger,
            topic="test_topic",
            producer_config={},
            tickers=None, 
            poll_interval=10,
            lookback_days=60, 
            user_email="student_demo@example.com"
        )

        producer.send = lambda env, key: print(f"   -> SENT: {env['data']['headline']}\n      PREVIEW: {env['data']['summary'][:150]}...") or True
        producer._running = True
        
        producer._run_loop()
    except ValueError as e:
        print(f"\nCONFIGURATION ERROR: {e}")
    except KeyboardInterrupt:
        producer.stop()