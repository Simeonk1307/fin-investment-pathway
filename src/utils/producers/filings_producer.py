import json, os, datetime, re, logging, time
from typing import Dict, Any, Optional
import feedparser
from dotenv import load_dotenv
from src.utils.producers.base_producer import BaseProducer

load_dotenv()
# TODO FOR FAZIL: CHANGE ALL DOTENV KEYS TO SETTINGS

class SecFilingsProducer(BaseProducer):
    def __init__(self, logger: logging.Logger, topic: str, producer_config: Dict[str, Any], 
                poll_interval: int = 60, user_agent: Optional[str] = None
    ):
        
        super().__init__(logger=logger, topic=topic, producer_config=producer_config)
        self.poll_interval = poll_interval
        self.headers = {'User-Agent': user_agent or os.getenv("SEC_USER_AGENT", "StudentProject contact@example.com")}
        self.rss_url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=&company=&dateb=&owner=exclude&start=0&count=40&output=atom"
        self.seen_links = set()

    def _extract_ticker(self, title: str) -> str:
        """Extracts ticker from SEC title format: 'Company Name [TICKER]'"""
        match = re.search(r'\[([A-Z]+)\]', title)
        return match.group(1) if match else "UNKNOWN"

    def _parse(self, entry: Any) -> Dict[str, Any]:
        """Parses a single RSS entry into a raw data dictionary."""
        full_title = entry.title
        if " - " in full_title:
            parts = full_title.split(" - ", 1)
            filing_type = parts[0]
            company_info = parts[1] if len(parts) > 1 else "Unknown"
        else:
            filing_type = "Unknown"
            company_info = full_title

        ticker = self._extract_ticker(company_info)
        
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

    def _run_loop(self):
        retries = 0
        
        while self._running.is_set():
            try:
                self.logger.debug(f"Fetching SEC Feed...")
                feed = feedparser.parse(self.rss_url, request_headers=self.headers)
                
                new_count = 0
                
                # Process oldest first to maintain timeline natural order
                for entry in reversed(feed.entries):
                    if not self._running: 
                        break
                    
                    if entry.link not in self.seen_links:
                        data = self._parse(entry)

                        self.producer.produce(
                            topic=self.topic,
                            value=json.dumps(data).encode('utf-8'), 
                            callback=self._delivery
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
                self._count_error(self.misc_errors)
                self.logger.error(f"Fetch error: {e}")
                time.sleep(min(10 * retries, 60))

            if self._running:
                time.sleep(self.poll_interval)