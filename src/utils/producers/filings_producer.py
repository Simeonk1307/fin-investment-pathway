import json
import time
import datetime
import re
import os
import requests
import feedparser
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional
import logging

# Imports from your project structure
from src.utils.producers.base_producer import BaseProducer
from src.config.logger_config import get_module_logger
from dotenv import load_dotenv

class SecFilingsProducer(BaseProducer):
    def __init__(self, logger: logging.Logger, topic: str, producer_config: Dict,
                 target_partition: int = 2, poll_interval: int = 60,
                 user_agent: Optional[str] = None):
        
        # Initialize Base Class
        super().__init__(logger=logger, topic=topic, producer_config=producer_config)
        
        self.target_partition = target_partition
        self.poll_interval = poll_interval
        self._running = True
        
        # SEC Identity Header (Required to avoid 403 Forbidden)
        self.headers = {
            'User-Agent': user_agent or os.getenv("SEC_USER_AGENT", "StudentProject contact@example.com")
        }
        
        # RSS Feed URL (Excluding ownership reports for cleaner data)
        self.rss_url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=&company=&dateb=&owner=exclude&start=0&count=40&output=atom"
        self.seen_links = set()

    def _extract_ticker(self, title: str) -> str:
        """Extracts ticker symbol from title like '8-K - Apple Inc. [AAPL]'"""
        match = re.search(r'\[([A-Z]+)\]', title)
        return match.group(1) if match else "UNKNOWN"

    def _scrape_filing_text(self, index_url: str) -> str:
        """
        Two-step scraper:
        1. Visits the Index Page to find the real document link.
        2. Visits the Document Page to extract the actual text.
        """
        try:
            # Rate limit (SEC max 10 req/sec)
            time.sleep(0.15) 
            
            # --- STEP 1: Get the Index Page ---
            resp = requests.get(index_url, headers=self.headers, timeout=10)
            resp.raise_for_status()
            soup_index = BeautifulSoup(resp.content, 'html.parser')
            
            # Find the link to the primary document
            # It's usually in a table with class="tableFile", first row
            doc_link = None
            for row in soup_index.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) > 2:
                    # Look for the document link in the 3rd column (Document)
                    link_tag = cells[2].find('a')
                    if link_tag and 'href' in link_tag.attrs:
                        # Construct full URL (SEC links are relative)
                        doc_link = "https://www.sec.gov" + link_tag['href']
                        break
            
            if not doc_link:
                return f"Error: Could not find document link on index page: {index_url}"

            # --- STEP 2: Get the Actual Document ---
            # Rate limit again before second request
            time.sleep(0.15)
            
            resp_doc = requests.get(doc_link, headers=self.headers, timeout=10)
            resp_doc.raise_for_status()
            soup_doc = BeautifulSoup(resp_doc.content, 'html.parser')
            
            # Extract clean text
            text = soup_doc.get_text(separator=' ', strip=True)
            
            # Truncate to keep message size manageable (e.g., first 2000 chars)
            # Kafka default max message size is 1MB. 
            return text[:2000] + "..."
            
        except Exception as e:
            self.logger.warning(f"⚠ Scraping failed for {index_url}: {e}")
            return "Content unavailable"

    def _parse_to_schema(self, entry: Any) -> Dict[str, Any]:
        """
        Maps RSS data to your exact SecFilingsSchema fields.
        """
        full_title = entry.title
        
        # Split Title
        if " - " in full_title:
            parts = full_title.split(" - ", 1)
            filing_type = parts[0]
            company_info = parts[1] if len(parts) > 1 else "Unknown"
        else:
            filing_type = "Unknown"
            company_info = full_title

        ticker = self._extract_ticker(company_info)
        
        # Parse Time
        try:
            published_dt = datetime.datetime(*entry.updated_parsed[:6])
            timestamp_ms = int(published_dt.timestamp() * 1000)
        except:
            timestamp_ms = int(time.time() * 1000)

        # SCRAPE REAL CONTENT
        # We pass the link to our helper function to get the actual text
        content_text = self._scrape_filing_text(entry.link)
        
        # Return exact schema match
        return {
            "source": "sec_edgar",
            "ticker": ticker,
            "company": company_info.replace(f"[{ticker}]", "").strip(),
            "form_type": filing_type,
            "headline": full_title,
            "content": content_text,
            "link": entry.link,
            "time_ms": timestamp_ms,
            "date": datetime.datetime.fromtimestamp(timestamp_ms/1000).strftime("%Y-%m-%d"),
        }

    def _fetch_feed(self):
        try:
            self.logger.debug(f"Fetching SEC Feed...")
            return feedparser.parse(self.rss_url, request_headers=self.headers)
        except Exception as e:
            # Assuming BaseProducer has _count_error logic
            if hasattr(self, '_count_error'): self._count_error("fetch_error")
            self.logger.error(f"Feed fetch error: {e}")
            return None

    def _run_loop(self):
        """Main execution loop called by BaseProducer.run()"""
        while self._running:
            try:
                feed = self._fetch_feed()
                
                if feed and hasattr(feed, 'entries'):
                    new_count = 0
                    
                    for entry in reversed(feed.entries):
                        if not self._running: break
                        
                        if entry.link in self.seen_links:
                            continue

                        try:
                            # 1. Parse data into Schema format
                            schema_data = self._parse_to_schema(entry)
                            
                            # 2. Produce using BaseProducer's producer
                            self.producer.produce(
                                topic=self.topic,
                                key=schema_data["ticker"].encode('utf-8'),
                                value=json.dumps(schema_data).encode('utf-8'), 
                                partition=self.target_partition, 
                                callback=self._delivery 
                            )
                            self.producer.poll(0)
                            
                            self.seen_links.add(entry.link)
                            new_count += 1
                            self.logger.info(f"✓ Sent: [{schema_data['ticker']}] {schema_data['form_type']}")

                        except Exception as e:
                            if hasattr(self, '_count_error'): self._count_error("send_error")
                            self.logger.error(f"Send error for {entry.link}: {e}")

                    if new_count > 0:
                        self.logger.info(f"Processed {new_count} new filings.")
                
                time.sleep(self.poll_interval)

            except Exception as e:
                self.logger.error(f"Loop error: {e}")
                time.sleep(10)

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. Load Environment Variables
    if os.path.exists(".env"):
        load_dotenv(".env")
    elif os.path.exists("src/.env"):
        load_dotenv("src/.env")
    
    # 2. Setup Logger
    logger = get_module_logger("SecFilingsProducer")

    # 3. Setup Redpanda Config
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

    # 4. Initialize and Run
    topic_name = os.getenv("REDPANDA_EVENTS_TOPIC", "raw_events")
    
    producer = SecFilingsProducer(
        logger=logger,
        topic=topic_name,
        producer_config=redpanda_producer_config,
        target_partition=2,
        poll_interval=60
    )
    
    logger.info("SEC Filings -> Redpanda Producer starting...")
    producer.run()
# import json
# import time
# import datetime
# import re
# import os
# import logging
# import feedparser
# from typing import Dict, Any, Optional

# # Import the Base Class
# from src.utils.producers.base_producer import BaseProducer

# class SecFilingsProducer(BaseProducer):
#     def __init__(self, logger: logging.Logger, topic: str, producer_config: Dict,
#                  target_partition: int = 2, poll_interval: int = 60,
#                  user_agent: Optional[str] = None):
        
#         # 1. Initialize Base Class
#         super().__init__(logger=logger, topic=topic, producer_config=producer_config)
        
#         # 2. Set specific configs
#         self.target_partition = target_partition
#         self.poll_interval = poll_interval
#         self._running = True
        
#         # SEC Identity Header
#         self.headers = {
#             'User-Agent': user_agent or os.getenv("SEC_USER_AGENT", "StudentProject contact@example.com")
#         }
        
#         # Exclude ownership reports (Form 3/4) to focus on important filings
#         self.rss_url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=&company=&dateb=&owner=exclude&start=0&count=40&output=atom"
#         self.seen_links = set()

#     def _extract_ticker(self, title: str) -> str:
#         """Extracts ticker from SEC title format: 'Company Name [TICKER]'"""
#         match = re.search(r'\[([A-Z]+)\]', title)
#         return match.group(1) if match else "UNKNOWN"

#     def _parse_to_schema(self, entry: Any) -> Dict[str, Any]:
#         """
#         Parses RSS entry into the SecFilingsSchema format.
#         Schema Fields: source, ticker, company, form_type, headline, content, link, time_ms, date
#         """
#         full_title = entry.title
        
#         # Split Title: "8-K - Apple Inc. [AAPL]"
#         if " - " in full_title:
#             parts = full_title.split(" - ", 1)
#             filing_type = parts[0]
#             company_info = parts[1] if len(parts) > 1 else "Unknown"
#         else:
#             filing_type = "Unknown"
#             company_info = full_title

#         ticker = self._extract_ticker(company_info)
        
#         # Handle Time
#         try:
#             published_dt = datetime.datetime(*entry.updated_parsed[:6])
#             timestamp_ms = int(published_dt.timestamp() * 1000)
#         except:
#             timestamp_ms = int(time.time() * 1000)

#         # Handle Content
#         content_text = ""
#         if hasattr(entry, 'summary'):
#             content_text = entry.summary
        
#         return {
#             "source": "sec_edgar",
#             "ticker": ticker,
#             "company": company_info.replace(f"[{ticker}]", "").strip(),
#             "form_type": filing_type,
#             "headline": full_title,
#             "content": content_text,
#             "link": entry.link,
#             "time_ms": timestamp_ms,
#             "date": datetime.datetime.fromtimestamp(timestamp_ms/1000).strftime("%Y-%m-%d"),
#         }

#     def _fetch_feed(self):
#         """Helper to fetch and parse the RSS feed."""
#         try:
#             self.logger.debug(f"Fetching SEC Feed from {self.rss_url}...")
#             return feedparser.parse(self.rss_url, request_headers=self.headers)
#         except Exception as e:
#             self._count_error(self.misc_errors) # Assuming BaseProducer has this
#             self.logger.error(f"Feed fetch error: {e}")
#             return None

#     def _run_loop(self):
#         """Main execution loop called by BaseProducer.run()"""
#         while self._running:
#             try:
#                 feed = self._fetch_feed()
                
#                 if feed and hasattr(feed, 'entries'):
#                     new_count = 0
                    
#                     # Process oldest first to maintain timeline natural order
#                     for entry in reversed(feed.entries):
#                         if not self._running: break
                        
#                         # Deduplication
#                         if entry.link in self.seen_links:
#                             continue

#                         try:
#                             # 1. Parse data
#                             schema_data = self._parse_to_schema(entry)
                            
#                             # 2. Produce to Redpanda
#                             # Note: Using partition logic as per your Unified Log doc
#                             self.producer.produce(
#                                 topic=self.topic,
#                                 key=schema_data["ticker"].encode('utf-8'),
#                                 value=json.dumps(schema_data).encode('utf-8'), 
#                                 partition=self.target_partition, 
#                                 callback=self._delivery # Using BaseProducer's callback
#                             )
#                             self.producer.poll(0)
                            
#                             self.seen_links.add(entry.link)
#                             new_count += 1
#                             self.logger.info(f"✓ Sent: [{schema_data['ticker']}] {schema_data['form_type']}")

#                         except Exception as e:
#                             self._count_error(self.misc_errors)
#                             self.logger.error(f"Send error for {entry.link}: {e}")

#                     if new_count > 0:
#                         self.logger.info(f"Processed {new_count} new filings.")
                
#                 # Respect SEC rate limits (10 req/sec max)
#                 time.sleep(self.poll_interval)

#             except Exception as e:
#                 self._count_error(self.misc_errors)
#                 self.logger.error(f"Loop error: {e}")
#                 time.sleep(10)