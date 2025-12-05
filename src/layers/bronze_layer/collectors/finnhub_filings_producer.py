import json, time, datetime, os, requests, logging, finnhub
from bs4 import BeautifulSoup
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from src.layers.bronze_layer.base.base_producer import BaseProducer
from src.layers.bronze_layer.event_envelope import create_event_envelope

load_dotenv()


class FinnhubFilingsProducer(BaseProducer):
    MAX_DEDUPE_CACHE = 50000
    MAX_CONSECUTIVE_ERRORS = 5
    ERROR_BACKOFF_SECONDS = 30
    MAX_CONSECUTIVE_SEC_ERRORS = 3
    SEC_ERROR_BACKOFF_SECONDS = 60

    def __init__(
        self,
        logger: logging.Logger,
        topic: str,
        producer_config: Dict,
        tickers: List[str],
        api_key: Optional[str] = None,
        poll_interval: int = 600,
        lookback_days: int = 2,
        max_retries: int = 5,
    ):
        super().__init__(logger=logger, topic=topic, producer_config=producer_config)
        self.tickers = tickers
        self.poll_interval = poll_interval
        self.lookback_days = lookback_days
        self.max_retries = max_retries
        self._running = True
        self.consecutive_errors = 0
        self.consecutive_sec_errors = 0
        self.last_successful_fetch = time.time()
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            logger.critical("[Filings:Config] FINNHUB_API_KEY missing")
            raise ValueError("FINNHUB_API_KEY required")
        self.client = finnhub.Client(api_key=self.api_key)
        self.seen_access_numbers = set()
        self.headers = {"User-Agent": os.getenv("SEC_USER_AGENT", "student.project@example.com")}
        self.logger.info("=" * 80)
        self.logger.info("[Filings:INIT] Producer initialized")
        self.logger.info("  Topic         : %s", topic)
        self.logger.info("  Tickers       : %s", tickers)
        self.logger.info("  Poll Interval : %s sec", poll_interval)
        self.logger.info("  Lookback Days : %s", lookback_days)
        self.logger.info("  Max Retries   : %s", max_retries)
        self.logger.info("=" * 80)

    def _safe_http_get(self, url: str, retries: int = 3, timeout: float = 10.0):
        if not url:
            return None
        if self.consecutive_sec_errors >= self.MAX_CONSECUTIVE_SEC_ERRORS:
            self.logger.error("[Filings:HTTP] Too many consecutive SEC errors (%s), backing off for %s seconds", self.consecutive_sec_errors, self.SEC_ERROR_BACKOFF_SECONDS)
            time.sleep(self.SEC_ERROR_BACKOFF_SECONDS)
            self.consecutive_sec_errors = 0
        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(url, headers=self.headers, timeout=timeout)
                if resp.status_code == 200:
                    self.consecutive_sec_errors = 0
                    return resp
                if resp.status_code == 429:
                    self.consecutive_sec_errors += 1
                    backoff = min(30, 5 * (2 ** min(self.consecutive_sec_errors, 4)))
                    time.sleep(backoff)
                    continue
                if resp.status_code == 403:
                    self.consecutive_sec_errors += 1
                    time.sleep(10)
                    continue
                if resp.status_code == 404:
                    return None
                time.sleep(1 * attempt)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                self.consecutive_sec_errors += 1
                if attempt < retries:
                    time.sleep(2 * attempt)
                continue
            except Exception:
                self.consecutive_sec_errors += 1
                if attempt < retries:
                    time.sleep(2 * attempt)
                continue
        return None

    def _extract_filing_text(self, url: str) -> str:
        if not url:
            return "Content unavailable (empty URL)."
        resp = self._safe_http_get(url, retries=4, timeout=15)
        if not resp or not resp.text:
            return "Content unavailable (network error)."
        try:
            soup = BeautifulSoup(resp.text, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator=" ", strip=True)
            words = text.split()
            if len(words) > 10000:
                text = " ".join(words[:10000]) + "... [truncated]"
            return text if text else "Content empty after parsing."
        except Exception:
            return "Content unavailable (parse error)."

    def _fetch_filings_for_ticker(self, ticker: str):
        if self.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
            self.logger.error("[Filings:Fetch] Too many consecutive errors (%s), backing off for %s seconds", self.consecutive_errors, self.ERROR_BACKOFF_SECONDS)
            time.sleep(self.ERROR_BACKOFF_SECONDS)
            self.consecutive_errors = 0
        today = datetime.date.today()
        from_date = today - datetime.timedelta(days=self.lookback_days)
        for attempt in range(1, self.max_retries + 1):
            try:
                if attempt > 1:
                    timeout_increase = min(30, 5 * attempt)
                    self.client._session.timeout = (10 + timeout_increase, 30 + timeout_increase)
                filings = self.client.filings(symbol=ticker, _from=from_date.strftime("%Y-%m-%d"), to=today.strftime("%Y-%m-%d"))
                self.client._session.timeout = (10, 30)
                time.sleep(0.2 + (0.05 * (hash(ticker) % 10) / 10))
                return filings if filings else []
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as e:
                self.consecutive_errors += 1
                if attempt < self.max_retries:
                    backoff = 5 * (2 ** (attempt - 1)) + (0.5 * (hash(ticker) % 10))
                    time.sleep(backoff)
                    continue
                return []
            except finnhub.FinnhubAPIException as e:
                self.consecutive_errors += 1
                if "Too many requests" in str(e):
                    time.sleep(30 * self.consecutive_errors)
                    break
                elif "Invalid API key" in str(e):
                    self.stop()
                    return []
                elif "API limit reached" in str(e):
                    time.sleep(60)
                    break
                else:
                    time.sleep(5 * self.consecutive_errors)
                    break
            except Exception:
                self.consecutive_errors += 1
                if attempt < self.max_retries:
                    backoff = 5 * (2 ** (attempt - 1)) + (0.5 * (hash(ticker) % 10))
                    time.sleep(backoff)
                    continue
                return []
        return []

    def _parse_filing_entry(self, entry: Dict[str, Any], ticker: str):
        required_fields = ["accessNumber", "form", "filedDate", "reportUrl"]
        missing_required = [field for field in required_fields if not entry.get(field)]
        if missing_required:
            return None
        access_number = entry["accessNumber"]
        form_type = entry["form"]
        filed_date = entry["filedDate"]
        url = entry["reportUrl"]
        timestamp_ms = int(time.time() * 1000)
        try:
            dt = datetime.datetime.strptime(filed_date, "%Y-%m-%d %H:%M:%S")
            timestamp_ms = int(dt.timestamp() * 1000)
        except ValueError:
            try:
                dt = datetime.datetime.strptime(filed_date, "%Y-%m-%d")
                timestamp_ms = int(dt.timestamp() * 1000)
            except ValueError:
                pass
        filing_text = self._extract_filing_text(url)
        return {
            "symbol": ticker,
            "timestamp": timestamp_ms,
            "form_type": form_type,
            "headline": f"{form_type} Filing for {ticker}",
            "content": filing_text,
            "url": url,
            "date": filed_date,
            "access_number": access_number,
            "source": "SEC",
            "source_type": "filing",
        }

    def _publish_filing(self, filing_data: Dict[str, Any], ticker: str) -> bool:
        access_number = filing_data.get("access_number")
        if not access_number or access_number in self.seen_access_numbers:
            return False
        try:
            event = create_event_envelope(payload=filing_data, source="finnhub", source_type="rest")
            if self.send(event, key=ticker):
                self.seen_access_numbers.add(access_number)
                self.logger.info("[Filings:Publish] Success ticker=%s form=%s access_number=%s", ticker, filing_data.get("form_type", "Unknown"), access_number)
                return True
            return False
        except Exception:
            return False

    def _run_loop(self):
        iteration = 0
        total_sent = 0
        total_errors = 0
        self.logger.info("[Filings:Loop] Starting producer loop...")
        while self._running:
            iteration += 1
            start_time = time.time()
            new_events = 0
            loop_errors = 0
            successful_fetches = 0
            try:
                for ticker in self.tickers:
                    if not self._running:
                        break
                    filings = self._fetch_filings_for_ticker(ticker)
                    if filings:
                        successful_fetches += 1
                        filings.sort(key=lambda x: x.get("filedDate", ""))
                        for entry in filings:
                            filing_data = self._parse_filing_entry(entry, ticker)
                            if filing_data and self._publish_filing(filing_data, ticker):
                                new_events += 1
                                total_sent += 1
                if successful_fetches > 0:
                    self.consecutive_errors = 0
                    self.last_successful_fetch = time.time()
                if len(self.seen_access_numbers) > self.MAX_DEDUPE_CACHE:
                    self.seen_access_numbers = set()
                loop_duration = time.time() - start_time
                if time.time() - self.last_successful_fetch > 3600:
                    self.logger.warning("[Filings:Health] No successful fetch in last hour")
                self.logger.info("[Filings:Loop] iter=%s new=%s total=%s errors=%s total_errors=%s duration=%.2fs", iteration, new_events, total_sent, loop_errors, total_errors, loop_duration)
                if loop_errors > len(self.tickers) // 2:
                    extra_sleep = min(300, self.poll_interval * 2)
                    time.sleep(extra_sleep)
                else:
                    time.sleep(self.poll_interval)
            except KeyboardInterrupt:
                self.stop()
            except SystemExit:
                raise
            except MemoryError:
                self.seen_access_numbers = set()
                time.sleep(60)
            except Exception:
                total_errors += 1
                backoff_time = min(300, 5 * (2 ** min(total_errors, 6)))
                time.sleep(backoff_time)
        self.logger.info("[Filings:Shutdown] Producer stopped. Stats: iterations=%s total_sent=%s total_errors=%s", iteration, total_sent, total_errors)

    def stop(self):
        self.logger.info("[Filings:Stop] Shutdown signal received.")
        self._running = False