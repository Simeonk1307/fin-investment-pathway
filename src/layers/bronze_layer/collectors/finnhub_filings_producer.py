import os
import time
import datetime
import logging
import requests
import finnhub
from typing import Dict, List
from dotenv import load_dotenv

from src.layers.bronze_layer.base.base_producer import BaseProducer
from src.layers.bronze_layer.event_envelope import create_event_envelope

# Load environment variables immediately
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
        tickers,
        api_key=None,
        poll_interval=600,
        lookback_days=2,
        max_retries=5,
        debug=False,
        debug_writer=None,
    ):
        super().__init__(logger, topic, producer_config)

        self.tickers = tickers
        self.poll_interval = poll_interval
        self.lookback_days = lookback_days
        self.max_retries = max_retries
        self.debug = debug
        self.debug_writer = debug_writer or (lambda *_: None)

        self.consecutive_errors = 0
        self.last_successful_fetch = time.time()

        # Load API Key from Init arg OR Environment
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            logger.critical("Missing FINNHUB_API_KEY in environment or init args")
            raise ValueError("FINNHUB_API_KEY required")

        self.client = finnhub.Client(api_key=self.api_key)
        self.seen = set()

        if debug:
            self.debug_writer("filings", "startup", {
                "topic": topic,
                "tickers": tickers,
                "poll_interval": poll_interval,
                "lookback_days": lookback_days,
            })

        logger.info(f"Filings ready (topic={topic}, tickers={len(tickers)})")

    def _fetch(self, ticker):
        """Fetches filings list only (no content scraping)."""
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

                self.client._session.timeout = (10, 30)
                time.sleep(0.1) # Rate limit courtesy
                return fs or []

            except (requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectTimeout,
                    requests.exceptions.ConnectionError):
                self.consecutive_errors += 1
                if r < self.max_retries - 1:
                    time.sleep(_RETRY[min(r, 2)])
                    continue
                return []

            except finnhub.FinnhubAPIException as e:
                self.consecutive_errors += 1
                msg = str(e)

                if "Invalid API key" in msg:
                    self.stop()
                    return []

                if "Too many requests" in msg:
                    time.sleep(30 * self.consecutive_errors)
                    break

                if "API limit reached" in msg:
                    time.sleep(60)
                    break

                time.sleep(5 * self.consecutive_errors)
                break

            except Exception:
                self.consecutive_errors += 1
                if r < self.max_retries - 1:
                    time.sleep(_RETRY[min(r, 2)])
                    continue
                return []

        return []

    def _parse(self, entry, ticker):
        acc = entry.get("accessNumber")
        form = entry.get("form")
        date = entry.get("filedDate")
        url = entry.get("reportUrl")

        if not acc or not form or not date or not url:
            return None

        try:
            ts = int(datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
        except Exception:
            try:
                ts = int(datetime.datetime.strptime(date, "%Y-%m-%d").timestamp() * 1000)
            except Exception:
                ts = int(time.time() * 1000)

        return {
            "symbol": ticker,
            "timestamp": ts,
            "form_type": form,
            "headline": f"{form} Filing for {ticker}",
            "content": None,  # EXPLICITLY IGNORED
            "url": url,
            "date": date,
            "access_number": acc,
            "source": "SEC",
            "source_type": "filing",
        }

    def _publish(self, f):
        acc = f.get("access_number")
        form = f.get("form_type")

        if not acc or acc in self.seen:
            return False

        try:
            ev = create_event_envelope(f, source="finnhub", source_type="rest")
            ok = self.send(ev, key="finnhub")

            if ok:
                self.seen.add(acc)
                if len(self.seen) > self.MAX_DEDUPE_CACHE:
                    self.seen.pop()
                
                self.logger.info(f"[Filings:Send] access={acc} form={form}")
                return True

            self.logger.error(f"[Filings:SendFail] access={acc} form={form}")
            return False

        except Exception:
            self.logger.error(f"[Filings:SendError] access={acc}")
            return False

    def _run_loop(self):
        i = sent = errs = 0
        self.logger.info("Loop start")

        while self._running:
            i += 1
            start = time.time()
            new = 0

            try:
                for t in self.tickers:
                    if not self._running:
                        break

                    filings = self._fetch(t)
                    if not filings:
                        continue

                    filings.sort(key=lambda x: x.get("filedDate", ""))

                    for e in filings:
                        p = self._parse(e, t)
                        if p and self._publish(p):
                            new += 1

                sent += new
                dur = time.time() - start

                if self.debug:
                    self.debug_writer("filings", "loop", {
                        "iteration": i,
                        "new": new,
                        "total": sent,
                        "errors": errs,
                        "duration": dur,
                    })

                self.logger.info(
                    f"iter={i} new={new} total={sent} err={errs} dur={dur:.2f}s"
                )

                time.sleep(self.poll_interval)

            except Exception:
                errs += 1
                self.logger.error("Loop error", exc_info=True)
                time.sleep(5)

        self.logger.info(f"Stopped iters={i} sent={sent} err={errs}")

    def stop(self):
        self._running = False
        self.logger.info("Stop")


# ---------------------------------------------------------
# RUNNABLE MAIN BLOCK
# ---------------------------------------------------------
if __name__ == "__main__":
    import sys
    
    # 1. Setup Logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger("FinnhubEnvTest")
    
    print("--- STARTING ENV TEST ---")

    # 2. Get Tickers from ENV
    env_tickers = os.getenv("TICKERS")
    
    if not env_tickers:
        print("ERROR: 'TICKERS' not found in .env file.")
        print("Please add: TICKERS=AAPL,TSLA,MSFT to your .env")
        sys.exit(1)

    # Convert "AAPL,TSLA,MSFT" -> ["AAPL", "TSLA", "MSFT"]
    ticker_list = [t.strip() for t in env_tickers.split(",") if t.strip()]
    print(f"Loaded {len(ticker_list)} tickers from env: {ticker_list}")

    # 3. Instantiate Producer
    try:
        producer = FinnhubFilingsProducer(
            logger=logger,
            topic="test_topic",
            producer_config={}, 
            tickers=ticker_list,  # <-- Using the list from ENV
            poll_interval=10,    
            lookback_days=60,     # 60 days to ensure data is found
            debug=True
        )

        # 4. Mock Send (so it doesn't need Kafka)
        producer.send = lambda env, key: print(f"   [MOCK SEND] Found: {env['data']['headline']}") or True
        producer._running = True

        # 5. Run
        producer._run_loop()

    except KeyboardInterrupt:
        print("\nStopping...")
        producer.stop()
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")