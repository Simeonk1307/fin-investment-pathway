import os, time, datetime, logging, finnhub, requests
from bs4 import BeautifulSoup
from typing import Dict, List
from dotenv import load_dotenv

from src.layers.bronze_layer.base.base_producer import BaseProducer
from src.layers.bronze_layer.event_envelope import create_event_envelope

load_dotenv()

_RETRY = (5, 10, 20)


class FinnhubFilingsProducer(BaseProducer):
    MAX_DEDUPE_CACHE = 50000
    MAX_CONSECUTIVE_ERRORS = 5
    BACKOFF = 30
    SEC_BACKOFF = 60

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
        self.sec_errors = 0
        self.last_successful_fetch = time.time()

        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            logger.critical("Missing FINNHUB_API_KEY")
            raise ValueError("FINNHUB_API_KEY required")

        self.client = finnhub.Client(api_key=self.api_key)
        self.seen = set()
        self.headers = {"User-Agent": os.getenv("SEC_USER_AGENT", "example@example.com")}

        if debug:
            self.debug_writer("filings", "startup", {
                "topic": topic,
                "tickers": tickers,
                "poll_interval": poll_interval,
                "lookback_days": lookback_days,
            })

        logger.info(f"Filings ready (topic={topic}, tickers={len(tickers)})")

    def _http(self, url):
        if self.sec_errors >= 3:
            self.logger.error(f"SEC errors, pausing {self.SEC_BACKOFF}s")
            time.sleep(self.SEC_BACKOFF)
            self.sec_errors = 0

        for r in range(3):
            try:
                resp = requests.get(url, headers=self.headers, timeout=15)

                if resp.status_code == 200:
                    self.sec_errors = 0
                    return resp.text

                if resp.status_code == 429:
                    self.sec_errors += 1
                    time.sleep(min(30, 5 * (2 ** min(self.sec_errors, 4))))
                    continue

                if resp.status_code == 403:
                    self.sec_errors += 1
                    time.sleep(10)
                    continue

                if resp.status_code == 404:
                    return ""

                time.sleep(1)

            except Exception:
                self.sec_errors += 1
                if r < 2:
                    time.sleep(_RETRY[min(r, 2)])
                    continue

        self.logger.error(f"SEC fail {url}")
        return ""

    def _extract(self, url):
        html = self._http(url)
        if not html:
            return ""

        try:
            soup = BeautifulSoup(html, "html.parser")
            for t in soup(["script", "style"]):
                t.decompose()

            txt = soup.get_text(" ", strip=True)
            words = txt.split()
            if len(words) > 10000:
                return " ".join(words[:10000]) + "..."

            return txt

        except Exception:
            return ""

    def _fetch(self, ticker):
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
                time.sleep(0.1)
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

        text = self._extract(url)

        return {
            "symbol": ticker,
            "timestamp": ts,
            "form_type": form,
            "headline": f"{form} Filing for {ticker}",
            "content": text,
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
                self.logger.error("Loop error")
                time.sleep(5)

        self.logger.info(f"Stopped iters={i} sent={sent} err={errs}")

    def stop(self):
        self._running = False
        self.logger.info("Stop")
