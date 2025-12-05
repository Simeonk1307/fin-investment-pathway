import os, time, datetime, logging, finnhub, requests
from typing import List, Dict
from dotenv import load_dotenv
from src.layers.bronze_layer.base.base_producer import BaseProducer
from src.layers.bronze_layer.event_envelope import create_event_envelope

load_dotenv()
_RETRY = (5, 10, 20)
_TIMEOUT = ((10, 30), (15, 40), (20, 50))

class FinnHubNewsProducer(BaseProducer):
    MAX_DEDUPE_CACHE = 50000
    MAX_CONSECUTIVE_ERRORS = 5
    BACKOFF = 30

    def __init__(self, logger, topic, producer_config, tickers, api_key=None,
                 poll_interval=120, lookback_days=1, debug=False, debug_writer=None):
        super().__init__(logger, topic, producer_config)
        self.tickers = tickers
        self.poll_interval = poll_interval
        self.lookback_days = lookback_days
        self.debug = debug
        self.debug_writer = debug_writer or (lambda *_: None)
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            logger.critical("Missing FINNHUB_API_KEY")
            raise ValueError("FINNHUB_API_KEY required")
        self.client = finnhub.Client(api_key=self.api_key)
        self.seen_ids = set()
        self.consecutive_errors = 0
        self.last_successful_fetch = time.time()
        self._j = {t: 0.1 + 0.002 * (hash(t) % 100) for t in tickers}
        self._rj = {t: 0.5 * (hash(t) % 10) for t in tickers}
        if debug:
            self.debug_writer("news", "startup", {"topic": topic, "tickers": tickers})
        logger.info(f"Finnhub ready (topic={topic}, tickers={len(tickers)})")

    def _fetch_news(self):
        if self.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
            self.logger.error(f"Too many errors, pausing {self.BACKOFF}s")
            time.sleep(self.BACKOFF)
            self.consecutive_errors = 0

        today = datetime.date.today()
        f, t = (today - datetime.timedelta(days=self.lookback_days)).isoformat(), today.isoformat()

        all_articles, ok, failed = [], 0, []

        for ticker in self.tickers:
            if not self._running:
                break
            for r in range(3):
                try:
                    if r: self.client._session.timeout = _TIMEOUT[r]
                    arts = self.client.company_news(ticker, _from=f, to=t)
                    if arts:
                        all_articles.extend(arts)
                        ok += 1
                        self.logger.info(f"{ticker}: {len(arts)}")
                    if r: self.client._session.timeout = _TIMEOUT[0]
                    time.sleep(self._j[ticker])
                    break
                except (requests.exceptions.ReadTimeout,
                        requests.exceptions.ConnectTimeout,
                        requests.exceptions.ConnectionError):
                    self.consecutive_errors += 1
                    if r < 2:
                        time.sleep(_RETRY[r] + self._rj[ticker])
                        continue
                    failed.append(ticker)
                    self.logger.error(f"{ticker}: network")
                    break
                except finnhub.FinnhubAPIException as e:
                    self.consecutive_errors += 1
                    msg = str(e)
                    if "Invalid API key" in msg:
                        self.logger.critical("Invalid API key")
                        self.stop()
                        return []
                    time.sleep(30 if "Too many requests" in msg else 5)
                    break
                except Exception:
                    self.consecutive_errors += 1
                    if r < 2:
                        time.sleep(_RETRY[r])
                        continue
                    failed.append(ticker)
                    self.logger.error(f"{ticker}: error")
                    break

        if ok:
            self.consecutive_errors = 0
            self.last_successful_fetch = time.time()
        if all_articles:
            self.logger.info(f"Fetched {len(all_articles)} (ok={ok}, fail={len(failed)})")
        elif failed:
            self.logger.warning(f"No articles (fail={failed})")

        if len(self.seen_ids) > self.MAX_DEDUPE_CACHE:
            self.seen_ids = set()

        return all_articles

    def _publish_article(self, a):
        art_id = a.get("id")
        if not art_id or art_id in self.seen_ids:
            return False
        if not a.get("headline") or not a.get("url"):
            return False
        if not a.get("summary"):
            a["summary"] = a["headline"][:200] + "..."
        try:
            ev = create_event_envelope(a, source="finnhub", source_type="rest")
            if self.send(ev, key="finnhub"):
                self.seen_ids.add(art_id)
                return True
        except Exception:
            self.logger.error(f"Publish fail {art_id}")
        return False

    def _run_loop(self):
        i = sent = errs = 0
        self.logger.info("Loop start")
        while self._running:
            i += 1
            start = time.time()
            new = loop_err = 0
            try:
                arts = self._fetch_news()
                if arts:
                    arts.sort(key=lambda x: x.get("datetime", 0))
                    for a in arts:
                        try:
                            if self._publish_article(a):
                                new += 1
                        except Exception:
                            loop_err += 1
                sent += new
                errs += loop_err
                dur = time.time() - start
                if time.time() - self.last_successful_fetch > 3600:
                    self.logger.warning("1h no fetch")
                self.logger.info(f"iter={i} new={new} total={sent} err={errs} dur={dur:.2f}s")
                time.sleep(self.poll_interval)
            except Exception:
                errs += 1
                self.logger.error("Loop error")
                time.sleep(5)
        self.logger.info(f"Stopped iters={i} sent={sent} err={errs}")

    def stop(self):
        self._running = False
        self.logger.info("Stop")
