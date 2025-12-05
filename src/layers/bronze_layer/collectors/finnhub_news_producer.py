import os
import time
import datetime
import logging
import finnhub
import requests
from typing import List, Dict
from dotenv import load_dotenv
from src.layers.bronze_layer.base.base_producer import BaseProducer
from src.layers.bronze_layer.event_envelope import create_event_envelope

load_dotenv()

_RETRY_DELAYS = (5, 10, 20)
_TIMEOUT_CONFIGS = ((10, 30), (15, 40), (20, 50))


class FinnHubNewsProducer(BaseProducer):
    MAX_DEDUPE_CACHE = 50_000
    MAX_CONSECUTIVE_ERRORS = 5
    ERROR_BACKOFF_SECONDS = 30

    def __init__(
        self,
        logger: logging.Logger,
        topic: str,
        producer_config: Dict,
        tickers: List[str],
        api_key: str | None = None,
        poll_interval: int = 120,
        lookback_days: int = 1,
        debug: bool = False,
        debug_writer=None,
    ) -> None:
        super().__init__(logger=logger, topic=topic, producer_config=producer_config)
        self.tickers = tickers
        self.poll_interval = poll_interval
        self.lookback_days = lookback_days
        self._running = True
        self.consecutive_errors = 0
        self.last_successful_fetch = time.time()
        self.debug = debug
        self.debug_writer = debug_writer or (lambda *_, **__: None)
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            logger.critical("FINNHUB_API_KEY missing")
            raise ValueError("FINNHUB_API_KEY required")
        self.client = finnhub.Client(api_key=self.api_key)
        self.seen_ids = set()
        self._ticker_jitter = {t: 0.1 + 0.002 * (hash(t) % 100) for t in tickers}
        self._retry_jitter = {t: 0.5 * (hash(t) % 10) for t in tickers}
        if debug:
            self.debug_writer("news", "startup", {
                "topic": topic, "tickers": tickers, "poll_interval": poll_interval,
                "lookback_days": lookback_days, "producer_config": producer_config,
            })
        logger.info("=" * 80)
        logger.info("[FinnHub] Initialized topic=%s tickers=%s poll=%ss lookback=%s debug=%s",
                    topic, tickers, poll_interval, lookback_days, debug)
        logger.info("=" * 80)

    def _fetch_news(self) -> List[Dict]:
        if self.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
            self.logger.error("Too many errors (%s), backoff %ss",
                              self.consecutive_errors, self.ERROR_BACKOFF_SECONDS)
            time.sleep(self.ERROR_BACKOFF_SECONDS)
            self.consecutive_errors = 0

        today = datetime.date.today()
        from_str = (today - datetime.timedelta(days=self.lookback_days)).isoformat()
        to_str = today.isoformat()

        all_articles = []
        successful = 0
        failed = []
        client = self.client
        logger = self.logger
        ticker_jitter = self._ticker_jitter
        retry_jitter = self._retry_jitter

        for ticker in self.tickers:
            if not self._running:
                break
            for retry in range(3):
                try:
                    if retry:
                        client._session.timeout = _TIMEOUT_CONFIGS[retry]
                    articles = client.company_news(ticker, _from=from_str, to=to_str)
                    if articles:
                        all_articles.extend(articles)
                        successful += 1
                        logger.info("[Fetch] %s: %d articles", ticker, len(articles))
                    if retry:
                        client._session.timeout = _TIMEOUT_CONFIGS[0]
                    time.sleep(ticker_jitter[ticker])
                    break
                except (requests.exceptions.ReadTimeout,
                        requests.exceptions.ConnectTimeout,
                        requests.exceptions.ConnectionError) as e:
                    self.consecutive_errors += 1
                    if retry < 2:
                        time.sleep(_RETRY_DELAYS[retry] + retry_jitter[ticker])
                        continue
                    failed.append(ticker)
                    logger.error("[Network] %s: %s", ticker, type(e).__name__)
                    break
                except finnhub.FinnhubAPIException as e:
                    self.consecutive_errors += 1
                    msg = str(e)
                    if "Invalid API key" in msg:
                        logger.critical("[Auth] Invalid API key")
                        self.stop()
                        return []
                    time.sleep(30 if "Too many requests" in msg else
                               60 if "API limit reached" in msg else
                               5 * self.consecutive_errors)
                    break
                except Exception as e:
                    self.consecutive_errors += 1
                    logger.error("[FetchError] %s: %s", ticker, e, exc_info=(retry == 2))
                    if retry < 2:
                        time.sleep(_RETRY_DELAYS[retry])
                        continue
                    failed.append(ticker)
                    break

        if successful:
            self.consecutive_errors = 0
            self.last_successful_fetch = time.time()

        if all_articles:
            logger.info("[Fetch] total=%d ok=%d fail=%d", len(all_articles), successful, len(failed))
        elif failed:
            logger.warning("[Fetch] No articles. Failed=%s", failed)

        if len(self.seen_ids) > self.MAX_DEDUPE_CACHE:
            self.seen_ids = set()

        return all_articles

    def _publish_article(self, article: Dict) -> bool:
        art_id = article.get("id")
        if not art_id or art_id in self.seen_ids:
            return False
        headline = article.get("headline")
        if not headline or not article.get("url"):
            return False
        if not article.get("summary"):
            article["summary"] = headline[:200] + "..."
        try:
            event = create_event_envelope(article, source="finnhub", source_type="rest")
            if self.send(event, key="finnhub"):
                self.seen_ids.add(art_id)
                return True
        except Exception as e:
            self.logger.error("[PublishError] %s: %s", art_id, e)
        return False

    def _run_loop(self) -> None:
        iteration = 0
        total_sent = 0
        total_errors = 0
        logger = self.logger
        poll_interval = self.poll_interval
        tickers_half = len(self.tickers) // 2

        logger.info("[Loop] Starting producer loop...")

        while self._running:
            iteration += 1
            start = time.time()
            new_events = 0
            loop_errors = 0

            try:
                articles = self._fetch_news()
                if articles:
                    articles.sort(key=lambda x: x.get("datetime", 0))
                    for a in articles:
                        try:
                            if self._publish_article(a):
                                new_events += 1
                        except Exception:
                            loop_errors += 1

                total_sent += new_events
                total_errors += loop_errors
                duration = time.time() - start

                if time.time() - self.last_successful_fetch > 3600:
                    logger.warning("[Health] No successful fetch in 1 hour")

                logger.info("[Loop] i=%d new=%d total=%d err=%d dur=%.2fs",
                            iteration, new_events, total_sent, total_errors, duration)

                time.sleep(min(60, poll_interval * 2) if loop_errors > tickers_half else poll_interval)

            except KeyboardInterrupt:
                self.stop()
            except SystemExit:
                raise
            except MemoryError:
                logger.critical("[MemoryError]")
                self.seen_ids = set()
                time.sleep(60)
            except Exception as e:
                total_errors += 1
                logger.error("[LoopError] %s", e, exc_info=True)
                time.sleep(min(300, 5 << min(total_errors, 6)))

        logger.info("[Shutdown] iters=%d sent=%d errors=%d", iteration, total_sent, total_errors)

    def stop(self) -> None:
        self._running = False
        self.logger.info("[Stop] Shutdown signal received.")