from dotenv import load_dotenv
import os, json, time, datetime, logging, finnhub
from typing import List, Dict, Optional
from src.utils.producers.base_producer import BaseProducer

load_dotenv()

class FinnHubNewsProducer(BaseProducer):
    def __init__(self, logger: logging.Logger, topic: str, producer_config: Dict,
        tickers: List[str], api_key: Optional[str] = None,
        poll_interval: int = 120, lookback_days: int = 1,
    ):
        super().__init__(logger=logger, topic=topic, producer_config=producer_config)

        self.tickers = tickers
        self.poll_interval = poll_interval
        self.lookback_days = lookback_days
        self._running = True

        # API key
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            self._count_error(self.misc_errors)
            raise ValueError("FINNHUB_API_KEY missing")

        # Finnhub client init
        try:
            self.client = finnhub.Client(api_key=self.api_key)
            self.logger.info("Finnhub client initialization SUCCESS")
        except Exception as e:
            self._count_error(self.initialisation_errors)
            self.logger.error(f"Finnhub client initialization FAILURE: {e}")
            raise

        self.seen_ids = set()
        self.last_ts = 0

    def _fetch_news(self):
        today = datetime.date.today()
        from_date = today - datetime.timedelta(days=self.lookback_days)
        all_articles = []

        for ticker in self.tickers:
            try:
                articles = self.client.company_news(
                    ticker,
                    _from=from_date.strftime("%Y-%m-%d"),
                    to=today.strftime("%Y-%m-%d"),
                )
                all_articles.extend(articles)
                time.sleep(0.1)  # safety throttle
            except Exception as e:
                self._count_error(self.misc_errors)
                self.logger.error(f"Fetch error for {ticker}: {e}")

        return all_articles

    def _run_loop(self):
        while self._running:
            try:
                articles = self._fetch_news()
                articles.sort(key=lambda x: x.get("datetime", 0))

                for art in articles:
                    art_id = art.get("id", 0)
                    ts = art.get("datetime", 0)

                    # Bronze dedupe
                    if not art_id or not ts:
                        continue
                    if art_id in self.seen_ids or ts <= self.last_ts:
                        continue

                    try:
                        key = (art.get("related") or "").encode()

                        self.producer.produce(
                            topic=self.topic,
                            key=key,
                            value=json.dumps(art).encode(),
                            callback=self._delivery,
                        )
                        self.producer.poll(0)

                        self.seen_ids.add(art_id)
                        self.last_ts = max(self.last_ts, ts)

                    except Exception as e:
                        self._count_error(self.misc_errors)
                        self.logger.error(f"Send error: {e}")

                time.sleep(self.poll_interval)

            except Exception as e:
                self._count_error(self.misc_errors)
                self.logger.error(f"Loop error: {e}")
                time.sleep(5)
