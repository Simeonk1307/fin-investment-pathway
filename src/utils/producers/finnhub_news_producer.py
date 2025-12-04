from dotenv import load_dotenv
import os, json, time, datetime, logging, finnhub
from typing import List, Dict, Optional
from src.utils.producers.base_producer import BaseProducer
from src.utils.event_envelope import create_event_envelope

load_dotenv()


class FinnHubNewsProducer(BaseProducer):
    def __init__(
        self,
        logger: logging.Logger,
        topic: str,
        producer_config: Dict,
        tickers: List[str],
        api_key: Optional[str] = None,
        poll_interval: int = 120,
        lookback_days: int = 1,
    ):
        super().__init__(logger=logger, topic=topic, producer_config=producer_config)

        self.tickers = tickers
        self.poll_interval = poll_interval
        self.lookback_days = lookback_days
        self._running = True

        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            self._count_error(self.misc_errors)
            self.logger.error("FINNHUB_API_KEY not found in environment or arguments")
            raise ValueError("FINNHUB_API_KEY missing")
        
        self.logger.debug("FINNHUB_API_KEY loaded successfully")

        try:
            self.logger.info("Initializing Finnhub client...")
            self.client = finnhub.Client(api_key=self.api_key)
            self.logger.info("Finnhub client initialized successfully")
        except Exception as e:
            self._count_error(self.initialisation_errors)
            self.logger.error(f"Finnhub client initialization failed: {e}")
            raise

        self.seen_ids = set()
        self.last_ts = 0
        
        self.logger.info("=" * 60)
        self.logger.info("FinnHub News Producer Configuration")
        self.logger.info(f"  Tickers        : {self.tickers}")
        self.logger.info(f"  Topic          : {self.topic}")
        self.logger.info(f"  Poll Interval  : {self.poll_interval}s")
        self.logger.info(f"  Lookback Days  : {self.lookback_days}")
        self.logger.info("=" * 60)


    def _fetch_news(self) -> List[Dict]:
        """Fetch news for all tickers from Finnhub API."""
        today = datetime.date.today()
        from_date = today - datetime.timedelta(days=self.lookback_days)
        
        self.logger.debug(f"Fetching news from {from_date} to {today}")
        
        all_articles = []
        fetch_start = time.time()

        for ticker in self.tickers:
            try:
                self.logger.debug(f"Fetching news for {ticker}...")
                ticker_start = time.time()
                
                articles = self.client.company_news(
                    ticker,
                    _from=from_date.strftime("%Y-%m-%d"),
                    to=today.strftime("%Y-%m-%d"),
                )
                
                ticker_duration = time.time() - ticker_start
                self.logger.debug(f"  {ticker}: {len(articles)} articles in {ticker_duration:.2f}s")
                
                all_articles.extend(articles)
                time.sleep(0.1)  # Rate limit throttle
                
            except finnhub.FinnhubAPIException as e:
                self._count_error(self.misc_errors)
                self.logger.warning(f"Finnhub API error for {ticker}: {e}")
            except Exception as e:
                self._count_error(self.misc_errors)
                self.logger.error(f"Unexpected fetch error for {ticker}: {e}")

        fetch_duration = time.time() - fetch_start
        self.logger.info(f"Fetched {len(all_articles)} total articles for {len(self.tickers)} tickers in {fetch_duration:.2f}s")
        
        return all_articles

    def _run_loop(self):
        """Main polling loop."""
        loop_count = 0
        total_published = 0
        
        self.logger.info("Starting news polling loop...")
        
        while self._running:
            loop_count += 1
            loop_start = time.time()
            
            self.logger.debug(f"Loop iteration #{loop_count} starting...")
            
            try:
                # Fetch articles
                articles = self._fetch_news()
                
                if not articles:
                    self.logger.debug("No articles returned from API")
                    time.sleep(self.poll_interval)
                    continue
                
                # Sort by timestamp
                articles.sort(key=lambda x: x.get("datetime", 0))
                self.logger.debug(f"Sorted {len(articles)} articles by datetime")
                
                # Process articles
                new_count = 0
                skipped_count = 0
                error_count = 0
                
                for art in articles:
                    art_id = art.get("id", 0)
                    ts = art.get("datetime", 0)
                    headline = art.get("headline", "")[:50]  # Truncate for logging

                    # Validation
                    if not art_id or not ts:
                        self.logger.debug(f"Skipping article with missing id/ts: {headline}...")
                        skipped_count += 1
                        continue
                    
                    # Deduplication
                    if art_id in self.seen_ids:
                        self.logger.debug(f"Skipping duplicate article id={art_id}")
                        skipped_count += 1
                        continue
                    
                    if ts <= self.last_ts:
                        self.logger.debug(f"Skipping old article ts={ts} <= last_ts={self.last_ts}")
                        skipped_count += 1
                        continue

                    # Publish
                    try:
                        data = create_event_envelope(
                            payload=art,
                            source="finnhub",
                            source_type="rest"
                        )
                        
                        self.producer.produce(
                            topic=self.topic,
                            key="finnhub",
                            value=json.dumps(data).encode(),
                            callback=self._delivery,
                        )
                        self.producer.poll(0)

                        # Update state
                        self.seen_ids.add(art_id)
                        self.last_ts = max(self.last_ts, ts)
                        new_count += 1
                        total_published += 1
                        
                        self.logger.debug(f"Published article id={art_id}: {headline}...")

                    except Exception as e:
                        self._count_error(self.misc_errors)
                        self.logger.error(f"Failed to publish article id={art_id}: {e}")
                        error_count += 1

                # Loop summary
                loop_duration = time.time() - loop_start
                self.logger.info(
                    f"Loop #{loop_count} completed in {loop_duration:.2f}s | "
                    f"New: {new_count} | Skipped: {skipped_count} | Errors: {error_count} | "
                    f"Total published: {total_published} | Seen IDs cached: {len(self.seen_ids)}"
                )
                
                # Wait for next poll
                self.logger.debug(f"Sleeping for {self.poll_interval}s...")
                time.sleep(self.poll_interval)

            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received, stopping...")
                self._running = False
                break
                
            except Exception as e:
                self._count_error(self.misc_errors)
                self.logger.error(f"Unexpected loop error: {e}", exc_info=True)
                self.logger.info("Retrying in 5s...")
                time.sleep(5)

        # Shutdown
        self.logger.info("=" * 60)
        self.logger.info("FinnHub News Producer Shutdown Summary")
        self.logger.info(f"  Total loops      : {loop_count}")
        self.logger.info(f"  Total published  : {total_published}")
        self.logger.info(f"  Cached IDs       : {len(self.seen_ids)}")
        self.logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def stop(self):
        """Signal graceful shutdown."""
        self.logger.info("Stop signal received...")
        self._running = False