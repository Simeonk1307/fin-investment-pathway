import os, json, time, datetime, logging, finnhub
import signal, sys
from typing import List, Dict
from src.layers.bronze_layer.base.base_producer import BaseProducer
from src.layers.bronze_layer.event_envelope import create_event_envelope
from dotenv import load_dotenv
import requests

load_dotenv()


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
        api_key: str = None,
        poll_interval: int = 120,
        lookback_days: int = 1,
    ):
        super().__init__(logger=logger, topic=topic, producer_config=producer_config)
        
        self.tickers = tickers
        self.poll_interval = poll_interval
        self.lookback_days = lookback_days
        self._running = True
        self.consecutive_errors = 0
        self.last_successful_fetch = time.time()

        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            logger.critical("[FinnHub:Config] FINNHUB_API_KEY missing")
            raise ValueError("FINNHUB_API_KEY required")
        
        self.client = finnhub.Client(api_key=self.api_key)
        self.seen_ids = set()
        self.last_ts = 0

        self.logger.info("=" * 80)
        self.logger.info("[FinnHub] Producer initialized")
        self.logger.info("  Topic         : %s", topic)
        self.logger.info("  Tickers       : %s", tickers)
        self.logger.info("  Poll Interval : %s sec", poll_interval)
        self.logger.info("  Lookback Days : %s", lookback_days)
        self.logger.info("=" * 80)


    # -------------------------------------------------------------------------
    # Fetch news for all tickers with enhanced error handling
    # -------------------------------------------------------------------------
    def _fetch_news(self) -> List[Dict]:
        if self.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
            self.logger.error(
                "[FinnHub:Fetch] Too many consecutive errors (%s), backing off for %s seconds",
                self.consecutive_errors, self.ERROR_BACKOFF_SECONDS
            )
            time.sleep(self.ERROR_BACKOFF_SECONDS)
            self.consecutive_errors = 0

        today = datetime.date.today()
        from_date = today - datetime.timedelta(days=self.lookback_days)
        from_str = from_date.strftime("%Y-%m-%d")
        to_str = today.strftime("%Y-%m-%d")
        
        all_articles = []
        successful_fetches = 0
        failed_tickers = []
        
        for ticker in self.tickers:
            if not self._running:  # Changed from self.shutting_down
                break
                
            max_retries = 3
            retry_delay = 5
            
            for retry in range(max_retries):
                try:
                    # Configure session with longer timeouts for retries
                    if retry > 0:
                        # Increase timeout on retries
                        self.client._session.timeout = (10 + (retry * 5), 30 + (retry * 10))
                        self.logger.debug(
                            "[FinnHub:Retry] Attempt %s/%s for ticker=%s with timeout=%s",
                            retry + 1, max_retries, ticker, self.client._session.timeout
                        )
                    
                    articles = self.client.company_news(ticker, _from=from_str, to=to_str)
                    
                    if articles:
                        all_articles.extend(articles)
                        successful_fetches += 1
                        self.logger.info(
                            "[FinnHub:Fetch] ticker=%s articles=%s retry=%s",
                            ticker, len(articles), retry if retry > 0 else "initial"
                        )
                    else:
                        self.logger.debug(
                            "[FinnHub:Fetch] ticker=%s no articles found",
                            ticker
                        )
                    
                    # Reset timeout to default after successful request
                    self.client._session.timeout = (10, 30)
                    
                    # Rate limiting with jitter
                    time.sleep(0.1 + (0.02 * (hash(ticker) % 10) / 10))
                    break  # Success, exit retry loop
                    
                except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as e:
                    self.consecutive_errors += 1
                    
                    if retry < max_retries - 1:
                        # Exponential backoff with jitter
                        backoff = retry_delay * (2 ** retry) + (0.5 * (hash(ticker) % 10))
                        self.logger.warning(
                            "[FinnHub:Timeout] ticker=%s attempt=%s/%s error=%s backing off %s seconds",
                            ticker, retry + 1, max_retries, type(e).__name__, round(backoff, 1)
                        )
                        time.sleep(backoff)
                        continue
                    else:
                        failed_tickers.append(ticker)
                        self.logger.error(
                            "[FinnHub:Timeout] ticker=%s failed after %s attempts error=%s",
                            ticker, max_retries, e
                        )
                        break
                        
                except requests.exceptions.ConnectionError as e:
                    self.consecutive_errors += 1
                    
                    if retry < max_retries - 1:
                        backoff = retry_delay * (2 ** retry) + (0.5 * (hash(ticker) % 10))
                        self.logger.warning(
                            "[FinnHub:Connection] ticker=%s attempt=%s/%s error=%s backing off %s seconds",
                            ticker, retry + 1, max_retries, type(e).__name__, round(backoff, 1)
                        )
                        time.sleep(backoff)
                        continue
                    else:
                        failed_tickers.append(ticker)
                        self.logger.error(
                            "[FinnHub:Connection] ticker=%s failed after %s attempts error=%s",
                            ticker, max_retries, e
                        )
                        break
                        
                except finnhub.FinnhubAPIException as e:
                    self.consecutive_errors += 1
                    
                    if "Too many requests" in str(e):
                        self.logger.warning(
                            "[FinnHub:RateLimit] ticker=%s consecutive_errors=%s",
                            ticker, self.consecutive_errors
                        )
                        time.sleep(30 * self.consecutive_errors)  # Exponential backoff
                        break  # Don't retry rate limits immediately
                    elif "Invalid API key" in str(e):
                        self.logger.critical(
                            "[FinnHub:Auth] Invalid API key for ticker=%s",
                            ticker
                        )
                        self.stop()
                        return []
                    elif "API limit reached" in str(e):
                        self.logger.error(
                            "[FinnHub:Limit] API limit reached for ticker=%s",
                            ticker
                        )
                        time.sleep(60)
                        break  # Don't retry immediately
                    else:
                        self.logger.warning(
                            "[FinnHub:API] ticker=%s error=%s consecutive_errors=%s",
                            ticker, e, self.consecutive_errors
                        )
                        time.sleep(5 * self.consecutive_errors)
                        break  # Don't retry other API errors
                        
                except Exception as e:
                    self.consecutive_errors += 1
                    self.logger.error(
                        "[FinnHub:FetchError] ticker=%s error=%s consecutive_errors=%s",
                        ticker, e, self.consecutive_errors,
                        exc_info=(retry == max_retries - 1)  # Full traceback only on final failure
                    )
                    
                    if retry < max_retries - 1:
                        backoff = retry_delay * (2 ** retry) + (0.5 * (hash(ticker) % 10))
                        time.sleep(backoff)
                        continue
                    else:
                        failed_tickers.append(ticker)
                        break
        
        # Reset error counter if we had successful fetches
        if successful_fetches > 0:
            self.consecutive_errors = 0
            self.last_successful_fetch = time.time()
        
        # Log summary
        if all_articles:
            self.logger.info(
                "[FinnHub:Fetch] total_articles=%s successful=%s/%s failed=%s",
                len(all_articles), successful_fetches, len(self.tickers), len(failed_tickers)
            )
        else:
            if failed_tickers:
                self.logger.warning(
                    "[FinnHub:Fetch] No articles found. Failed tickers: %s",
                    failed_tickers
                )
            else:
                self.logger.warning(
                    "[FinnHub:Fetch] No articles found for any ticker"
                )
            
        # Dedupe cache safety reset
        if len(self.seen_ids) > self.MAX_DEDUPE_CACHE:
            self.logger.warning(
                "[FinnHub:Dedupe] Cache exceeded threshold (%s). Resetting.",
                self.MAX_DEDUPE_CACHE
            )
            self.seen_ids = set()
            
        return all_articles

    # -------------------------------------------------------------------------
    # Publish single article with enhanced error handling
    # -------------------------------------------------------------------------
    def _publish_article(self, article: Dict) -> bool:
        art_id = article.get("id")
        ts = article.get("datetime", 0)
        
        if not art_id:
            self.logger.warning(
                "[FinnHub:Publish] Article missing ID, skipping"
            )
            return False
        
        if not ts:
            self.logger.warning(
                "[FinnHub:Publish] Article missing timestamp art_id=%s",
                art_id
            )
            return False
        
        if art_id in self.seen_ids:
            self.logger.debug(
                "[FinnHub:Publish] Duplicate article art_id=%s",
                art_id
            )
            return False
        
        if ts <= self.last_ts:
            self.logger.debug(
                "[FinnHub:Publish] Old article art_id=%s ts=%s last_ts=%s",
                art_id, ts, self.last_ts
            )
            return False
        
        try:
            # Validate article structure - only require essential fields
            required_fields = ["headline", "url"]
            missing_required = [field for field in required_fields if not article.get(field)]
            
            if missing_required:
                self.logger.warning(
                    "[FinnHub:Publish] Missing required fields art_id=%s missing=%s",
                    art_id, missing_required
                )
                return False
            
            # Check for optional fields and log at debug level if missing
            optional_fields = ["summary", "source", "image", "related"]
            missing_optional = [field for field in optional_fields if not article.get(field)]
            
            if missing_optional:
                self.logger.debug(
                    "[FinnHub:Publish] Missing optional fields art_id=%s missing=%s",
                    art_id, missing_optional
                )
            
            # Add placeholder for missing summary if needed
            if not article.get("summary"):
                article["summary"] = article.get("headline", "")[:200] + "..."
            
            event = create_event_envelope(
                payload=article,
                source="finnhub",
                source_type="rest"
            )
            
            if self.send(event, key=str(art_id)):
                self.seen_ids.add(art_id)
                self.last_ts = max(self.last_ts, ts)
                self.logger.debug(
                    "[FinnHub:Publish] Success art_id=%s ts=%s headline=%s",
                    art_id, ts, article.get("headline", "N/A")[:50]
                )
                return True
            else:
                self.logger.error(
                    "[FinnHub:Publish] Send failed art_id=%s",
                    art_id
                )
                return False
                
        except json.JSONEncodeError as e:
            self.logger.error(
                "[FinnHub:Publish] JSON encode error art_id=%s error=%s",
                art_id, e
            )
            return False
            
        except KeyError as e:
            self.logger.error(
                "[FinnHub:Publish] Missing key art_id=%s error=%s",
                art_id, e
            )
            return False
            
        except Exception as e:
            self.logger.error(
                "[FinnHub:Publish] Failed art_id=%s error=%s",
                art_id, e,
                exc_info=True
            )
            return False


    # -------------------------------------------------------------------------
    # Main producer loop with enhanced error handling
    # -------------------------------------------------------------------------
    def _run_loop(self):
        iteration = 0
        total_sent = 0
        total_errors = 0

        self.logger.info("[FinnHub:Loop] Starting producer loop...")

        while self._running:
            iteration += 1
            start_time = time.time()
            new_events = 0
            loop_errors = 0

            try:
                articles = self._fetch_news()
                
                if articles:
                    articles.sort(key=lambda x: x.get("datetime", 0))
                    
                    for article in articles:
                        try:
                            if self._publish_article(article):
                                new_events += 1
                                total_sent += 1
                        except Exception as e:
                            loop_errors += 1
                            total_errors += 1
                            self.logger.error(
                                "[FinnHub:Loop] Article processing error error=%s",
                                e
                            )
                
                loop_duration = time.time() - start_time
                
                # Health check logging
                if time.time() - self.last_successful_fetch > 3600:  # 1 hour
                    self.logger.warning(
                        "[FinnHub:Health] No successful fetch in last hour"
                    )
                
                self.logger.info(
                    "[FinnHub:Loop] iter=%s new=%s total=%s errors=%s total_errors=%s duration=%.2fs",
                    iteration, new_events, total_sent, loop_errors, total_errors, loop_duration
                )

                # Adaptive sleep based on performance
                if loop_errors > len(self.tickers) // 2:
                    extra_sleep = min(60, self.poll_interval * 2)
                    self.logger.warning(
                        "[FinnHub:Loop] High error rate, extending sleep to %s seconds",
                        extra_sleep
                    )
                    time.sleep(extra_sleep)
                else:
                    time.sleep(self.poll_interval)

            except KeyboardInterrupt:
                self.logger.info("[FinnHub:Loop] Keyboard interrupt received")
                self.stop()

            except SystemExit:
                self.logger.info("[FinnHub:Loop] System exit received")
                raise

            except MemoryError as e:
                self.logger.critical(
                    "[FinnHub:Loop] Memory error: %s. Clearing cache and restarting.",
                    e
                )
                self.seen_ids = set()
                time.sleep(60)
                
            except Exception as e:
                total_errors += 1
                self.logger.error(
                    "[FinnHub:LoopError] Unexpected error error=%s total_errors=%s",
                    e, total_errors,
                    exc_info=True
                )
                
                # Exponential backoff on repeated errors
                backoff_time = min(300, 5 * (2 ** min(total_errors, 6)))  # Max 5 minutes
                self.logger.warning(
                    "[FinnHub:LoopError] Backing off for %s seconds",
                    backoff_time
                )
                time.sleep(backoff_time)

        self.logger.info(
            "[FinnHub:Shutdown] Producer stopped. Stats: iterations=%s total_sent=%s total_errors=%s",
            iteration, total_sent, total_errors
        )


    # -------------------------------------------------------------------------
    def stop(self):
        self.logger.info("[FinnHub:Stop] Shutdown signal received.")
        self._running = False