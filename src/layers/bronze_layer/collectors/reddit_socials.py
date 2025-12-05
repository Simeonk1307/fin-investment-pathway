import logging
import time
import requests
from bs4 import BeautifulSoup
from typing import List, Dict

from src.layers.bronze_layer.base.base_producer import BaseProducer
from src.layers.bronze_layer.event_envelope import create_event_envelope


HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BronzeSocialsBot/1.0; +https://yourdomain.com)"
}


class RedditHtmlProducer(BaseProducer):

    MAX_DEDUPE_CACHE = 50_000
    MAX_CONSECUTIVE_ERRORS = 5
    ERROR_BACKOFF_SECONDS = 30

    def __init__(
        self,
        logger: logging.Logger,
        topic: str,
        producer_config: Dict,
        tickers: List[str],
        subreddit: str = "stocks",
        poll_interval: int = 120,
        limit: int = 10,
    ):
        super().__init__(logger=logger, topic=topic, producer_config=producer_config)

        self.tickers = tickers
        self.subreddit = subreddit
        self.poll_interval = poll_interval
        self.limit = limit
        self._running = True
        self.consecutive_errors = 0
        self.last_successful_fetch = time.time()

        self.seen_urls = set()

        self.logger.info("=" * 80)
        self.logger.info("[Reddit] Producer initialized")
        self.logger.info("  Topic         : %s", topic)
        self.logger.info("  Subreddit     : r/%s", subreddit)
        self.logger.info("  Tickers       : %s", tickers)
        self.logger.info("  Poll Interval : %s sec", poll_interval)
        self.logger.info("  Fetch Limit   : %s", limit)
        self.logger.info("=" * 80)


    # -------------------------------------------------------------------------
    # HTTP fetch with retry & backoff - Enhanced with timeout handling
    # -------------------------------------------------------------------------
    def _http_get(self, url: str, retries: int = 3, backoff: float = 0.2) -> str:
        if self.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
            self.logger.error(
                "[Reddit:HTTP] Too many consecutive errors (%s), backing off for %s seconds",
                self.consecutive_errors, self.ERROR_BACKOFF_SECONDS
            )
            time.sleep(self.ERROR_BACKOFF_SECONDS)
            self.consecutive_errors = 0

        for attempt in range(1, retries + 1):
            try:
                # Increase timeout on retries
                timeout = (5 + (attempt * 2), 10 + (attempt * 5))
                resp = requests.get(url, headers=HEADERS, timeout=timeout)

                if resp.status_code == 200:
                    self.consecutive_errors = 0
                    self.last_successful_fetch = time.time()
                    return resp.text

                self.logger.warning(
                    "[Reddit:HTTP] Non-200 status=%s url=%s attempt=%s",
                    resp.status_code, url, attempt
                )
                
                if resp.status_code == 429:  # Rate limited
                    self.logger.warning("[Reddit:HTTP] Rate limited, backing off")
                    time.sleep(300)
                    continue
                    
                return ""

            except requests.exceptions.Timeout as e:
                self.consecutive_errors += 1
                self.logger.warning(
                    "[Reddit:HTTP] Timeout attempt=%s url=%s error=%s",
                    attempt, url, type(e).__name__
                )
                if attempt < retries:
                    sleep_time = backoff * (2 ** attempt) + (0.1 * (hash(url) % 10))
                    time.sleep(sleep_time)
                continue
                
            except requests.exceptions.ConnectionError as e:
                self.consecutive_errors += 1
                self.logger.warning(
                    "[Reddit:HTTP] Connection error attempt=%s url=%s error=%s",
                    attempt, url, type(e).__name__
                )
                if attempt < retries:
                    sleep_time = backoff * (2 ** attempt) + (0.1 * (hash(url) % 10))
                    time.sleep(sleep_time)
                continue
                
            except Exception as e:
                self.consecutive_errors += 1
                self.logger.error(
                    "[Reddit:HTTP] Fetch attempt=%s url=%s error=%s",
                    attempt, url, e,
                    exc_info=(attempt == retries)  # Full traceback only on final failure
                )
                if attempt < retries:
                    sleep_time = backoff * (2 ** attempt) + (0.1 * (hash(url) % 10))
                    time.sleep(sleep_time)
                continue

        self.logger.error("[Reddit:HTTP] Failed after retries url=%s", url)
        return ""


    # -------------------------------------------------------------------------
    # Extract single post body with enhanced error handling
    # -------------------------------------------------------------------------
    def _extract_post_body(self, url: str) -> str:
        if not self._running:
            return ""

        html = self._http_get(url)
        if not html:
            return ""

        try:
            soup = BeautifulSoup(html, "html.parser")
            expando = soup.find("div", class_="expando")

            if not expando:
                self.logger.debug("[Reddit:Body] No expando section url=%s", url)
                return ""

            md = expando.find("div", class_="md")
            if not md:
                self.logger.debug("[Reddit:Body] No markdown content url=%s", url)
                return ""

            return md.get_text(separator="\n").strip()

        except Exception as e:
            self.logger.error(
                "[Reddit:Body] Failed to parse post url=%s error=%s",
                url, e,
                exc_info=True
            )
            return ""


    # -------------------------------------------------------------------------
    # Fetch all posts for a specific ticker/company with enhanced error handling
    # -------------------------------------------------------------------------
    def _fetch_posts(self, company: str) -> List[Dict]:
        if not self._running:
            return []

        query = company.replace(" ", "+")
        url = f"https://old.reddit.com/r/{self.subreddit}/search/?q={query}&restrict_sr=1&sort=new"

        html = self._http_get(url)
        if not html:
            return []

        try:
            soup = BeautifulSoup(html, "html.parser")
            results = soup.find_all("div", class_="search-result")
        except Exception as e:
            self.logger.error(
                "[Reddit:Parse] Failed to parse search results url=%s error=%s",
                url, e,
                exc_info=True
            )
            return []

        posts = []
        failed_posts = 0
        new_urls = []

        for item in results[: self.limit]:
            try:
                title_tag = item.find("a", class_="search-title")
                if not title_tag:
                    continue

                title = title_tag.text.strip()
                link = title_tag.get("href", "")

                # dedupe across fetch cycles
                if link in self.seen_urls:
                    continue

                body = self._extract_post_body(link)

                # Validate post has minimum required content
                if not title or not link:
                    self.logger.debug(
                        "[Reddit:Fetch] Skipping invalid post company=%s title=%s",
                        company, title[:50] if title else "N/A"
                    )
                    continue

                posts.append({
                    "source": f"reddit:{self.subreddit}",
                    "company": company,
                    "title": title,
                    "url": link,
                    "content": body,
                })

                new_urls.append(link)

            except Exception as e:
                failed_posts += 1
                self.logger.error(
                    "[Reddit:ParseItem] Error parsing individual post error=%s",
                    e,
                    exc_info=True
                )

        # Only add to seen_urls after successful processing
        for url in new_urls:
            self.seen_urls.add(url)

        self.logger.info(
            "[Reddit:Fetch] company=%s extracted=%s failed=%s cache_size=%s",
            company, len(posts), failed_posts, len(self.seen_urls)
        )

        # dedupe safety reset
        if len(self.seen_urls) > self.MAX_DEDUPE_CACHE:
            self.logger.warning("[Reddit:Dedupe] Cache exceeded threshold. Resetting.")
            self.seen_urls = set()

        return posts


    # -------------------------------------------------------------------------
    # Publish single post with enhanced error handling
    # -------------------------------------------------------------------------
    def _publish_post(self, post: Dict) -> bool:
        url = post.get("url", "")
        
        if not url:
            self.logger.warning(
                "[Reddit:Publish] Post missing URL, skipping"
            )
            return False
        
        # Note: We don't check self.seen_urls here anymore
        # Deduplication is handled in _fetch_posts
        
        try:
            # Validate post structure - only require essential fields
            required_fields = ["title", "url", "company"]
            missing_required = [field for field in required_fields if not post.get(field)]
            
            if missing_required:
                self.logger.warning(
                    "[Reddit:Publish] Missing required fields url=%s missing=%s",
                    url, missing_required
                )
                return False
            
            # Check for optional fields and log at debug level if missing
            optional_fields = ["content"]
            missing_optional = [field for field in optional_fields if not post.get(field)]
            
            if missing_optional:
                self.logger.debug(
                    "[Reddit:Publish] Missing optional fields url=%s missing=%s",
                    url, missing_optional
                )
            
            event = create_event_envelope(
                payload=post,
                source="reddit",
                source_type="webscrape"
            )
            
            if self.send(event, key=url):
                self.logger.debug(
                    "[Reddit:Publish] Success url=%s title=%s",
                    url, post.get("title", "N/A")[:50]
                )
                return True
            else:
                self.logger.error(
                    "[Reddit:Publish] Send failed url=%s",
                    url
                )
                return False
                
        except Exception as e:
            self.logger.error(
                "[Reddit:Publish] Failed url=%s error=%s",
                url, e,
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

        self.logger.info("[Reddit:Loop] Starting producer loop...")

        while self._running:
            iteration += 1
            start_time = time.time()
            new_events = 0
            loop_errors = 0

            try:
                for ticker in self.tickers:
                    if not self._running:
                        break
                        
                    posts = self._fetch_posts(ticker)

                    for post in posts:
                        try:
                            if self._publish_post(post):
                                new_events += 1
                                total_sent += 1
                        except Exception as e:
                            loop_errors += 1
                            total_errors += 1
                            self.logger.error(
                                "[Reddit:Loop] Post processing error error=%s",
                                e
                            )

                loop_duration = time.time() - start_time
                
                # Health check logging
                if time.time() - self.last_successful_fetch > 3600:  # 1 hour
                    self.logger.warning(
                        "[Reddit:Health] No successful fetch in last hour"
                    )
                
                self.logger.info(
                    "[Reddit:Loop] iter=%s new=%s total=%s errors=%s total_errors=%s duration=%.2fs",
                    iteration, new_events, total_sent, loop_errors, total_errors, loop_duration
                )

                # Adaptive sleep based on performance
                if loop_errors > len(self.tickers) // 2:
                    extra_sleep = min(60, self.poll_interval * 2)
                    self.logger.warning(
                        "[Reddit:Loop] High error rate, extending sleep to %s seconds",
                        extra_sleep
                    )
                    time.sleep(extra_sleep)
                else:
                    time.sleep(self.poll_interval)

            except KeyboardInterrupt:
                self.logger.info("[Reddit:Loop] Keyboard interrupt received")
                self.stop()

            except SystemExit:
                self.logger.info("[Reddit:Loop] System exit received")
                raise

            except MemoryError as e:
                self.logger.critical(
                    "[Reddit:Loop] Memory error: %s. Clearing cache and restarting.",
                    e
                )
                self.seen_urls = set()
                time.sleep(60)
                
            except Exception as e:
                total_errors += 1
                self.logger.error(
                    "[Reddit:LoopError] Unexpected error error=%s total_errors=%s",
                    e, total_errors,
                    exc_info=True
                )
                
                # Exponential backoff on repeated errors
                backoff_time = min(300, 5 * (2 ** min(total_errors, 6)))  # Max 5 minutes
                self.logger.warning(
                    "[Reddit:LoopError] Backing off for %s seconds",
                    backoff_time
                )
                time.sleep(backoff_time)

        self.logger.info(
            "[Reddit:Shutdown] Producer stopped. Stats: iterations=%s total_sent=%s total_errors=%s",
            iteration, total_sent, total_errors
        )


    # -------------------------------------------------------------------------
    def stop(self):
        self.logger.info("[Reddit:Stop] Shutdown signal received.")
        self._running = False