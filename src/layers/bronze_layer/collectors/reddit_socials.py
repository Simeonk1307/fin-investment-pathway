import logging, time, requests
from bs4 import BeautifulSoup
from typing import List, Dict
from src.layers.bronze_layer.base.base_producer import BaseProducer
from src.layers.bronze_layer.event_envelope import create_event_envelope

HEADERS = {"User-Agent": "Mozilla/5.0 (BronzeSocialsBot/1.0)"}
_RETRY = (5, 10, 20)


class RedditHtmlProducer(BaseProducer):
    MAX_DEDUPE_CACHE = 50000
    MAX_CONSECUTIVE_ERRORS = 5
    BACKOFF = 30

    def __init__(self, logger, topic, producer_config, tickers,
                 subreddit="stocks", poll_interval=120, limit=10,
                 debug=False, debug_writer=None):
        super().__init__(logger, topic, producer_config)
        self.tickers = tickers
        self.subreddit = subreddit
        self.poll_interval = poll_interval
        self.limit = limit
        self.debug = debug
        self.debug_writer = debug_writer or (lambda *_: None)
        self.seen = set()
        self.consecutive_errors = 0
        self.last_successful_fetch = time.time()
        if debug:
            self.debug_writer("reddit", "startup", {"topic": topic, "tickers": tickers})
        logger.info(f"Reddit ready (topic={topic}, sub={subreddit}, tickers={len(tickers)})")

    def _http(self, url):
        if self.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
            self.logger.error(f"Too many errors, pausing {self.BACKOFF}s")
            time.sleep(self.BACKOFF)
            self.consecutive_errors = 0
        for r in range(3):
            try:
                resp = requests.get(url, headers=HEADERS,
                                    timeout=(10 + r * 5, 20 + r * 10))
                if resp.status_code == 200:
                    self.consecutive_errors = 0
                    self.last_successful_fetch = time.time()
                    return resp.text
                if resp.status_code == 429:
                    time.sleep(300)
                    continue
                self.logger.warning(f"HTTP {resp.status_code} {url}")
                return ""
            except requests.exceptions.Timeout:
                self.consecutive_errors += 1
                if r < 2:
                    time.sleep(_RETRY[r])
                    continue
            except Exception:
                self.consecutive_errors += 1
                if r < 2:
                    time.sleep(_RETRY[r])
                    continue
        self.logger.error(f"HTTP fail {url}")
        return ""

    def _extract_body(self, url):
        html = self._http(url)
        if not html:
            return ""
        try:
            soup = BeautifulSoup(html, "html.parser")
            md = soup.select_one(".expando .md")
            return md.get_text("\n").strip() if md else ""
        except Exception:
            return ""

    def _fetch_posts(self, ticker):
        q = ticker.replace(" ", "+")
        url = f"https://old.reddit.com/r/{self.subreddit}/search/?q={q}&restrict_sr=1&sort=new"
        html = self._http(url)
        if not html:
            return []
        try:
            soup = BeautifulSoup(html, "html.parser")
            results = soup.find_all("div", class_="search-result")[: self.limit]
        except Exception:
            self.logger.error(f"{ticker}: parse error")
            return []

        posts, ok, fail = [], 0, 0
        for item in results:
            try:
                tag = item.find("a", class_="search-title")
                if not tag:
                    continue
                title = tag.text.strip()
                link = tag.get("href", "")
                if not title or not link or link in self.seen:
                    continue
                body = self._extract_body(link)
                posts.append({
                    "company": ticker,
                    "title": title,
                    "url": link,
                    "content": body,
                    "source": f"reddit:{self.subreddit}"
                })
                self.seen.add(link)
                ok += 1
            except Exception:
                fail += 1

        if ok or fail:
            self.logger.info(f"{ticker}: {ok} posts (fail={fail})")

        if len(self.seen) > self.MAX_DEDUPE_CACHE:
            self.seen = set()

        return posts

    def _publish(self, p):
        url = p.get("url")
        if not url:
            return False
        try:
            ev = create_event_envelope(p, source="reddit", source_type="web")
            if self.send(ev, key="reddit"):
                return True
        except Exception:
            pass
        self.logger.error(f"Publish fail {url}")
        return False

    def _run_loop(self):
        i = sent = errs = 0
        self.logger.info("Loop start")
        while self._running:
            i += 1
            start = time.time()
            new = loop_err = 0
            try:
                for t in self.tickers:
                    if not self._running:
                        break
                    posts = self._fetch_posts(t)
                    for p in posts:
                        try:
                            if self._publish(p):
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
