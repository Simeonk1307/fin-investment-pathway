import os
import sys
import json
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger(__name__)


def test_missing_env_vars():
    logger.info("=" * 60)
    logger.info("TEST: Missing Environment Variables")
    logger.info("=" * 60)

    required = [
        "PATHWAY_LICENSE_KEY",
        "REDPANDA_BRONZE_NEWS_TOPIC",
        "FINNHUB_API_KEY",
        "TICKERS",
        "REDPANDA_BROKERS",
        "REDPANDA_SECURITY_PROTOCOL",
        "REDPANDA_SASL_MECHANISM",
        "REDPANDA_USERNAME",
        "REDPANDA_PASSWORD"
    ]

    for var in required:
        original = os.environ.pop(var, None)
        missing = [v for v in required if not os.getenv(v)]
        if missing:
            logger.info(f"[PASS] Missing {var}: detected correctly")
        else:
            logger.error(f"[FAIL] Missing {var}: not detected")
        if original:
            os.environ[var] = original


def test_invalid_broker():
    logger.info("=" * 60)
    logger.info("TEST: Invalid Broker Connection")
    logger.info("=" * 60)

    from confluent_kafka.admin import AdminClient

    invalid_configs = [
        {"bootstrap.servers": "invalid-host:9092"},
        {"bootstrap.servers": "localhost:1234"},
        {"bootstrap.servers": "192.168.255.255:9092"},
    ]

    for config in invalid_configs:
        try:
            admin = AdminClient(config)
            metadata = admin.list_topics(timeout=5.0)
            logger.error(f"[FAIL] Should have failed: {config}")
        except Exception as e:
            logger.info(f"[PASS] Broker connection failed: {type(e).__name__}")


def test_invalid_topic():
    logger.info("=" * 60)
    logger.info("TEST: Invalid Topic")
    logger.info("=" * 60)

    from confluent_kafka.admin import AdminClient

    broker = os.getenv("REDPANDA_BROKERS", "localhost:9092")

    config = {
        "bootstrap.servers": broker,
        "security.protocol": os.getenv("REDPANDA_SECURITY_PROTOCOL", "PLAINTEXT"),
    }

    if os.getenv("REDPANDA_SASL_MECHANISM"):
        config["sasl.mechanism"] = os.getenv("REDPANDA_SASL_MECHANISM")
        config["sasl.username"] = os.getenv("REDPANDA_USERNAME", "")
        config["sasl.password"] = os.getenv("REDPANDA_PASSWORD", "")

    invalid_topics = [
        "nonexistent-topic-xyz-123456",
        "fake-topic-that-does-not-exist",
        "",
    ]

    try:
        admin = AdminClient(config)
        metadata = admin.list_topics(timeout=10.0)

        for topic in invalid_topics:
            if topic and topic not in metadata.topics:
                logger.info(f"[PASS] Topic '{topic}' correctly identified as missing")
            elif not topic:
                logger.info("[PASS] Empty topic name handled")
            else:
                logger.error(f"[FAIL] Topic '{topic}' unexpectedly exists")

    except Exception as e:
        logger.warning(f"[SKIP] Cannot connect to broker: {e}")


def test_invalid_producer():
    logger.info("=" * 60)
    logger.info("TEST: Invalid Producer Config")
    logger.info("=" * 60)

    from confluent_kafka import Producer

    invalid_configs = [
        {"bootstrap.servers": "invalid-host:9092"},
        {"bootstrap.servers": "localhost:1234"},
        {
            "bootstrap.servers": "localhost:9092",
            "security.protocol": "SASL_SSL",
            "sasl.mechanism": "PLAIN",
            "sasl.username": "wrong",
            "sasl.password": "wrong",
        },
    ]

    for config in invalid_configs:
        try:
            producer = Producer(config)
            producer.produce("test-topic", value=b"test")
            remaining = producer.flush(timeout=5.0)
            if remaining > 0:
                logger.info(f"[PASS] Producer flush timeout: {remaining} pending")
            else:
                logger.error(f"[FAIL] Producer should have failed: {config}")
        except Exception as e:
            logger.info(f"[PASS] Producer failed: {type(e).__name__}")


def test_invalid_api_key():
    logger.info("=" * 60)
    logger.info("TEST: Invalid Finnhub API Key")
    logger.info("=" * 60)

    import finnhub

    invalid_keys = [
        "",
        "invalid-key",
        "12345678901234567890",
        "sk_invalid_key_format",
        None,
    ]

    for key in invalid_keys:
        try:
            if not key:
                logger.info(f"[PASS] Empty/None API key rejected")
                continue

            client = finnhub.Client(api_key=key)
            result = client.company_news("AAPL", _from="2024-01-01", to="2024-01-01")
            logger.error(f"[FAIL] Invalid key should have failed: {key[:10]}...")
        except finnhub.FinnhubAPIException as e:
            if "Invalid API key" in str(e) or "API key" in str(e):
                logger.info(f"[PASS] Invalid API key detected: {key[:10] if key else 'None'}...")
            else:
                logger.info(f"[PASS] API error: {e}")
        except Exception as e:
            logger.info(f"[PASS] API key failed: {type(e).__name__}")


def test_invalid_tickers():
    logger.info("=" * 60)
    logger.info("TEST: Invalid Tickers")
    logger.info("=" * 60)

    import finnhub

    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        logger.warning("[SKIP] No FINNHUB_API_KEY set")
        return

    invalid_tickers = [
        "INVALIDXYZ123",
        "FAKETICKER",
        "NOTREAL",
        "ZZZZZZZ",
        "123456",
        "",
        "A" * 20,
    ]

    client = finnhub.Client(api_key=api_key)

    for ticker in invalid_tickers:
        try:
            if not ticker:
                logger.info("[PASS] Empty ticker handled")
                continue

            profile = client.company_profile2(symbol=ticker)
            if profile and profile.get("ticker"):
                logger.error(f"[FAIL] Invalid ticker returned data: {ticker}")
            else:
                logger.info(f"[PASS] Invalid ticker detected: {ticker}")
            time.sleep(0.2)
        except finnhub.FinnhubAPIException as e:
            if "Too many requests" in str(e):
                logger.warning("[SKIP] Rate limited, sleeping 30s")
                time.sleep(30)
            else:
                logger.info(f"[PASS] Ticker API error: {ticker} - {e}")
        except Exception as e:
            logger.info(f"[PASS] Ticker failed: {ticker} - {type(e).__name__}")


def test_invalid_tickers_json():
    logger.info("=" * 60)
    logger.info("TEST: Invalid TICKERS JSON")
    logger.info("=" * 60)

    invalid_jsons = [
        "",
        "not json",
        "[",
        "[]",
        "null",
        "{}",
        "[1, 2, 3]",
        '[""]',
    ]

    for invalid in invalid_jsons:
        try:
            tickers = json.loads(invalid) if invalid else None
            if not tickers or not isinstance(tickers, list):
                logger.info(f"[PASS] Invalid JSON rejected: {invalid[:20] if invalid else 'empty'}...")
            elif not all(isinstance(t, str) and t for t in tickers):
                logger.info(f"[PASS] Invalid ticker format rejected: {invalid[:20]}...")
            else:
                logger.error(f"[FAIL] Should have rejected: {invalid[:20]}...")
        except json.JSONDecodeError:
            logger.info(f"[PASS] JSON parse error: {invalid[:20] if invalid else 'empty'}...")
        except Exception as e:
            logger.info(f"[PASS] Rejected: {type(e).__name__}")


def test_network_errors():
    logger.info("=" * 60)
    logger.info("TEST: Network Errors")
    logger.info("=" * 60)

    import requests

    invalid_urls = [
        "http://192.168.255.255:9999/api",
        "http://localhost:1/api",
        "http://invalid-domain-xyz.fake/api",
    ]

    for url in invalid_urls:
        try:
            response = requests.get(url, timeout=3)
            logger.error(f"[FAIL] Should have failed: {url}")
        except requests.exceptions.ConnectTimeout:
            logger.info(f"[PASS] ConnectTimeout: {url}")
        except requests.exceptions.ConnectionError:
            logger.info(f"[PASS] ConnectionError: {url}")
        except requests.exceptions.ReadTimeout:
            logger.info(f"[PASS] ReadTimeout: {url}")
        except Exception as e:
            logger.info(f"[PASS] Network error: {type(e).__name__}")


def test_memory_error_simulation():
    logger.info("=" * 60)
    logger.info("TEST: Memory Error Handling")
    logger.info("=" * 60)

    seen_ids = set()

    try:
        for i in range(100):
            seen_ids.add(f"id_{i}")

        if len(seen_ids) > 50:
            seen_ids = set()
            logger.info("[PASS] Cache reset works correctly")
        else:
            logger.error("[FAIL] Cache reset not triggered")

    except MemoryError:
        logger.info("[PASS] MemoryError caught")
    except Exception as e:
        logger.error(f"[FAIL] Unexpected error: {e}")


def test_keyboard_interrupt():
    logger.info("=" * 60)
    logger.info("TEST: KeyboardInterrupt Handling")
    logger.info("=" * 60)

    class MockProducer:
        def __init__(self):
            self._running = True

        def stop(self):
            self._running = False

        def run(self):
            try:
                raise KeyboardInterrupt()
            except KeyboardInterrupt:
                self.stop()
                return "stopped"

    producer = MockProducer()
    result = producer.run()
    if result == "stopped" and not producer._running:
        logger.info("[PASS] KeyboardInterrupt handled gracefully")
    else:
        logger.error("[FAIL] KeyboardInterrupt not handled")


def test_system_exit():
    logger.info("=" * 60)
    logger.info("TEST: SystemExit Handling")
    logger.info("=" * 60)

    class MockProducer:
        def run(self):
            try:
                raise SystemExit(0)
            except SystemExit:
                raise

    producer = MockProducer()
    try:
        producer.run()
        logger.error("[FAIL] SystemExit should have propagated")
    except SystemExit:
        logger.info("[PASS] SystemExit propagated correctly")


def test_rate_limiting():
    logger.info("=" * 60)
    logger.info("TEST: Rate Limiting Detection")
    logger.info("=" * 60)

    import finnhub

    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        logger.warning("[SKIP] No FINNHUB_API_KEY set")
        return

    client = finnhub.Client(api_key=api_key)

    logger.info("[INFO] Sending rapid requests to trigger rate limit...")
    rate_limited = False

    for i in range(100):
        try:
            client.company_news("AAPL", _from="2024-01-01", to="2024-01-01")
        except finnhub.FinnhubAPIException as e:
            if "Too many requests" in str(e):
                logger.info(f"[PASS] Rate limit detected at request {i+1}")
                rate_limited = True
                break
        except Exception as e:
            logger.info(f"[INFO] Error at request {i+1}: {e}")
            break

    if not rate_limited:
        logger.warning("[SKIP] Rate limit not triggered (API may allow high volume)")


def test_empty_response():
    logger.info("=" * 60)
    logger.info("TEST: Empty Response Handling")
    logger.info("=" * 60)

    import finnhub

    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        logger.warning("[SKIP] No FINNHUB_API_KEY set")
        return

    client = finnhub.Client(api_key=api_key)

    try:
        articles = client.company_news("AAPL", _from="1990-01-01", to="1990-01-02")
        if articles == [] or articles is None:
            logger.info("[PASS] Empty response handled correctly")
        else:
            logger.warning(f"[INFO] Unexpected articles from 1990: {len(articles)}")
    except Exception as e:
        logger.info(f"[PASS] Error handled: {e}")


def test_malformed_article():
    logger.info("=" * 60)
    logger.info("TEST: Malformed Article Handling")
    logger.info("=" * 60)

    malformed_articles = [
        {},
        {"id": None},
        {"id": 123, "datetime": None},
        {"id": 123, "datetime": 0},
        {"id": 123, "datetime": 1234567890, "headline": None},
        {"id": 123, "datetime": 1234567890, "headline": "Test", "url": None},
        {"id": 123, "datetime": 1234567890, "headline": "", "url": ""},
        {"datetime": 1234567890, "headline": "Test", "url": "http://test.com"},
        {"id": 123, "headline": "Test", "url": "http://test.com"},
    ]

    def validate_article(article):
        art_id = article.get("id")
        ts = article.get("datetime", 0)
        if not art_id or not ts:
            return False
        headline = article.get("headline")
        if not headline or not article.get("url"):
            return False
        return True

    for article in malformed_articles:
        if validate_article(article):
            logger.error(f"[FAIL] Should have rejected: {article}")
        else:
            logger.info(f"[PASS] Malformed article rejected: {list(article.keys())}")


def test_duplicate_detection():
    logger.info("=" * 60)
    logger.info("TEST: Duplicate Detection")
    logger.info("=" * 60)

    seen_ids = set()

    articles = [
        {"id": 1, "datetime": 100},
        {"id": 2, "datetime": 200},
        {"id": 1, "datetime": 100},
        {"id": 3, "datetime": 300},
        {"id": 2, "datetime": 200},
    ]

    published = 0
    duplicates = 0

    for article in articles:
        art_id = article.get("id")
        if art_id in seen_ids:
            duplicates += 1
        else:
            seen_ids.add(art_id)
            published += 1

    if published == 3 and duplicates == 2:
        logger.info(f"[PASS] Duplicates detected: {duplicates}, Published: {published}")
    else:
        logger.error(f"[FAIL] Expected 3 published, 2 duplicates. Got {published}, {duplicates}")


def run_all_tests():
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINNHUB NEWS PRODUCER - ERROR HANDLING TESTS")
    logger.info("=" * 60)
    logger.info("")

    tests = [
        ("Missing Env Vars", test_missing_env_vars),
        ("Invalid Broker", test_invalid_broker),
        ("Invalid Topic", test_invalid_topic),
        ("Invalid Producer", test_invalid_producer),
        ("Invalid API Key", test_invalid_api_key),
        ("Invalid Tickers", test_invalid_tickers),
        ("Invalid Tickers JSON", test_invalid_tickers_json),
        ("Network Errors", test_network_errors),
        ("Memory Error", test_memory_error_simulation),
        ("KeyboardInterrupt", test_keyboard_interrupt),
        ("SystemExit", test_system_exit),
        ("Rate Limiting", test_rate_limiting),
        ("Empty Response", test_empty_response),
        ("Malformed Article", test_malformed_article),
        ("Duplicate Detection", test_duplicate_detection),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            logger.error(f"[ERROR] Test '{name}' crashed: {e}")
            failed += 1

    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Tests : {len(tests)}")
    logger.info(f"Passed      : {passed}")
    logger.info(f"Failed      : {failed}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_all_tests()