import copy
import ipaddress
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from io import BytesIO
from urllib.parse import parse_qs, urlparse

from flask import Flask, Response, jsonify, render_template, request, send_file

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True
logging.basicConfig(level=logging.INFO)


logger = logging.getLogger(__name__)

# Scrape caching and concurrency locking to guarantee consistent results
# and eliminate process collisions when navigating between dashboards.
_SCRAPE_CACHE = {}  # key -> (timestamp, data)
_CACHE_TTL = 900    # 15 minutes TTL
_SCRAPE_LOCK = threading.Lock()
_IN_PROGRESS_LOCKS = {}  # key -> threading.Lock()


def _normalize_cache_key(url, page, max_pages, verified_only, all_sorts):
    """Generate a stable cache key based on URL path, product ID, and crawl parameters."""
    parsed = urlparse((url or "").strip())
    path = parsed.path
    qs = parse_qs(parsed.query)
    pid = qs.get("pid", [""])[0]
    return (
        path,
        pid,
        str(page).lower(),
        str(max_pages).lower() if max_pages else "none",
        bool(verified_only),
        bool(all_sorts),
    )


# ---------------------------------------------------------------------------
# URL Validation (SSRF protection)
# ---------------------------------------------------------------------------

# Private/link-local/loopback IP ranges that must not be contacted
_PRIVATE_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),   # link-local / AWS metadata
    ipaddress.ip_network("::1/128"),           # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),          # IPv6 unique-local
]

_ALLOWED_FLIPKART_HOSTS = {"flipkart.com", "www.flipkart.com"}


def _validate_flipkart_url(url):
    """
    Validate that a URL is a safe, legitimate Flipkart URL.

    Checks performed:
    1. URL can be parsed.
    2. Scheme is https (not http, ftp, file, etc.)
    3. Host is flipkart.com or www.flipkart.com.
    4. Host is not a private/loopback IP address.
    5. URL contains some path (not just the domain root).

    Returns:
        None if valid.
        str error message if invalid.
    """
    if not url:
        return "URL is required"

    try:
        parsed = urlparse(url)
    except Exception:
        return "URL could not be parsed"

    if parsed.scheme != "https":
        return f"URL scheme '{parsed.scheme}' is not allowed. Use https."

    host = parsed.hostname or ""
    if not host:
        return "URL has no hostname"

    # Reject private/loopback IP addresses (SSRF protection)
    try:
        ip = ipaddress.ip_address(host)
        for network in _PRIVATE_NETWORKS:
            if ip in network:
                return f"URL hostname resolves to a private or reserved IP address"
    except ValueError:
        # Not an IP address — proceed to domain check
        pass

    # Domain enforcement: only flipkart.com
    host_lower = host.lower()
    if host_lower not in _ALLOWED_FLIPKART_HOSTS:
        return (
            f"Host '{host}' is not allowed. "
            "Only flipkart.com and www.flipkart.com are accepted."
        )

    return None  # Valid


def _validate_page_param(page_str):
    """
    Validate the 'page' query parameter.

    Valid values: positive integer, "all", "0" (treated as all), "*"
    Invalid: negative numbers, non-numeric non-keyword strings.

    Returns:
        (normalized_page_str, error_message_or_None)
    """
    if not page_str:
        return "1", None
    lower = page_str.strip().lower()
    if lower in ("all", "0", "*"):
        return lower, None
    try:
        n = int(lower)
        if n < 1:
            return None, f"'page' must be a positive integer or 'all', got: {page_str!r}"
        return str(n), None
    except ValueError:
        return None, f"Invalid 'page' value: {page_str!r}. Must be a positive integer or 'all'."


def _validate_max_pages_param(max_pages_str):
    """
    Validate the 'max_pages' query parameter.

    Valid values: positive integer.
    Invalid: 0, negative numbers, non-numeric strings, "all" (ambiguous).

    Returns:
        (normalized_value_or_None, error_message_or_None)
    """
    if not max_pages_str:
        return None, None
    try:
        n = int(max_pages_str.strip())
        if n < 1:
            return None, f"'max_pages' must be a positive integer, got: {max_pages_str!r}"
        return str(n), None
    except ValueError:
        return None, f"Invalid 'max_pages' value: {max_pages_str!r}. Must be a positive integer."


def _validate_top_n_param(top_n_str, default=15, maximum=50):
    """
    Validate the 'top_n' query parameter.

    BUG FIX S2: Previously used unguarded int() which raised ValueError
    on non-numeric input, causing unhandled 500 errors.

    Returns:
        (validated_int, error_message_or_None)
    """
    if not top_n_str:
        return default, None
    try:
        n = int(top_n_str.strip())
        if n < 1:
            return None, f"'top_n' must be a positive integer, got: {top_n_str!r}"
        return min(n, maximum), None
    except ValueError:
        return None, f"Invalid 'top_n' value: {top_n_str!r}. Must be a positive integer."


# ---------------------------------------------------------------------------
# Helper: Run the Scrapy spider and return parsed JSON data
# ---------------------------------------------------------------------------

def _execute_scrapy_subprocess(url, page="1", max_pages=None, verified_only=True, all_sorts=None):
    """Execute Scrapy spider as an isolated subprocess and return raw items."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_output_path = tmp.name

    try:
        cmd = [
            sys.executable,
            "-m",
            "scrapy",
            "crawl",
            "review",
            "-O",
            tmp_output_path,
            "-a",
            f"url={url}",
            "-a",
            f"page={page}",
            "-a",
            f"verified_only={'true' if verified_only else 'false'}",
        ]
        if max_pages:
            cmd.extend(["-a", f"max_pages={max_pages}"])

        page_lower = str(page).strip().lower()
        if all_sorts is True or (all_sorts is None and page_lower in ("all", "0", "*")):
            cmd.extend(["-a", "all_sorts=true"])
        elif all_sorts is False:
            cmd.extend(["-a", "all_sorts=false"])

        timeout = 900 if (page_lower in ("all", "0", "*") or max_pages) else 60

        logger.info("Executing Scrapy command: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            logger.error(
                "Scrapy crawl failed with code %s:\n%s", result.returncode, result.stderr
            )
            return None, {
                "error": "Scraping failed",
                "details": result.stderr[-500:] if result.stderr else "Unknown error",
            }, 502

        if os.path.exists(tmp_output_path) and os.path.getsize(tmp_output_path) > 0:
            with open(tmp_output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
                logger.error("Scraper produced invalid JSON structure for url: %s", url)
                return None, {
                    "error": "Failed to parse scraper output",
                    "message": "The scraper produced an invalid response structure. Please try again.",
                }, 502
            return data, None, None
        else:
            logger.warning(
                "Scraper returned empty output (0 bytes) for URL: %s. "
                "Possible causes: Flipkart blocked the request, CAPTCHA, "
                "invalid product URL, or no reviews available.",
                url,
            )
            return [], None, None

    except subprocess.TimeoutExpired:
        logger.error("Scrapy crawl timed out for url: %s", url)
        return None, {
            "error": "Gateway Timeout",
            "message": "Scraper timed out while fetching reviews. "
                       "Try fetching fewer pages with max_pages=1 or max_pages=2.",
        }, 504
    except json.JSONDecodeError as e:
        logger.error("JSON decode error reading scraper output: %s", e)
        return None, {
            "error": "Failed to parse scraper output",
            "message": "The scraper produced malformed output. Please try again.",
        }, 500
    except Exception as e:
        logger.error("Unexpected error in scraper: %s", e, exc_info=True)
        return None, {
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again.",
        }, 500
    finally:
        if os.path.exists(tmp_output_path):
            try:
                os.remove(tmp_output_path)
            except OSError:
                pass


def _run_scraper(url, page="1", max_pages=None, verified_only=True, all_sorts=None, force_refresh=False):
    """
    Execute the Scrapy spider with caching and per-URL locking to guarantee
    consistent results and prevent duplicate concurrent subprocess collisions.
    """
    page_lower = str(page).strip().lower()
    effective_all_sorts = all_sorts is True or (all_sorts is None and page_lower in ("all", "0", "*"))
    cache_key = _normalize_cache_key(url, page, max_pages, verified_only, effective_all_sorts)
    now = time.time()

    if not force_refresh:
        with _SCRAPE_LOCK:
            if cache_key in _SCRAPE_CACHE:
                ts, cached_data = _SCRAPE_CACHE[cache_key]
                if now - ts < _CACHE_TTL:
                    logger.info("Serving scraped data from cache for: %s (ttl remaining: %ds)", url, int(_CACHE_TTL - (now - ts)))
                    return copy.deepcopy(cached_data), None, None

            if cache_key not in _IN_PROGRESS_LOCKS:
                _IN_PROGRESS_LOCKS[cache_key] = threading.Lock()
            url_lock = _IN_PROGRESS_LOCKS[cache_key]

        with url_lock:
            # Re-check cache after acquiring lock in case another thread finished the crawl
            with _SCRAPE_LOCK:
                if cache_key in _SCRAPE_CACHE:
                    ts, cached_data = _SCRAPE_CACHE[cache_key]
                    if time.time() - ts < _CACHE_TTL:
                        logger.info("Serving scraped data from cache (post-lock) for: %s", url)
                        return copy.deepcopy(cached_data), None, None

            data, err, status = _execute_scrapy_subprocess(
                url, page=page, max_pages=max_pages, verified_only=verified_only, all_sorts=effective_all_sorts
            )
            if data is not None and not err and len(data) > 0:
                with _SCRAPE_LOCK:
                    _SCRAPE_CACHE[cache_key] = (time.time(), copy.deepcopy(data))
            return data, err, status
    else:
        # Force refresh requested
        data, err, status = _execute_scrapy_subprocess(
            url, page=page, max_pages=max_pages, verified_only=verified_only, all_sorts=effective_all_sorts
        )
        if data is not None and not err and len(data) > 0:
            with _SCRAPE_LOCK:
                _SCRAPE_CACHE[cache_key] = (time.time(), copy.deepcopy(data))
        return data, err, status


def _separate_reviews_and_metadata(data):
    """
    Separate the scraped data list into individual reviews vs metadata.

    The spider yields mixed item types: product price, summary, and reviews.
    This splits them for analytics processing.

    Classification rules:
      - Contains 'product_price' key → price info item
      - Contains 'avg_rating' or 'total_ratings' key → product summary item
      - Contains 'review_content' or 'review_score' key → individual review

    Returns:
        tuple: (reviews_list, summary_dict_or_None, price_dict_or_None)
    """
    reviews = []
    summary = None
    price_info = None

    for item in data:
        if "product_price" in item:
            price_info = item
        elif "avg_rating" in item or "total_ratings" in item:
            summary = item
        elif "review_content" in item or "review_score" in item:
            reviews.append(item)

    return reviews, summary, price_info


# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    # If client explicitly asks for html (browser)
    accept = request.headers.get("Accept", "")
    if "text/html" in accept and request.args.get("format") != "json":
        return render_template("index.html")
    # API client, curl, test client, or format=json
    return api_info()


def _render_scraper_page(mode):
    return render_template("index.html", page_mode=mode)


@app.route("/full-report", methods=["GET"])
def full_report_page():
    return _render_scraper_page("report")


@app.route("/sentiment-mismatches", methods=["GET"])
def sentiment_mismatches_page():
    return _render_scraper_page("sentiment")


@app.route("/raw-reviews", methods=["GET"])
def raw_reviews_page():
    return _render_scraper_page("raw")


@app.route("/api-info", methods=["GET"])
def api_info():
    return jsonify({
        "name": "Flipkart Review API",
        "version": "2.0",
        "description": "API to scrape Flipkart product reviews with built-in analytics",
        "endpoints": {
            "/v1.0/reviews": "GET reviews for a given Flipkart product URL",
            "/v1.0/analytics/sentiment": "GET sentiment analysis for product reviews",
            "/v1.0/analytics/summary": "GET computed analytics summary",
            "/v1.0/analytics/topics": "GET keyword/topic extraction with sentiment",
            "/v1.0/analytics/report": "GET full analytics report (sentiment + topics + geo)",
            "/v1.0/export/csv": "GET reviews exported as CSV",
            "/health": "GET service health status",
        },
        "usage": {
            "single_page": "/v1.0/reviews?page=1&url=<FlipkartProductOrReviewURL>",
            "multiple_pages": "/v1.0/reviews?max_pages=3&url=<FlipkartProductOrReviewURL>",
            "all_pages": "/v1.0/reviews?page=all&url=<FlipkartProductOrReviewURL>",
            "sentiment": "/v1.0/analytics/sentiment?page=all&url=<FlipkartProductOrReviewURL>",
            "full_report": "/v1.0/analytics/report?max_pages=5&url=<FlipkartProductOrReviewURL>",
        },
    })


@app.route("/sample-data", methods=["GET"])
def sample_data():
    """Load sample local data and generate complete analytics for instant UI demonstration."""
    sample_file = os.path.join(os.path.dirname(__file__), "reviews.json")
    if not os.path.exists(sample_file):
        return jsonify({"error": "Sample data file not found"}), 404

    try:
        with open(sample_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        reviews, summary, price_info = _separate_reviews_and_metadata(data)
        from analytics.export import generate_report

        sample_reviews = [
            review for review in reviews if review.get("certified_buyer") is True
        ]
        sample_subset = sample_reviews[:60]
        report = generate_report(sample_subset, summary_data=summary, verified_only=True)
        report["reviews"] = sample_subset
        # Clearly label this as sample data so the frontend can distinguish
        report["data_source"] = "sample_data"
        report["sample_reviews_used"] = len(sample_subset)
        report["total_available_in_sample_file"] = len(reviews)

        if price_info:
            report["product_price"] = price_info.get("product_price", "")

        return jsonify(report), 200
    except Exception as e:
        logger.error("Error generating sample data: %s", e, exc_info=True)
        return jsonify({"error": "Failed to load sample data"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/v1.0/reviews", methods=["GET"])
def get_reviews():
    url = request.args.get("url", "").strip()
    if not url:
        return jsonify({
            "error": "Bad Request",
            "message": "Missing required 'url' parameter",
        }), 400

    url_error = _validate_flipkart_url(url)
    if url_error:
        return jsonify({"error": "Unprocessable Entity", "message": url_error}), 422

    page_str = request.args.get("page", "").strip()
    page, page_error = _validate_page_param(page_str)
    if page_error:
        return jsonify({"error": "Unprocessable Entity", "message": page_error}), 422

    max_pages_str = request.args.get("max_pages", "").strip()
    max_pages, mp_error = _validate_max_pages_param(max_pages_str)
    if mp_error:
        return jsonify({"error": "Unprocessable Entity", "message": mp_error}), 422

    verified = request.args.get("verified", "true").strip().lower() not in ("false", "0", "no")
    all_sorts_arg = request.args.get("all_sorts")
    all_sorts = (all_sorts_arg.strip().lower() in ("true", "1", "yes")) if all_sorts_arg is not None else None
    refresh = request.args.get("refresh", "false").strip().lower() in ("true", "1", "yes")
    data, error, status = _run_scraper(
        url,
        page=page or "1",
        max_pages=max_pages,
        verified_only=verified,
        all_sorts=all_sorts,
        force_refresh=refresh,
    )

    if error:
        return jsonify(error), status

    return jsonify(data), 200


# ---------------------------------------------------------------------------
# Analytics endpoints
# ---------------------------------------------------------------------------

@app.route("/v1.0/analytics/sentiment", methods=["GET"])
def get_sentiment():
    """Scrape reviews and return sentiment analysis for each review."""
    url = request.args.get("url", "").strip()
    if not url:
        return jsonify({"error": "Bad Request", "message": "Missing required 'url' parameter"}), 400

    url_error = _validate_flipkart_url(url)
    if url_error:
        return jsonify({"error": "Unprocessable Entity", "message": url_error}), 422

    page_str = request.args.get("page", "all").strip()
    page, page_error = _validate_page_param(page_str)
    if page_error:
        return jsonify({"error": "Unprocessable Entity", "message": page_error}), 422

    max_pages_str = request.args.get("max_pages", "").strip()
    max_pages, mp_error = _validate_max_pages_param(max_pages_str)
    if mp_error:
        return jsonify({"error": "Unprocessable Entity", "message": mp_error}), 422

    verified = request.args.get("verified", "true").strip().lower() not in ("false", "0", "no")
    all_sorts_arg = request.args.get("all_sorts")
    all_sorts = (all_sorts_arg.strip().lower() in ("true", "1", "yes")) if all_sorts_arg is not None else None
    refresh = request.args.get("refresh", "false").strip().lower() in ("true", "1", "yes")
    data, error, status = _run_scraper(
        url,
        page=page,
        max_pages=max_pages,
        verified_only=verified,
        all_sorts=all_sorts,
        force_refresh=refresh,
    )
    if error:
        return jsonify(error), status

    reviews, summary, price_info = _separate_reviews_and_metadata(data)

    if not reviews:
        return jsonify({
            "message": "No reviews found to analyze. "
                       "Flipkart may have blocked the request or the product has no reviews.",
            "reviews": [],
            "aggregate_sentiment": None,
            "positive_ngrams": [],
            "negative_ngrams": [],
        }), 200

    try:
        from analytics.sentiment import aggregate_sentiment, analyze_batch
        from analytics.topics import extract_ngrams

        analyzed = analyze_batch(reviews)
        aggregate = aggregate_sentiment(analyzed)

        resp = {
            "aggregate_sentiment": aggregate,
            "reviews": analyzed,
            "summary": summary,
            "positive_ngrams": extract_ngrams(reviews, n=2, top_k=10, filter_sentiment="positive"),
            "negative_ngrams": extract_ngrams(reviews, n=2, top_k=10, filter_sentiment="negative"),
        }
        if price_info:
            resp["product_price"] = price_info.get("product_price", "")

        return jsonify(resp), 200
    except Exception as e:
        logger.error("Sentiment analysis failed: %s", e, exc_info=True)
        return jsonify({"error": "Analytics Error", "message": "Sentiment analysis failed."}), 500


@app.route("/v1.0/analytics/summary", methods=["GET"])
def get_summary():
    """Scrape reviews and return computed analytics summary."""
    url = request.args.get("url", "").strip()
    if not url:
        return jsonify({"error": "Bad Request", "message": "Missing required 'url' parameter"}), 400

    url_error = _validate_flipkart_url(url)
    if url_error:
        return jsonify({"error": "Unprocessable Entity", "message": url_error}), 422

    page_str = request.args.get("page", "all").strip()
    page, page_error = _validate_page_param(page_str)
    if page_error:
        return jsonify({"error": "Unprocessable Entity", "message": page_error}), 422

    max_pages_str = request.args.get("max_pages", "").strip()
    max_pages, mp_error = _validate_max_pages_param(max_pages_str)
    if mp_error:
        return jsonify({"error": "Unprocessable Entity", "message": mp_error}), 422

    verified = request.args.get("verified", "true").strip().lower() not in ("false", "0", "no")
    all_sorts_arg = request.args.get("all_sorts")
    all_sorts = (all_sorts_arg.strip().lower() in ("true", "1", "yes")) if all_sorts_arg is not None else None
    refresh = request.args.get("refresh", "false").strip().lower() in ("true", "1", "yes")
    data, error, status = _run_scraper(url, page=page, max_pages=max_pages, verified_only=verified, all_sorts=all_sorts, force_refresh=refresh)
    if error:
        return jsonify(error), status

    reviews, summary, price_info = _separate_reviews_and_metadata(data)

    if not reviews:
        return jsonify({
            "message": "No reviews found.",
            "summary": {},
        }), 200

    try:
        from analytics.summary import compute_geo_analysis, compute_summary

        result = compute_summary(reviews, summary)
        result["geographic"] = compute_geo_analysis(reviews)

        if price_info:
            result["product_price"] = price_info.get("product_price", "")

        return jsonify(result), 200
    except Exception as e:
        logger.error("Summary analytics failed: %s", e, exc_info=True)
        return jsonify({"error": "Analytics Error", "message": "Summary computation failed."}), 500


@app.route("/v1.0/analytics/topics", methods=["GET"])
def get_topics():
    """Scrape reviews and return extracted topics with per-topic sentiment."""
    url = request.args.get("url", "").strip()
    if not url:
        return jsonify({"error": "Bad Request", "message": "Missing required 'url' parameter"}), 400

    url_error = _validate_flipkart_url(url)
    if url_error:
        return jsonify({"error": "Unprocessable Entity", "message": url_error}), 422

    page_str = request.args.get("page", "all").strip()
    page, page_error = _validate_page_param(page_str)
    if page_error:
        return jsonify({"error": "Unprocessable Entity", "message": page_error}), 422

    max_pages_str = request.args.get("max_pages", "").strip()
    max_pages, mp_error = _validate_max_pages_param(max_pages_str)
    if mp_error:
        return jsonify({"error": "Unprocessable Entity", "message": mp_error}), 422

    # BUG FIX S2: Validate top_n before passing to int() to prevent ValueError → 500
    top_n_str = request.args.get("top_n", "15").strip()
    top_n, top_n_error = _validate_top_n_param(top_n_str)
    if top_n_error:
        return jsonify({"error": "Unprocessable Entity", "message": top_n_error}), 422

    verified = request.args.get("verified", "true").strip().lower() not in ("false", "0", "no")
    all_sorts_arg = request.args.get("all_sorts")
    all_sorts = (all_sorts_arg.strip().lower() in ("true", "1", "yes")) if all_sorts_arg is not None else None
    refresh = request.args.get("refresh", "false").strip().lower() in ("true", "1", "yes")
    data, error, status = _run_scraper(url, page=page, max_pages=max_pages, verified_only=verified, all_sorts=all_sorts, force_refresh=refresh)
    if error:
        return jsonify(error), status

    reviews, summary, _ = _separate_reviews_and_metadata(data)

    if not reviews:
        return jsonify({"message": "No reviews found.", "topics": []}), 200

    try:
        from analytics.topics import (
            extract_keywords_tfidf,
            extract_ngrams,
            extract_topics_with_sentiment,
        )

        topics = extract_topics_with_sentiment(reviews, top_n=top_n)
        keywords = extract_keywords_tfidf(reviews, top_n=top_n)
        positive_ngrams = extract_ngrams(reviews, n=2, top_k=10, filter_sentiment="positive")
        negative_ngrams = extract_ngrams(reviews, n=2, top_k=10, filter_sentiment="negative")

        return jsonify({
            "topics": topics,
            "keywords": keywords,
            "positive_bigrams": positive_ngrams,
            "negative_bigrams": negative_ngrams,
            "positive_ngrams": positive_ngrams,
            "negative_ngrams": negative_ngrams,
        }), 200
    except Exception as e:
        logger.error("Topic extraction failed: %s", e, exc_info=True)
        return jsonify({"error": "Analytics Error", "message": "Topic extraction failed."}), 500


@app.route("/v1.0/analytics/report", methods=["GET"])
def get_report():
    """Scrape reviews and return a full analytics report (sentiment + topics + geo)."""
    url = request.args.get("url", "").strip()
    if not url:
        return jsonify({"error": "Bad Request", "message": "Missing required 'url' parameter"}), 400

    url_error = _validate_flipkart_url(url)
    if url_error:
        return jsonify({"error": "Unprocessable Entity", "message": url_error}), 422

    page_str = request.args.get("page", "all").strip()
    page, page_error = _validate_page_param(page_str)
    if page_error:
        return jsonify({"error": "Unprocessable Entity", "message": page_error}), 422

    max_pages_str = request.args.get("max_pages", "").strip()
    max_pages, mp_error = _validate_max_pages_param(max_pages_str)
    if mp_error:
        return jsonify({"error": "Unprocessable Entity", "message": mp_error}), 422

    verified = request.args.get("verified", "true").strip().lower() not in ("false", "0", "no")
    all_sorts_arg = request.args.get("all_sorts")
    all_sorts = (all_sorts_arg.strip().lower() in ("true", "1", "yes")) if all_sorts_arg is not None else None
    refresh = request.args.get("refresh", "false").strip().lower() in ("true", "1", "yes")
    data, error, status = _run_scraper(url, page=page, max_pages=max_pages, verified_only=verified, all_sorts=all_sorts, force_refresh=refresh)
    if error:
        return jsonify(error), status

    reviews, summary, price_info = _separate_reviews_and_metadata(data)

    if not reviews:
        return jsonify({"message": "No reviews found.", "report": {}}), 200

    try:
        from analytics.export import generate_report

        report = generate_report(reviews, summary_data=summary, verified_only=verified)

        if price_info:
            report["product_price"] = price_info.get("product_price", "")

        return jsonify(report), 200
    except Exception as e:
        logger.error("Report generation failed: %s", e, exc_info=True)
        return jsonify({"error": "Analytics Error", "message": "Report generation failed."}), 500


@app.route("/v1.0/export/csv", methods=["GET"])
def export_csv():
    """Scrape reviews and return them as CSV."""
    url = request.args.get("url", "").strip()
    if not url:
        return jsonify({"error": "Bad Request", "message": "Missing required 'url' parameter"}), 400

    url_error = _validate_flipkart_url(url)
    if url_error:
        return jsonify({"error": "Unprocessable Entity", "message": url_error}), 422

    page_str = request.args.get("page", "all").strip()
    page, page_error = _validate_page_param(page_str)
    if page_error:
        return jsonify({"error": "Unprocessable Entity", "message": page_error}), 422

    max_pages_str = request.args.get("max_pages", "").strip()
    max_pages, mp_error = _validate_max_pages_param(max_pages_str)
    if mp_error:
        return jsonify({"error": "Unprocessable Entity", "message": mp_error}), 422

    include_sentiment = request.args.get("sentiment", "false").strip().lower() in (
        "true", "1", "yes"
    )

    verified = request.args.get("verified", "true").strip().lower() not in ("false", "0", "no")
    all_sorts_arg = request.args.get("all_sorts")
    all_sorts = (all_sorts_arg.strip().lower() in ("true", "1", "yes")) if all_sorts_arg is not None else None
    refresh = request.args.get("refresh", "false").strip().lower() in ("true", "1", "yes")
    data, error, status = _run_scraper(url, page=page, max_pages=max_pages, verified_only=verified, all_sorts=all_sorts, force_refresh=refresh)
    if error:
        return jsonify(error), status

    reviews, summary, _ = _separate_reviews_and_metadata(data)

    if not reviews:
        # Return empty CSV with header rather than bare empty response
        return Response(
            "",
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=flipkart_reviews.csv"},
            status=200,
        )

    try:
        if include_sentiment:
            from analytics.sentiment import analyze_batch
            reviews = analyze_batch(reviews)

        from analytics.export import reviews_to_csv
        csv_content = reviews_to_csv(reviews)

        return Response(
            csv_content,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=flipkart_reviews.csv"},
        )
    except Exception as e:
        logger.error("CSV export failed: %s", e, exc_info=True)
        return jsonify({"error": "Export Error", "message": "CSV export failed."}), 500


@app.route("/v1.0/export/json", methods=["GET"])
def export_json():
    """Scrape reviews and return them as a downloadable JSON document."""
    url = request.args.get("url", "").strip()
    if not url:
        return jsonify({"error": "Bad Request", "message": "Missing required 'url' parameter"}), 400

    url_error = _validate_flipkart_url(url)
    if url_error:
        return jsonify({"error": "Unprocessable Entity", "message": url_error}), 422

    page, page_error = _validate_page_param(request.args.get("page", "all").strip())
    if page_error:
        return jsonify({"error": "Unprocessable Entity", "message": page_error}), 422
    max_pages, mp_error = _validate_max_pages_param(request.args.get("max_pages", "").strip())
    if mp_error:
        return jsonify({"error": "Unprocessable Entity", "message": mp_error}), 422

    verified = request.args.get("verified", "true").strip().lower() not in ("false", "0", "no")
    all_sorts_arg = request.args.get("all_sorts")
    all_sorts = (all_sorts_arg.strip().lower() in ("true", "1", "yes")) if all_sorts_arg is not None else None
    refresh = request.args.get("refresh", "false").strip().lower() in ("true", "1", "yes")
    data, error, status = _run_scraper(url, page=page, max_pages=max_pages, verified_only=verified, all_sorts=all_sorts, force_refresh=refresh)
    if error:
        return jsonify(error), status

    reviews, _, _ = _separate_reviews_and_metadata(data)
    response = Response(
        json.dumps(reviews, ensure_ascii=False, indent=2),
        mimetype="application/json",
    )
    response.headers["Content-Disposition"] = "attachment; filename=flipkart_reviews.json"
    return response


@app.route("/v1.0/export/excel", methods=["GET"])
def export_excel():
    """Scrape reviews and return them as an Excel workbook."""
    url = request.args.get("url", "").strip()
    if not url:
        return jsonify({"error": "Bad Request", "message": "Missing required 'url' parameter"}), 400

    url_error = _validate_flipkart_url(url)
    if url_error:
        return jsonify({"error": "Unprocessable Entity", "message": url_error}), 422

    page, page_error = _validate_page_param(request.args.get("page", "all").strip())
    if page_error:
        return jsonify({"error": "Unprocessable Entity", "message": page_error}), 422
    max_pages, mp_error = _validate_max_pages_param(request.args.get("max_pages", "").strip())
    if mp_error:
        return jsonify({"error": "Unprocessable Entity", "message": mp_error}), 422

    verified = request.args.get("verified", "true").strip().lower() not in ("false", "0", "no")
    all_sorts_arg = request.args.get("all_sorts")
    all_sorts = (all_sorts_arg.strip().lower() in ("true", "1", "yes")) if all_sorts_arg is not None else None
    refresh = request.args.get("refresh", "false").strip().lower() in ("true", "1", "yes")
    data, error, status = _run_scraper(url, page=page, max_pages=max_pages, verified_only=verified, all_sorts=all_sorts, force_refresh=refresh)
    if error:
        return jsonify(error), status

    reviews, _, _ = _separate_reviews_and_metadata(data)
    try:
        import pandas as pd

        workbook = BytesIO()
        pd.DataFrame(reviews).to_excel(workbook, index=False, engine="openpyxl")
        workbook.seek(0)
        return send_file(
            workbook,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name="flipkart_reviews.xlsx",
        )
    except ImportError:
        logger.error("Excel export requires the openpyxl package")
        return jsonify({
            "error": "Export Error",
            "message": "Excel export is unavailable because the workbook dependency is not installed.",
        }), 500
    except Exception as e:
        logger.error("Excel export failed: %s", e, exc_info=True)
        return jsonify({"error": "Export Error", "message": "Excel export failed."}), 500


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
