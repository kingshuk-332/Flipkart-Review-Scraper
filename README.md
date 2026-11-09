# Flipkart Review API & Analytics Engine

A production-grade Python REST API, CLI tool, and analytics dashboard that scrapes Flipkart product reviews, ratings, and metadata using Scrapy. Supports single-page, multi-page, and full review extraction. Returns structured JSON with validated review scores, text, verified reviewer metadata, geographic breakdown, TF-IDF topic modeling, and VADER sentiment analytics. Auto-bootstraps on Windows, macOS, and Linux — just run `python run.py`.

---

## Architecture Overview

```
                      ┌──────────────────────┐
                      │    Client / Web UI   │
                      │ (index.html / cURL)  │
                      └──────────┬───────────┘
                                 │ HTTP GET
                                 ▼
                      ┌──────────────────────┐
                      │   Flask API (app.py) │
                      │  - SSRF/URL Guard    │
                      │  - Param Validation  │
                      └─────┬──────────┬─────┘
                            │          │
         Scrape Subprocess  │          │ Analyzed In-Memory
                            ▼          ▼
┌──────────────────────────────┐     ┌───────────────────────────────────┐
│ Scrapy Spider (reviewSpider) │     │ Analytics Suite                   │
│ - Initial State Parser       │     │ - VADER Sentiment (sentiment.py)  │
│ - HTML DOM Fallback          │     │ - Statistics & Geo (summary.py)   │
│ - Deduplication & Clean Loc  │     │ - TF-IDF Topics (topics.py)       │
└──────────────────────────────┘     │ - CSV / JSON Export (export.py)   │
                                     └───────────────────────────────────┘
```

---

## Quick Start

```bash
python run.py                 # Auto-setup venv & start API server
python run.py crawl -a url="<FlipkartURL>" -O output.json  # Run scraper CLI
python run.py test            # Run automated test suite
```

### Platform Launchers

| Platform | Command |
|---|---|
| macOS / Linux | `./run.sh` |
| Windows (CMD) | `run.bat` |
| Windows (PowerShell) | `.\run.ps1` |
| Docker Compose | `docker compose up --build` |

---

## API Endpoints & Usage

### 1. Scraping Endpoints
- **GET `/v1.0/reviews?url=<URL>&page=1`**
  Fetch reviews for a specific page.
- **GET `/v1.0/reviews?url=<URL>&max_pages=5`**
  Fetch up to `max_pages` pages sequentially.
- **GET `/v1.0/reviews?url=<URL>&page=all`**
  Crawl all review pages.

### 2. Analytics Endpoints
- **GET `/v1.0/analytics/sentiment?url=<URL>&page=all`**
  Individual review sentiment classification (`POSITIVE`, `NEGATIVE`, `NEUTRAL`), compound polarity score, and aggregate percentage distribution (guaranteed 100.0% sum) with star-text mismatch detection.
- **GET `/v1.0/analytics/summary?url=<URL>&max_pages=5`**
  Comprehensive statistics: average rating, median rating, standard deviation, controversy/polarization score, helpfulness ratio, review quality metrics, certified buyer ratio, and geographic state/city breakdown.
- **GET `/v1.0/analytics/topics?url=<URL>&top_n=15`**
  TF-IDF keyword extraction, n-grams, and per-topic sentiment calculated strictly from reviews mentioning each topic.
- **GET `/v1.0/analytics/report?url=<URL>&max_pages=5`**
  Full combined analytics report (summary + sentiment + topics + geo).
- **GET `/v1.0/export/csv?url=<URL>&sentiment=true`**
  Stream or download reviews directly as a CSV spreadsheet.

### 3. Web UI Workspaces (Area Design System)
The user interface has been completely redesigned around the **Area** modern editorial design language (warm off-white `#F7F6F0`, deep olive `#566344`, light sage `#DCE4D2`, sand cards `#EAE5D9`, and elegant serif typography).

- **Landing Page (`/`)**: Editorial marketing homepage featuring "Understand Every Review.", panoramic analytics preview, 4-column benefits grid, interactive geospatial density cards, and product showcase.
- **Workspace 01 — Full Report (`/full-report`)**: Complete review intelligence with all 6 statistical KPIs, VADER sentiment doughnut, rating vs. sentiment cross-analysis, review velocity, TF-IDF topics, n-grams, geographic state distribution, and full review collection.
- **Workspace 02 — Sentiment & Mismatches (`/sentiment-mismatches`)**: Focused emotional analysis, suspicious rating-versus-text contradiction detection (flags 5★ complaints and 1★ praises), mismatch ratios, and polarity distributions.
- **Workspace 03 — Raw Reviews Only (`/raw-reviews`)**: Clean, spacious customer review table focused strictly on the verified review dataset, search, and direct CSV/JSON downloads without analytics chart overhead.

### 4. Utility Endpoints
- **GET `/health`** — Service health status check.
- **GET `/sample-data`** — Offline sample review dataset with full analytics for instant dashboard demonstration.
- **GET `/api-info`** — API documentation and endpoint registry.

---

## Review Data Schema

Every extracted review conforms to this validated schema:

| Field | Type | Description |
|---|---|---|
| `review_id` | String | Unique review identifier (permalink or author-title-body hash) |
| `review_score` | String | Star rating (1 to 5) |
| `review_title` | String | Headline/title of the review |
| `review_content` | String | Cleaned textual content of the review |
| `reviewer_name` | String | Name of author or "Flipkart Customer" |
| `reviewer_location` | String | Clean "City, State" format (no leading comma) |
| `reviewer_date` | String | Relative or absolute date string |
| `permalink` | String | Direct URL path to review on Flipkart |
| `certified_buyer` | Boolean | Verification flag (`true` or `false`) |
| `upvote` | String | Helpful count |
| `downvote` | String | Unhelpful count |

---

## Running the Automated Test Suite

Run all unit, integration, and mathematical validation tests:

```bash
pytest tests/ -v
```
or via the runner:
```bash
python run.py test
```

---

## Security & Reliability Hardening

1. **SSRF & Host Restriction**: Strictly blocks private IPs, AWS/GCP metadata endpoints (`169.254.169.254`), loopbacks, and non-Flipkart domain requests.
2. **Input Sanitization**: Query parameters (`page`, `max_pages`, `top_n`) are validated with explicit bounds and 422 HTTP responses.
3. **Information Disclosure Protection**: Generic and honest error messages prevent leaking internal server traces or system paths.
4. **Resilient Parser**: Dual extraction path featuring React hydration state (`window.__INITIAL_STATE__`) and multi-selector DOM fallback.
