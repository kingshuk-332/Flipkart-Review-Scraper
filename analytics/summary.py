"""
Aggregate review analytics and statistical summary.

Computes derived metrics from raw scraped review data including
helpfulness ratios, controversy scores, review velocity, rating
distribution analysis, and reviewer demographics.

Formulas and interpretations are documented inline.
"""

import math
import re
import statistics
from collections import Counter
from datetime import datetime, timedelta


def _safe_int(val):
    """
    Safely convert a value to int, returning 0 on failure.

    Handles comma-formatted numbers like "1,234" common in scraped data.
    Also handles string floats like "5.0".
    Returns 0 (not None) so callers get a neutral zero rather than crashing.
    Missing values are treated as 0 ONLY for vote counts; callers should
    separately handle missing ratings by filtering zero values.
    """
    try:
        return int(float(str(val).replace(",", "").strip()))
    except (ValueError, TypeError):
        return 0


def _safe_float(val):
    """Safely convert a value to float, returning 0.0 on failure."""
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return 0.0


def _parse_date(date_str):
    """
    Parse Flipkart review dates into approximate datetime objects.

    Handles multiple formats:
    1. Relative: "5 months ago", "3 days ago"
    2. Absolute month-year: "Jan, 2024", "October 2023"
    3. ISO: "2024-01-15"
    4. Flipkart short: "15 Jan, 2024"

    Returns:
        datetime or None if parsing fails.
    """
    if not date_str:
        return None
    date_str = str(date_str).strip()

    # 1. Relative dates: "N unit(s) ago"
    rel_match = re.match(
        r"(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago",
        date_str,
        re.IGNORECASE,
    )
    if rel_match:
        amount = int(rel_match.group(1))
        unit = rel_match.group(2).lower()
        now = datetime.now()
        if unit == "second":
            return now - timedelta(seconds=amount)
        elif unit == "minute":
            return now - timedelta(minutes=amount)
        elif unit == "hour":
            return now - timedelta(hours=amount)
        elif unit == "day":
            return now - timedelta(days=amount)
        elif unit == "week":
            return now - timedelta(weeks=amount)
        elif unit == "month":
            return now - timedelta(days=amount * 30)
        elif unit == "year":
            return now - timedelta(days=amount * 365)

    # 2. "15 Jan, 2024" or "15 January 2024"
    full_match = re.match(
        r"(\d{1,2})\s+([A-Za-z]+)[,\s]+(\d{4})",
        date_str,
    )
    if full_match:
        for fmt in ("%d %b %Y", "%d %B %Y"):
            try:
                return datetime.strptime(
                    f"{full_match.group(1)} {full_match.group(2)} {full_match.group(3)}",
                    fmt,
                )
            except ValueError:
                continue

    # 3. "Jan, 2024" or "January 2024"
    month_year_match = re.match(
        r"([A-Za-z]+)[,\s]+(\d{4})$",
        date_str,
    )
    if month_year_match:
        for fmt in ("%b %Y", "%B %Y"):
            try:
                return datetime.strptime(
                    f"{month_year_match.group(1)} {month_year_match.group(2)}",
                    fmt,
                )
            except ValueError:
                continue

    # 4. ISO format "2024-01-15"
    iso_match = re.match(r"(\d{4})-(\d{2})-(\d{2})", date_str)
    if iso_match:
        try:
            return datetime(
                int(iso_match.group(1)),
                int(iso_match.group(2)),
                int(iso_match.group(3)),
            )
        except ValueError:
            pass

    return None


def compute_summary(reviews, summary_data=None):
    """
    Compute comprehensive analytics summary from review data.

    Args:
        reviews: list of individual review dicts (with review_score, upvote, etc.).
        summary_data: optional dict with aggregate metadata from scraper
                      (avg_rating, total_ratings, star distribution, etc.).

    Returns:
        dict with computed analytics metrics.

    Key formulas:
        avg_rating      = mean(valid scores)          [excludes missing/invalid]
        median_rating   = median(valid scores)
        std_deviation   = sample std dev of scores    [requires ≥2 observations]
        controversy     = normalized (std_dev/2) * (0.5 + extreme_ratio*0.5)
                          where extreme_ratio = (5-star + 1-star) / total_scored
                          clamped to [0, 1]
        helpfulness     = upvotes / (upvotes + downvotes) per review, averaged
        certified_pct   = certified_buyers / total_reviews * 100
    """
    if not reviews:
        return {"total_reviews": 0, "message": "No reviews to analyze"}

    # --- Valid scores only (1–5 range) ---
    # BUG FIX A2: Use 'total_reviews' key (not 'error') for empty case.
    # BUG FIX: Exclude 0/invalid scores from rating calculations to avoid
    # treating missing ratings as 0 (which would skew averages downward).
    raw_scores = [_safe_int(r.get("review_score", 0)) for r in reviews]
    scores = [s for s in raw_scores if 1 <= s <= 5]

    upvotes = [_safe_int(r.get("upvote", 0)) for r in reviews]
    downvotes = [_safe_int(r.get("downvote", 0)) for r in reviews]

    # --- Basic Rating Stats ---
    avg_rating = round(statistics.mean(scores), 2) if scores else None
    median_rating = statistics.median(scores) if scores else None
    rating_std_dev = round(statistics.stdev(scores), 2) if len(scores) > 1 else 0.0

    # Rating distribution from sampled reviews
    # Formula: count of each star / total scored reviews * 100
    rating_dist = Counter(scores)
    total_scored = len(scores)
    rating_distribution = {}
    for star in range(5, 0, -1):
        count = rating_dist.get(star, 0)
        pct = round(count / total_scored * 100, 1) if total_scored else 0.0
        rating_distribution[f"{star}_star"] = {"count": count, "percentage": pct}

    # --- Positive/Negative review percentages ---
    # Positive = 4 or 5 stars; Negative = 1 or 2 stars; Neutral = 3 stars
    positive_reviews = sum(1 for s in scores if s >= 4)
    negative_reviews = sum(1 for s in scores if s <= 2)
    neutral_reviews = sum(1 for s in scores if s == 3)
    positive_pct = round(positive_reviews / total_scored * 100, 1) if total_scored else 0.0
    negative_pct = round(negative_reviews / total_scored * 100, 1) if total_scored else 0.0

    # --- Controversy Score ---
    # Formula: controversy = min(1, (std_dev/2) * (0.5 + extreme_ratio*0.5))
    # Interpretation:
    #   extreme_ratio = proportion of 1-star or 5-star reviews (bimodal signal)
    #   std_dev/2 normalizes std_dev (max ~2 for 1-5 scale) to 0-1
    #   Higher score = more polarized opinion
    # Range: [0, 1]. Labels: ≥0.6 HIGHLY_POLARIZING, ≥0.35 MODERATELY_POLARIZING
    if total_scored > 1:
        extreme_count = rating_dist.get(5, 0) + rating_dist.get(1, 0)
        extreme_ratio = extreme_count / total_scored
        # Normalize std_dev: max theoretical std on 1-5 scale ≈ 2.0
        normalized_std = min(1.0, rating_std_dev / 2.0)
        controversy_score = round(
            min(1.0, normalized_std * (0.5 + extreme_ratio * 0.5)), 2
        )
    else:
        controversy_score = 0.0

    if controversy_score >= 0.6:
        controversy_label = "HIGHLY_POLARIZING"
    elif controversy_score >= 0.35:
        controversy_label = "MODERATELY_POLARIZING"
    else:
        controversy_label = "CONSENSUS"

    # --- Helpfulness Analysis ---
    # Formula: helpfulness_ratio = upvotes / (upvotes + downvotes) per review
    # Only computed for reviews with at least one vote.
    # avg_helpfulness_ratio = mean of per-review ratios (not global sum ratio)
    helpfulness_ratios = []
    for i, r in enumerate(reviews):
        up = upvotes[i]
        down = downvotes[i]
        total_votes = up + down
        if total_votes > 0:
            helpfulness_ratios.append(up / total_votes)

    avg_helpfulness = (
        round(statistics.mean(helpfulness_ratios), 4) if helpfulness_ratios else None
    )
    total_upvotes = sum(upvotes)
    total_downvotes = sum(downvotes)

    # --- Review Quality Metrics ---
    content_lengths = [
        len(str(r.get("review_content") or "")) for r in reviews
    ]
    avg_review_length = round(statistics.mean(content_lengths)) if content_lengths else 0
    median_review_length = int(statistics.median(content_lengths)) if content_lengths else 0

    # Short review: < 20 chars (likely a single word or emoji)
    # Detailed review: > 200 chars
    short_reviews = sum(1 for l in content_lengths if l < 20)
    detailed_reviews = sum(1 for l in content_lengths if l > 200)

    # --- Certified Buyer Analysis ---
    # BUG FIX: Only count True boolean values, not truthy strings.
    certified_count = sum(
        1 for r in reviews if r.get("certified_buyer") is True
    )
    certified_pct = round(certified_count / len(reviews) * 100, 1) if reviews else 0.0

    # --- Time / Velocity Analysis ---
    # Formula: reviews_per_day = count_of_dated_reviews / date_range_days
    # Only computed when ≥2 reviews have parseable dates.
    # NOTE: Absolute dates from Flipkart (e.g. "Jan, 2024") are approximate;
    # velocity figures should be interpreted as estimates.
    dates = []
    for r in reviews:
        dt = _parse_date(r.get("reviewer_date", ""))
        if dt:
            dates.append(dt)

    velocity = {}
    if len(dates) >= 2:
        dates_sorted = sorted(dates)
        date_range = (dates_sorted[-1] - dates_sorted[0]).days
        if date_range > 0:
            velocity = {
                "reviews_per_day": round(len(dates) / date_range, 2),
                "reviews_per_week": round(len(dates) / date_range * 7, 2),
                "date_range_days": date_range,
                "earliest_date": dates_sorted[0].strftime("%Y-%m-%d"),
                "latest_date": dates_sorted[-1].strftime("%Y-%m-%d"),
            }

    # --- Compose Result ---
    result = {
        "total_reviews_analyzed": len(reviews),
        "reviews_with_valid_scores": total_scored,
        "rating_stats": {
            "avg_rating": avg_rating,
            "median_rating": median_rating,
            "std_deviation": rating_std_dev,
            "distribution": rating_distribution,
            "positive_reviews_pct": positive_pct,
            "negative_reviews_pct": negative_pct,
        },
        "controversy": {
            "score": controversy_score,
            "label": controversy_label,
        },
        "helpfulness": {
            "avg_helpfulness_ratio": avg_helpfulness,
            "total_upvotes": total_upvotes,
            "total_downvotes": total_downvotes,
            "reviews_with_votes": len(helpfulness_ratios),
        },
        "review_quality": {
            "avg_length_chars": avg_review_length,
            "median_length_chars": median_review_length,
            "short_reviews_count": short_reviews,
            "short_reviews_pct": (
                round(short_reviews / len(reviews) * 100, 1) if reviews else 0.0
            ),
            "detailed_reviews_count": detailed_reviews,
            "detailed_reviews_pct": (
                round(detailed_reviews / len(reviews) * 100, 1) if reviews else 0.0
            ),
        },
        "certified_buyers": {
            "count": certified_count,
            "percentage": certified_pct,
        },
    }

    if velocity:
        result["review_velocity"] = velocity

    # Include scraper summary data if available
    if summary_data:
        total_ratings = re.sub(r"[\s&]+$", "", str(summary_data.get("total_ratings", "")).replace("&", "")).strip()
        total_reviews = re.sub(r"^[\s&]+", "", str(summary_data.get("total_reviews", "")).replace("&", "")).strip()
        if total_ratings and not total_ratings.lower().endswith("ratings"):
            total_ratings = f"{total_ratings} Ratings"
        if total_reviews and not total_reviews.lower().endswith("reviews"):
            total_reviews = f"{total_reviews} Reviews"
        result["product_summary"] = {
            "title": summary_data.get("review_title", ""),
            "avg_rating_flipkart": summary_data.get("avg_rating", ""),
            "total_ratings": total_ratings,
            "total_reviews": total_reviews,
            "star_distribution_flipkart": {
                "5": summary_data.get("5", ""),
                "4": summary_data.get("4", ""),
                "3": summary_data.get("3", ""),
                "2": summary_data.get("2", ""),
                "1": summary_data.get("1", ""),
            },
        }

    return result


def compute_geo_analysis(reviews):
    """
    Analyze review distribution and sentiment by geographic location.

    Args:
        reviews: list of review dicts with 'reviewer_location' field.

    Returns:
        dict with by_state and by_city breakdowns.

    Note:
        Location data comes directly from scraped content. Only locations
        explicitly provided by reviewers are included. No location data
        is inferred or fabricated.
    """
    state_data = {}
    city_data = {}
    reviews_with_location = 0

    for review in reviews:
        loc = (review.get("reviewer_location") or "").strip().strip(",").strip()
        if not loc:
            continue
        reviews_with_location += 1

        # BUG FIX B7: Locations from spider may have leading ", " prefix.
        # Strip that before splitting.
        loc = loc.lstrip(", ").strip()
        if not loc:
            continue

        parts = [p.strip() for p in loc.split(",") if p.strip()]
        city = parts[0] if len(parts) >= 1 else ""
        state = parts[1] if len(parts) >= 2 else (parts[0] if len(parts) == 1 else "")

        score = _safe_int(review.get("review_score", 0))
        valid_score = score if 1 <= score <= 5 else None

        if state:
            if state not in state_data:
                state_data[state] = {"scores": [], "count": 0}
            state_data[state]["count"] += 1
            if valid_score is not None:
                state_data[state]["scores"].append(valid_score)

        if city and city != state:
            if city not in city_data:
                city_data[city] = {"scores": [], "count": 0}
            city_data[city]["count"] += 1
            if valid_score is not None:
                city_data[city]["scores"].append(valid_score)

    # Build sorted results
    by_state = []
    for state, data in sorted(
        state_data.items(), key=lambda x: x[1]["count"], reverse=True
    ):
        entry = {
            "state": state,
            "review_count": data["count"],
            "share_of_reviews": round(data["count"] / len(reviews) * 100, 1),
        }
        if data["scores"]:
            entry["avg_rating"] = round(statistics.mean(data["scores"]), 2)
        by_state.append(entry)

    by_city = []
    for city, data in sorted(
        city_data.items(), key=lambda x: x[1]["count"], reverse=True
    )[:30]:
        entry = {
            "city": city,
            "review_count": data["count"],
            "share_of_reviews": round(data["count"] / len(reviews) * 100, 1),
        }
        if data["scores"]:
            entry["avg_rating"] = round(statistics.mean(data["scores"]), 2)
        by_city.append(entry)

    return {
        "total_locations": len(state_data),
        "total_reviews": len(reviews),
        "reviews_with_location": reviews_with_location,
        "by_state": by_state,
        "top_cities": by_city,
    }
