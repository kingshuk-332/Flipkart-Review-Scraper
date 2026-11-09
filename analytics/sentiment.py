"""
Sentiment analysis for Flipkart product reviews.

Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) which is
specifically tuned for social media and user-generated content — making it
ideal for product reviews without needing large model downloads.

Notes on accuracy:
  - VADER works well for English-language social media text.
  - Code-mixed Indic-English reviews will have reduced accuracy.
  - Sarcasm and complex irony are NOT reliably detected; this is documented
    as a known limitation.
  - Rating–text mismatch detection is a heuristic flag, not a classification.
"""

import re
import statistics

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Lazy-initialized singleton to avoid repeated setup cost
_analyzer = None


# Domain-specific lexicon enhancements for e-commerce and electronics product reviews
_DOMAIN_LEXICON = {
    # Defect and product complaints (not present or neutral in standard VADER)
    "issue": -1.6,
    "issues": -1.6,
    "heating": -1.8,
    "overheating": -2.6,
    "overheats": -2.5,
    "defect": -2.4,
    "defective": -2.5,
    "drain": -1.6,
    "draining": -1.8,
    "drains": -1.7,
    "lag": -1.6,
    "lagging": -1.8,
    "lags": -1.7,
    "hang": -1.6,
    "hanging": -1.8,
    "hangs": -1.8,
    "crash": -2.2,
    "crashing": -2.4,
    "crashes": -2.3,
    "faulty": -2.3,
    "scam": -3.2,
    "fake": -3.0,
    "useless": -2.6,
    "waste": -2.6,
    "disappointed": -2.4,
    "disappointing": -2.4,
    "cheap": -1.4,
    "blurry": -1.5,
    "stuck": -1.6,
    "damage": -2.0,
    "damaged": -2.3,
    # High-signal positive review terms
    "top-notch": 2.6,
    "topnotch": 2.6,
    "superb": 2.6,
    "worth": 1.9,
    "flawless": 2.7,
    "seamless": 2.2,
    "premium": 2.0,
    "stunning": 2.6,
}


def _get_analyzer():
    """Return a cached VADER SentimentIntensityAnalyzer instance with domain lexicon."""
    global _analyzer
    if _analyzer is None:
        import nltk
        # Download VADER lexicon if not already present (silent, one-time)
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        _analyzer = SentimentIntensityAnalyzer()
        _analyzer.lexicon.update(_DOMAIN_LEXICON)
    return _analyzer


def _clean_text(text):
    """Normalize review text for sentiment analysis."""
    if not text:
        return ""
    # Remove HTML entities like &rsquo; &amp; etc.
    text = re.sub(r"&\w+;", " ", text)
    # Remove emojis and special unicode characters (keep basic punctuation)
    text = re.sub(
        r"[^\w\s.,!?;:'\"-]", " ", text, flags=re.UNICODE
    )
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def analyze_single(review_text, review_title=""):
    """
    Analyze sentiment of a single review.

    Args:
        review_text: The main review body text.
        review_title: The review title/headline (optional).

    Returns:
        dict with keys: label, compound, positive, neutral, negative

    Note:
        VADER's compound score thresholds:
          >= 0.05  → POSITIVE
          <= -0.05 → NEGATIVE
          else     → NEUTRAL
        These are standard VADER thresholds and suitable for review text.
    """
    sia = _get_analyzer()

    # Combine title and text for full context (title carries strong signal)
    combined = f"{review_title}. {review_text}" if review_title else review_text
    combined = _clean_text(combined)

    if not combined:
        return {
            "label": "NEUTRAL",
            "compound": 0.0,
            "positive": 0.0,
            "neutral": 1.0,
            "negative": 0.0,
        }

    scores = sia.polarity_scores(combined)

    # Classify using VADER compound thresholds
    compound = scores["compound"]
    if compound >= 0.05:
        label = "POSITIVE"
    elif compound <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return {
        "label": label,
        "compound": round(compound, 4),
        "positive": round(scores["pos"], 4),
        "neutral": round(scores["neu"], 4),
        "negative": round(scores["neg"], 4),
    }


def detect_score_text_mismatch(review_score, sentiment_label):
    """
    Detect when a review's star rating contradicts its text sentiment.

    Examples of mismatches:
      - 5-star review with NEGATIVE sentiment text
      - 1-star review with POSITIVE sentiment text

    This is a heuristic flag. It does not reclassify the review's sentiment;
    it only signals that the rating and text are in disagreement.

    Args:
        review_score: Integer star rating (1-5), or string representation.
        sentiment_label: One of "POSITIVE", "NEGATIVE", "NEUTRAL".

    Returns:
        bool — True if there is a mismatch.
    """
    try:
        score = int(str(review_score).replace(",", "").strip())
    except (ValueError, TypeError):
        return False

    if score < 1 or score > 5:
        return False

    if score >= 4 and sentiment_label == "NEGATIVE":
        return True
    if score <= 2 and sentiment_label == "POSITIVE":
        return True
    return False


def analyze_batch(reviews):
    """
    Analyze sentiment for a list of review dicts.

    Each review dict should have at minimum:
      - review_content (str)
      - review_title (str, optional)
      - review_score (str or int, optional)

    Returns:
        list of enriched review dicts with added sentiment fields.
    """
    results = []
    for review in reviews:
        content = review.get("review_content") or ""
        title = review.get("review_title") or ""
        score = review.get("review_score", "")

        sentiment = analyze_single(content, title)
        mismatch = detect_score_text_mismatch(score, sentiment["label"])

        enriched = {
            **review,
            "sentiment_label": sentiment["label"],
            "sentiment_score": sentiment["compound"],
            "sentiment_positive": sentiment["positive"],
            "sentiment_neutral": sentiment["neutral"],
            "sentiment_negative": sentiment["negative"],
            "score_text_mismatch": mismatch,
        }
        results.append(enriched)

    return results


def aggregate_sentiment(analyzed_reviews):
    """
    Compute aggregate sentiment statistics across all analyzed reviews.

    Sentiment percentages are guaranteed to sum to exactly 100.0 by
    assigning the rounding residual to the largest bucket.

    Args:
        analyzed_reviews: list of dicts from analyze_batch().

    Returns:
        dict with overall sentiment summary.
    """
    if not analyzed_reviews:
        return {
            "total_reviews_analyzed": 0,
            "avg_sentiment_score": 0.0,
            "sentiment_distribution": {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0},
            "mismatch_count": 0,
            "mismatch_percentage": 0.0,
        }

    compounds = [r.get("sentiment_score", 0.0) for r in analyzed_reviews]
    labels = [r.get("sentiment_label", "NEUTRAL") for r in analyzed_reviews]
    mismatches = [r for r in analyzed_reviews if r.get("score_text_mismatch")]

    distribution = {
        "POSITIVE": labels.count("POSITIVE"),
        "NEGATIVE": labels.count("NEGATIVE"),
        "NEUTRAL": labels.count("NEUTRAL"),
    }

    total = len(analyzed_reviews)

    # --- BUG FIX A1: Guarantee percentages sum to exactly 100.0 ---
    # Round each percentage independently, then assign the floating-point
    # residual to the bucket with the largest raw count to avoid 99.9 / 100.1.
    raw_pos = distribution["POSITIVE"] / total * 100
    raw_neg = distribution["NEGATIVE"] / total * 100
    raw_neu = distribution["NEUTRAL"] / total * 100

    pos_pct = round(raw_pos, 1)
    neg_pct = round(raw_neg, 1)
    neu_pct = round(raw_neu, 1)

    rounding_error = round(100.0 - pos_pct - neg_pct - neu_pct, 1)
    if rounding_error != 0.0:
        # Apply residual to the bucket with the highest raw count
        max_key = max(distribution, key=distribution.get)
        if max_key == "POSITIVE":
            pos_pct = round(pos_pct + rounding_error, 1)
        elif max_key == "NEGATIVE":
            neg_pct = round(neg_pct + rounding_error, 1)
        else:
            neu_pct = round(neu_pct + rounding_error, 1)

    return {
        "total_reviews_analyzed": total,
        "avg_sentiment_score": round(statistics.mean(compounds), 4),
        "median_sentiment_score": round(statistics.median(compounds), 4),
        "sentiment_std_dev": round(statistics.stdev(compounds), 4) if total > 1 else 0.0,
        "sentiment_distribution": distribution,
        "positive_pct": pos_pct,
        "negative_pct": neg_pct,
        "neutral_pct": neu_pct,
        "mismatch_count": len(mismatches),
        "mismatch_percentage": round(len(mismatches) / total * 100, 1),
    }
