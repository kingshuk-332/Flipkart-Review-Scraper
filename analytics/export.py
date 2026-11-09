"""
Export utilities for review data and analytics results.

Supports CSV export and structured JSON report generation.
"""

import csv
import io
import json
from datetime import datetime


def reviews_to_csv(reviews):
    """
    Convert a list of review dicts to CSV format.

    Args:
        reviews: list of review dicts.

    Returns:
        str — CSV-formatted string.
    """
    if not reviews:
        return ""

    # Collect all possible fields across all reviews
    fieldnames = []
    seen = set()
    for review in reviews:
        for key in review.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for review in reviews:
        writer.writerow(review)

    return output.getvalue()


def generate_report(reviews, summary_data=None, include_sentiment=True, include_topics=True, verified_only=True):
    """
    Generate a comprehensive JSON analytics report.

    Combines scraping results with analytics into a single structured report.

    Args:
        reviews: list of individual review dicts.
        summary_data: optional product summary dict from scraper.
        include_sentiment: whether to run sentiment analysis.
        include_topics: whether to extract topics.
        verified_only: whether verified-only filtering was applied.

    Returns:
        dict — Complete analytics report.
    """
    from analytics.sentiment import aggregate_sentiment, analyze_batch
    from analytics.summary import compute_geo_analysis, compute_summary
    from analytics.topics import extract_ngrams, extract_topics_with_sentiment

    report = {
        "report_generated_at": datetime.now().isoformat(),
        "data_source": "Flipkart Review Scraper",
        "total_reviews": len(reviews),
        "verified_only": verified_only,
    }

    # Statistical summary
    report["summary"] = compute_summary(reviews, summary_data)

    # Sentiment analysis
    if include_sentiment and reviews:
        analyzed = analyze_batch(reviews)
        report["sentiment"] = aggregate_sentiment(analyzed)

        # Top mismatched reviews (interesting anomalies)
        mismatches = [r for r in analyzed if r.get("score_text_mismatch")]
        if mismatches:
            report["score_text_mismatches"] = [
                {
                    "review_title": r.get("review_title", ""),
                    "review_score": r.get("review_score", ""),
                    "sentiment_label": r.get("sentiment_label", ""),
                    "sentiment_score": r.get("sentiment_score", 0),
                    "review_content": r.get("review_content", "")[:200],
                }
                for r in mismatches[:10]  # Limit to top 10
            ]
    else:
        analyzed = reviews

    # Topic extraction
    if include_topics and reviews:
        report["topics"] = extract_topics_with_sentiment(reviews, analyzed)
        pos_ng = extract_ngrams(reviews, n=2, top_k=10, filter_sentiment="positive")
        neg_ng = extract_ngrams(reviews, n=2, top_k=10, filter_sentiment="negative")
        report["positive_ngrams"] = pos_ng
        report["negative_ngrams"] = neg_ng
        report["positive_bigrams"] = pos_ng
        report["negative_bigrams"] = neg_ng

    # Geographic analysis
    report["geographic"] = compute_geo_analysis(reviews)

    # Scraped and sentiment-annotated reviews
    report["reviews"] = analyzed

    return report
