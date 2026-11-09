"""
Keyword and topic extraction from Flipkart product reviews.

Uses TF-IDF (Term Frequency–Inverse Document Frequency) to identify
the most distinctive terms in review text, with optional sentiment
scoring per topic. No heavy ML models required.
"""

import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer

# Common stop words for review text (English + e-commerce specific)
_STOP_WORDS = {
    # Standard English stop words
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
    "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o", "re",
    "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven",
    "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren",
    "won", "wouldn",
    # E-commerce / review specific filler words
    "product", "buy", "bought", "purchase", "purchased", "flipkart", "amazon",
    "online", "delivery", "delivered", "order", "ordered", "customer", "review",
    "star", "stars", "rating", "rated", "recommend", "recommended", "overall",
    "also", "like", "one", "got", "get", "use", "used", "using", "would", "could",
    "really", "much", "even", "thing", "things", "lot", "bit", "well", "still",
    "every", "make", "made", "day", "days", "month", "months", "week", "time",
    "first", "second", "last", "new", "old", "good", "bad", "best", "worst",
    "nice", "great", "awesome", "excellent", "amazing", "fabulous", "super",
    "wonderful", "fantastic", "terrible", "horrible", "awful", "poor", "phone",
    "mobile", "guys", "guy", "reason", "giving", "gives", "give", "advertisement",
    "ad", "highly", "please", "anyone", "everyone", "someone", "observed",
    "noticeable", "persists", "there", "mins", "minutes", "said", "say", "saying",
    "think", "thought", "look", "looks", "seem", "seems", "going", "goes", "went",
}

# Strong domain indicators for sentiment phrase categorization
_CRITICAL_INDICATORS = {
    "issue", "issues", "heating", "overheating", "overheats", "heat", "problem", "problems",
    "drain", "draining", "drains", "drainage", "lag", "lagging", "lags", "hang", "hanging", "hangs",
    "crash", "crashing", "crashes", "defect", "defective", "faulty", "scam", "fake",
    "waste", "useless", "disappointed", "disappointing", "damage", "damaged", "broken",
    "poor", "bad", "worst", "terrible", "horrible", "awful", "slow", "blurry", "stuck",
    "fail", "failed", "failing", "glitch", "noise", "hot", "flimsy", "cheap", "drainage",
    "cracked", "delay", "delayed", "struggle", "disgusting", "flicker", "flickering",
    "below average",
}

_PRAISE_INDICATORS = {
    "good", "great", "excellent", "awesome", "super", "superb", "best", "perfect",
    "love", "loved", "top", "notch", "worth", "nice", "crystal", "clear", "smooth",
    "fast", "premium", "stunning", "fantastic", "wonderful", "flawless", "impressive",
    "mindblowing", "delightful", "satisfying", "value", "fabulous", "brilliant",
}

_JUNK_WORDS = {
    "phone", "mobile", "device", "handset", "flipkart", "amazon", "online", "product",
    "guys", "guy", "reason", "giving", "gives", "give", "advertisement", "ad", "highly",
    "bought", "buy", "purchase", "purchased", "ordered", "order", "delivery", "delivered",
    "seller", "day", "days", "month", "months", "week", "weeks", "time", "times",
    "customer", "review", "reviews", "star", "stars", "rating", "ratings",
    "moto", "motorola", "edge", "fusion", "vivo", "realme", "samsung", "oppo", "poco",
    "oneplus", "xiaomi", "redmi", "iqoo", "apple", "iphone", "pantone", "pro", "lite",
    "said", "say", "saying", "think", "thought", "mins", "minutes", "hours", "there",
    "also", "even", "got", "get", "use", "used", "using", "well", "thing", "things", "segment",
}

_FEATURE_WORDS = {
    "battery", "camera", "display", "performance", "charging", "sound", "screen", "build",
    "quality", "design", "backup", "life", "weight", "value", "money", "look", "looks",
    "speaker", "processor", "gaming", "fingerprint", "sensor", "refresh", "rate",
}


def _safe_int(val):
    """
    Safely convert a value to int, returning 0 on failure.

    BUG FIX A3: Uses str().replace(",", "") to handle comma-formatted
    numbers like "1,234" that are common in scraped Flipkart data.
    Also handles string floats like "5.0".
    """
    try:
        return int(float(str(val).replace(",", "").strip()))
    except (ValueError, TypeError):
        return 0


def _clean_for_tfidf(text):
    """Preprocess text for TF-IDF extraction."""
    if not text:
        return ""
    # Remove HTML entities
    text = re.sub(r"&\w+;", " ", text)
    # Remove special characters (keep alphanumerics and spaces)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    # Lowercase and collapse whitespace
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _count_ngram_occurrences(texts, ngram):
    """
    BUG FIX A4: Count actual bigram/ngram occurrences rather than summing
    component word frequencies.

    Previous code computed:
        freq = word_freq["battery"] + word_freq["life"]
    for the bigram "battery life", which is wrong — it counts each word's
    frequency separately, not the co-occurrence.

    This function counts how many times the exact ngram appears across all texts.
    """
    pattern = re.compile(r"\b" + re.escape(ngram) + r"\b")
    return sum(len(pattern.findall(text)) for text in texts)


def extract_keywords_tfidf(reviews, top_n=20, ngram_range=(1, 2)):
    """
    Extract top keywords from review text using TF-IDF.

    TF-IDF measures how distinctive a term is across the review corpus.
    Terms that appear frequently in a document but rarely across all documents
    receive higher scores.

    Args:
        reviews: list of review dicts with 'review_content' field.
        top_n: Number of top keywords to return.
        ngram_range: Tuple (min_n, max_n) for n-gram extraction.

    Returns:
        list of dicts: [{"keyword": str, "tfidf_score": float, "frequency": int}, ...]
    """
    texts = [
        _clean_for_tfidf(str(r.get("review_content") or ""))
        for r in reviews
    ]
    texts = [t for t in texts if t and len(t.split()) >= 2]

    if not texts:
        return []

    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=ngram_range,
        stop_words=list(_STOP_WORDS),
        min_df=2,   # Term must appear in at least 2 documents
        max_df=0.85,  # Ignore terms appearing in > 85% of documents
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        # Not enough documents or vocabulary is empty after filtering
        return []

    feature_names = vectorizer.get_feature_names_out()
    # Average TF-IDF score across all documents for each term
    avg_scores = tfidf_matrix.mean(axis=0).A1

    scored = []
    for idx, score in enumerate(avg_scores):
        term = feature_names[idx]
        # BUG FIX A4: Count actual ngram occurrences, not sum of component words
        freq = _count_ngram_occurrences(texts, term)
        scored.append({
            "keyword": term,
            "tfidf_score": round(float(score), 4),
            "frequency": freq,
        })

    # Sort by TF-IDF score descending
    scored.sort(key=lambda x: x["tfidf_score"], reverse=True)
    return scored[:top_n]


def extract_topics_with_sentiment(reviews, analyzed_reviews=None, top_n=15):
    """
    Extract topics and compute per-topic sentiment from review data.

    Identifies the most discussed product aspects and determines whether
    sentiment around each topic is positive, negative, or mixed.

    Topic sentiment is computed ONLY from reviews that mention the topic,
    not from all reviews in the corpus.

    Args:
        reviews: list of review dicts with 'review_content'.
        analyzed_reviews: optional list of sentiment-analyzed reviews
                          (from sentiment.analyze_batch). If None,
                          sentiment is computed on-the-fly.
        top_n: Number of topics to return.

    Returns:
        list of topic dicts with keyword, mentions, avg_sentiment, and label.
    """
    from analytics.sentiment import analyze_single

    keywords = extract_keywords_tfidf(reviews, top_n=top_n * 2, ngram_range=(1, 2))

    if not keywords:
        return []

    # For each keyword, find reviews mentioning it and compute topic sentiment
    # Topic sentiment uses ONLY reviews mentioning that topic.
    topics = []
    for kw_entry in keywords:
        keyword = kw_entry["keyword"]
        pattern = re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)

        mention_sentiments = []
        mention_count = 0

        source = analyzed_reviews if analyzed_reviews else reviews
        for review in source:
            content = (
                str(review.get("review_content") or "") + " " +
                str(review.get("review_title") or "")
            ).lower()
            if pattern.search(content):
                mention_count += 1
                if analyzed_reviews and "sentiment_score" in review:
                    mention_sentiments.append(review["sentiment_score"])
                else:
                    sent = analyze_single(
                        review.get("review_content") or "",
                        review.get("review_title") or "",
                    )
                    mention_sentiments.append(sent["compound"])

        if mention_count < 2:
            # Skip topics with fewer than 2 mentions — insufficient evidence
            continue

        avg_sent = (
            sum(mention_sentiments) / len(mention_sentiments)
            if mention_sentiments
            else 0.0
        )

        if avg_sent >= 0.05:
            label = "POSITIVE"
        elif avg_sent <= -0.05:
            label = "NEGATIVE"
        else:
            label = "MIXED"

        topics.append({
            "topic": keyword,
            "mentions": mention_count,
            "avg_sentiment": round(avg_sent, 4),
            "sentiment_label": label,
        })

    # Sort by mention count descending, take top_n
    topics.sort(key=lambda x: x["mentions"], reverse=True)
    return topics[:top_n]


def extract_ngrams(reviews, n=2, top_k=20, filter_sentiment=None):
    """
    Extract top n-grams from reviews, filtered and classified by phrase sentiment.

    - "positive" phrases: praise and high-performing features (e.g. "battery life", "top notch",
      "fast charging"). Terms indicating issues (e.g. "heating issue", "battery drain") are strictly
      excluded, even if mentioned in 4-5 star reviews.
    - "negative" phrases: critical complaints and product defects (e.g. "heating issue", "overheating",
      "battery drain", "camera lag"). Sourced from low-rated reviews AND from critical phrases in
      higher-rated reviews so prominent complaints are never lost.
    - None: unfiltered top n-grams across all reviews.

    Args:
        reviews: list of review dicts.
        n: n-gram size (1=unigrams, 2=bigrams, 3=trigrams).
        top_k: Number of top n-grams to return.
        filter_sentiment: Optional — "positive", "negative", or None.

    Returns:
        list of dicts: [{"ngram": str, "count": int}, ...]
    """
    if not reviews:
        return []

    # Active stop words for n-grams: preserve critical/praise indicators and negations
    # while eliminating brand names and generic conversational junk
    active_stops = (
        {w for w in _STOP_WORDS if w not in _CRITICAL_INDICATORS and w not in _PRAISE_INDICATORS and w not in ("not", "no", "never")}
        | _JUNK_WORDS
    )

    candidates = []
    unigram_fallback = []
    negation_tokens = ("not", "no", "never")

    for review in reviews:
        score_val = _safe_int(review.get("review_score", 0))
        raw_content = str(review.get("review_content") or "")

        # Split on sentence boundaries and contrasting conjunctions so n-grams do not
        # cross clause boundaries (prevents nonsense bigrams like "nice heating")
        clauses = re.split(r"[.,!?;:\n\-]+|\b(?:but|however|although|except|while)\b", raw_content, flags=re.IGNORECASE)

        for clause in clauses:
            cleaned = _clean_for_tfidf(clause)
            words = [
                w for w in cleaned.split()
                if (w not in active_stops and len(w) > 2) or w in negation_tokens
            ]

            for i in range(len(words) - n + 1):
                ngram_words = words[i : i + n]
                if any(w in _JUNK_WORDS for w in ngram_words):
                    continue

                ngram = " ".join(ngram_words)

                has_critical = any(w in _CRITICAL_INDICATORS for w in ngram_words) or any(w in negation_tokens for w in ngram_words)
                has_praise = any(w in _PRAISE_INDICATORS for w in ngram_words)
                has_feature = any(w in _FEATURE_WORDS for w in ngram_words)

                if filter_sentiment == "positive":
                    # Exclude any phrase containing critical defect terms or negations
                    if has_critical:
                        continue
                    # Must be praise or product feature from positive/unspecified review
                    if (has_praise or has_feature) and (score_val >= 4 or score_val == 0):
                        candidates.append(ngram)

                elif filter_sentiment == "negative":
                    # Exclude phrases containing pure unmitigated praise
                    if has_praise and not has_critical:
                        continue
                    # Sourced from low ratings OR any review with explicit critical defect terms / negations
                    if has_critical or (score_val > 0 and score_val <= 3):
                        candidates.append(ngram)

                else:
                    candidates.append(ngram)

            # Collect unigram fallback candidates for small review sets
            for w in words:
                if w not in _JUNK_WORDS and len(w) > 2:
                    if filter_sentiment == "negative" and (w in _CRITICAL_INDICATORS or (score_val > 0 and score_val <= 2)):
                        if w not in _PRAISE_INDICATORS:
                            unigram_fallback.append(w)
                    elif filter_sentiment == "positive" and w in _PRAISE_INDICATORS and (score_val >= 4 or score_val == 0):
                        unigram_fallback.append(w)

    counts = Counter(candidates).most_common(top_k)

    # If bigrams yielded fewer than 3 items, supplement with high-signal unigrams
    if len(counts) < 3 and unigram_fallback:
        existing_words = set(" ".join(k for k, _ in counts).split())
        for ug, cnt in Counter(unigram_fallback).most_common(top_k):
            if ug not in existing_words:
                counts.append((ug, cnt))
            if len(counts) >= top_k:
                break

    return [{"ngram": ngram, "count": count} for ngram, count in counts]
