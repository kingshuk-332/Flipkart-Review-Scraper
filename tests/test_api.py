"""
API endpoint and UI routing tests.
Validates both REST API endpoints and the Area editorial HTML pages.
"""
import unittest
from app import app


class TestAppAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_root_endpoint_json_for_api_clients(self):
        """API clients without Accept: text/html get the JSON metadata."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("name"), "Flipkart Review API")
        self.assertIn("/v1.0/reviews", data.get("endpoints", {}))

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "healthy")

    def test_sample_data_endpoint(self):
        response = self.client.get("/sample-data")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("summary", data)
        self.assertIn("sentiment", data)
        self.assertIn("reviews", data)
        self.assertIn("positive_ngrams", data)
        self.assertIn("negative_ngrams", data)
        self.assertGreater(len(data["reviews"]), 0)

    def test_missing_url_parameter(self):
        response = self.client.get("/v1.0/reviews")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get("error"), "Bad Request")
        self.assertIn("Missing required 'url' parameter", data.get("message", ""))

    def test_invalid_url_returns_422(self):
        response = self.client.get("/v1.0/reviews?url=not-a-url")
        self.assertEqual(response.status_code, 422)

    def test_non_flipkart_url_returns_422(self):
        response = self.client.get("/v1.0/reviews?url=https://amazon.in/product")
        self.assertEqual(response.status_code, 422)

    def test_localhost_url_returns_422(self):
        response = self.client.get("/v1.0/reviews?url=http://localhost:8080")
        self.assertEqual(response.status_code, 422)

    def test_invalid_top_n_returns_422_not_500(self):
        """Previously raised ValueError → 500. Now returns 422."""
        response = self.client.get(
            "/v1.0/analytics/topics?url=https://www.flipkart.com/p/id&top_n=abc"
        )
        self.assertEqual(response.status_code, 422)

    # -----------------------------------------------------------------------
    # Area Editorial UI Pages Tests
    # -----------------------------------------------------------------------

    def test_landing_page_html(self):
        """Browser accessing / gets the Area editorial landing page."""
        response = self.client.get("/", headers={"Accept": "text/html"})
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        # Check Area editorial headline & content
        self.assertIn("Understand Every Review.", html)
        self.assertIn("We've cracked the code.", html)
        self.assertIn("See the Big Picture.", html)
        self.assertIn("Full Report", html)
        self.assertIn("Sentiment &amp; Mismatches", html)
        self.assertIn("Raw Reviews Only", html)
        self.assertIn("Start Scraping ↗", html)

    def test_full_report_page_html(self):
        """Accessing /full-report gets Workspace 01 with complete analytics."""
        response = self.client.get("/full-report", headers={"Accept": "text/html"})
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Complete Review Intelligence.", html)
        self.assertIn("starBreakdownBars", html)
        self.assertIn("sentimentChart", html)
        self.assertIn("ratingVsSentimentChart", html)
        self.assertIn("topicsContainer", html)
        self.assertIn("kpiAvgRating", html)
        self.assertIn("kpiControversy", html)
        self.assertIn("Load Sample Data", html)
        self.assertIn("productScrapedCount", html)

    def test_sentiment_mismatches_page_html(self):
        """Accessing /sentiment-mismatches gets Workspace 02 focusing on sentiment and anomalies."""
        response = self.client.get("/sentiment-mismatches", headers={"Accept": "text/html"})
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Understand What Customers Feel.", html)
        self.assertIn("sentimentChart", html)
        self.assertIn("kpiMismatchCount", html)
        self.assertIn("What are Review Mismatches?", html)
        self.assertIn("ratingVsSentimentChart", html)

    def test_raw_reviews_page_html(self):
        """Accessing /raw-reviews gets Workspace 03 with clean table, without chart clutter."""
        response = self.client.get("/raw-reviews", headers={"Accept": "text/html"})
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Every Review. One Place.", html)
        self.assertIn("reviewsFeed", html)
        self.assertIn("Download CSV", html)
        self.assertIn("Download JSON", html)


if __name__ == "__main__":
    unittest.main()
