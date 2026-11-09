"""
Spider unit tests — kept minimal here since comprehensive tests
are in test_analytics.py (TestReviewSpider class).
"""
import json
import unittest
from scrapy.http import HtmlResponse
from reviewapi.spiders.reviewSpider import ReviewSpider


class TestReviewSpider(unittest.TestCase):
    def setUp(self):
        self.spider = ReviewSpider(
            url="https://www.flipkart.com/sample-product/p/itm12345?pid=PID123",
            page=2
        )

    def test_build_review_url(self):
        url = "https://www.flipkart.com/sample-product/p/itm12345?pid=PID123"
        rev_url = self.spider._build_review_url(url, page=3)
        self.assertIn("/product-reviews/", rev_url)
        self.assertIn("page=3", rev_url)
        self.assertIn("pid=PID123", rev_url)

    def test_extract_initial_state(self):
        sample_state = {"multiWidgetState": {"testKey": "testValue"}}
        html = f"""
        <html>
            <head>
                <script id="is_script">window.__INITIAL_STATE__ = {json.dumps(sample_state)};</script>
            </head>
            <body></body>
        </html>
        """
        response = HtmlResponse(
            url="https://www.flipkart.com/sample",
            body=html,
            encoding="utf-8"
        )
        extracted = self.spider._extract_initial_state(response)
        self.assertIsNotNone(extracted)
        self.assertEqual(extracted["multiWidgetState"]["testKey"], "testValue")

    def test_pagination_modes(self):
        s_single = ReviewSpider(url="https://www.flipkart.com/sample/p/123", page=1)
        self.assertFalse(s_single.scrape_all)
        self.assertFalse(s_single.all_sorts)
        self.assertEqual(s_single.target_max_pages, 1)

        s_all = ReviewSpider(url="https://www.flipkart.com/sample/p/123", page="all")
        self.assertTrue(s_all.scrape_all)
        # BUG FIX: all_sorts is now False by default — previously True
        self.assertFalse(s_all.all_sorts)
        self.assertIsNone(s_all.target_max_pages)

        s_max = ReviewSpider(url="https://www.flipkart.com/sample/p/123", max_pages=5)
        self.assertTrue(s_max.scrape_all)
        self.assertFalse(s_max.all_sorts)
        self.assertEqual(s_max.target_max_pages, 5)

        s_custom_sort = ReviewSpider(
            url="https://www.flipkart.com/sample/p/123", all_sorts=True
        )
        self.assertTrue(s_custom_sort.all_sorts)

    def test_parse_reviews_with_initial_state(self):
        spider = ReviewSpider(
            url="https://www.flipkart.com/sample-product/product-reviews/itm123?pid=PID123",
            page=1,
        )
        mock_state = {
            "multiWidgetState": {
                "pageDataResponse": {
                    "pageNumber": 1,
                    "seoData": {"seo": {"keywords": "Test Product Reviews"}}
                },
                "widgetsData": {
                    "slots": [
                        {
                            "slotData": {
                                "widget": {
                                    "type": "RATING",
                                    "data": {
                                        "rating": {
                                            "value": {
                                                "average": 4.5,
                                                "ratingCount": 1000,
                                                "reviewCount": 200,
                                                "histogram": [50, 50, 100, 300, 500]
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        {
                            "slotData": {
                                "widget": {
                                    "type": "REVIEWS",
                                    "data": {
                                        "renderableComponents": [
                                            {
                                                "value": {
                                                    "rating": 5,
                                                    "title": "Great purchase",
                                                    "text": "Loved the product!",
                                                    "author": "Jane Doe",
                                                    "location": {"city": "Mumbai", "state": "Maharashtra"},
                                                    "created": "Jan, 2024",
                                                    "url": "/reviews/12345",
                                                    "upvote": {"value": {"count": 25}},
                                                    "downvote": {"value": {"count": 2}},
                                                    "certifiedBuyer": True,
                                                }
                                            }
                                        ]
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        }
        html = f"""
        <html>
            <head>
                <script id="is_script">window.__INITIAL_STATE__ = {json.dumps(mock_state)};</script>
            </head>
            <body></body>
        </html>
        """
        response = HtmlResponse(
            url="https://www.flipkart.com/sample/product-reviews/itm123?pid=PID123&page=1",
            body=html,
            encoding="utf-8"
        )
        items = list(spider.parse_reviews(response))
        self.assertEqual(len(items), 2)

        summary = items[0]
        self.assertEqual(summary["review_title"], "Test Product Reviews")
        self.assertEqual(summary["avg_rating"], "4.5")
        self.assertEqual(summary["total_ratings"], "1,000 Ratings")
        self.assertEqual(summary["total_reviews"], "200 Reviews")
        self.assertEqual(summary["5"], "500")
        self.assertEqual(summary["1"], "50")

        review = items[1]
        self.assertEqual(review["review_score"], "5")
        self.assertEqual(review["review_title"], "Great purchase")
        self.assertEqual(review["review_content"], "Loved the product!")
        self.assertEqual(review["reviewer_name"], "Jane Doe")
        # BUG FIX B7: No leading ", " in location
        self.assertEqual(review["reviewer_location"], "Mumbai, Maharashtra")
        self.assertFalse(review["reviewer_location"].startswith(","))
        self.assertEqual(review["upvote"], "25")
        self.assertEqual(review["downvote"], "2")
        # BUG FIX B3: review_id is now present
        self.assertIn("review_id", review)


if __name__ == "__main__":
    unittest.main()
