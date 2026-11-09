# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class ReviewapiItem(scrapy.Item):
    pass


class ProductPriceItem(scrapy.Item):
    product_price = scrapy.Field()
    requested_url = scrapy.Field()


class ReviewSummaryItem(scrapy.Item):
    review_title = scrapy.Field()
    avg_rating = scrapy.Field()
    total_ratings = scrapy.Field()
    total_reviews = scrapy.Field()
    total_pages = scrapy.Field()
    star_5 = scrapy.Field()
    star_4 = scrapy.Field()
    star_3 = scrapy.Field()
    star_2 = scrapy.Field()
    star_1 = scrapy.Field()


class ReviewItem(scrapy.Item):
    review_id = scrapy.Field()
    review_score = scrapy.Field()
    review_title = scrapy.Field()
    review_content = scrapy.Field()
    reviewer_name = scrapy.Field()
    reviewer_location = scrapy.Field()
    reviewer_date = scrapy.Field()
    permalink = scrapy.Field()
    certified_buyer = scrapy.Field()
    upvote = scrapy.Field()
    downvote = scrapy.Field()
