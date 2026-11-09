# Scrapy settings for reviewapi project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = 'reviewapi'

SPIDER_MODULES = ['reviewapi.spiders']
NEWSPIDER_MODULE = 'reviewapi.spiders'

# Realistic desktop browser User-Agent to prevent bot blocking / 429 / 500 errors
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'

# Obey robots.txt rules
# NOTE: Flipkart's robots.txt disallows /product-reviews/* and /reviews/*,
# which causes Scrapy to drop review page requests when set to True.
ROBOTSTXT_OBEY = False

# Configure maximum concurrent requests performed by Scrapy
CONCURRENT_REQUESTS = 4

# Configure a delay for requests for the same website to avoid rate limiting
DOWNLOAD_DELAY = 1.0

# Enable cookies (needed for Flipkart session handling)
COOKIES_ENABLED = True

# Override the default request headers to mimic a genuine browser request
DEFAULT_REQUEST_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'sec-ch-ua': '"Chromium";v="128", "Google Chrome";v="128", "Not?A_Brand";v="99"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'none',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
}

# Retry settings for transient network/rate-limiting responses
RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 522, 524, 408, 429]

# Export JSON feeds with UTF-8 encoding
FEED_EXPORT_ENCODING = 'utf-8'
