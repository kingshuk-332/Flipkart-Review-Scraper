import json
import math
import re
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import scrapy


class ReviewSpider(scrapy.Spider):
    name = "review"

    SORT_ORDERS = ["MOST_HELPFUL", "MOST_RECENT", "NEGATIVE_FIRST", "POSITIVE_FIRST"]

    def __init__(
        self,
        url="",
        page=1,
        max_pages=None,
        sort=None,
        all_sorts=None,
        verified_only=True,
        *args,
        **kwargs,
    ):
        super(ReviewSpider, self).__init__(*args, **kwargs)
        self.start_url = (url or "").strip()
        self.sort = sort
        self.verified_only = str(verified_only).strip().lower() in ("true", "1", "yes")

        # --- Page mode parsing ---
        page_str = str(page).strip().lower()
        if page_str in ("all", "0", "*"):
            self.scrape_all = True
            self.start_page = 1
            self.target_max_pages = None
        else:
            self.scrape_all = False
            try:
                self.start_page = max(1, int(page))
            except (ValueError, TypeError):
                self.start_page = 1

        if max_pages is not None:
            mp_str = str(max_pages).strip().lower()
            if mp_str in ("all", "0", "*", "none"):
                self.scrape_all = True
                self.target_max_pages = None
            else:
                try:
                    mp = int(mp_str)
                    if mp >= 1:
                        self.target_max_pages = mp
                        self.scrape_all = True
                    # max_pages=0 or negative: treat as invalid, ignore
                except (ValueError, TypeError):
                    pass
        else:
            if not self.scrape_all:
                self.target_max_pages = self.start_page

        # all_sorts: whether to crawl all Flipkart sort orders (produces 4× requests)
        # BUG NOTE B5: all_sorts=True by default for page=all increases network load
        # significantly and risks rate-limiting. Default changed to False (single sort)
        # unless the user explicitly sets all_sorts=True.
        if all_sorts is not None:
            self.all_sorts = str(all_sorts).strip().lower() in ("true", "1", "yes", "all")
        else:
            # Conservative default: only crawl one sort order
            self.all_sorts = False

        # BUG FIX A5: summary_yielded and seen_reviews must be instance-level
        # (they are) so concurrent requests within the same spider instance share state.
        # Scrapy's async model means parse_reviews may be called concurrently for
        # different pages; the seen_reviews set prevents duplicate yields correctly.
        self.summary_yielded = False
        self.seen_reviews = set()
        self.seen_page_review_ids = set()
        self.seen_review_ids_per_sort = {s: set() for s in self.SORT_ORDERS}
        self.seen_review_ids_per_sort["default"] = set()
        self.total_pages = 1

    async def start(self):
        """Asynchronous entry point for Scrapy 2.13+."""
        for request in self.start_requests():
            yield request

    def start_requests(self):
        """Synchronous entry point for Scrapy backward compatibility."""
        if not self.start_url:
            self.logger.error("No URL provided. Use -a url=<FlipkartURL>")
            return

        # If it's a product page URL (/p/), scrape product page first for price,
        # then follow review page
        if "/p/" in self.start_url and "/product-reviews/" not in self.start_url:
            yield scrapy.Request(
                url=self.start_url,
                callback=self.parse_product,
                meta={"current_page": self.start_page},
                dont_filter=True,
            )
        else:
            if self.all_sorts:
                for s_order in self.SORT_ORDERS:
                    rev_url = self._build_review_url(self.start_url, 1, sort_order=s_order)
                    yield scrapy.Request(
                        url=rev_url,
                        callback=self.parse_reviews,
                        meta={"current_page": 1, "sort_order": s_order},
                        dont_filter=True,
                    )
            else:
                review_url = self._build_review_url(
                    self.start_url, self.start_page, sort_order=self.sort
                )
                yield scrapy.Request(
                    url=review_url,
                    callback=self.parse_reviews,
                    meta={"current_page": self.start_page, "sort_order": self.sort},
                    dont_filter=True,
                )

    def _build_review_url(self, url, page=1, sort_order=None):
        parsed = urlparse(url)
        path = parsed.path
        if "/p/" in path:
            path = path.replace("/p/", "/product-reviews/")

        query = parse_qs(parsed.query)
        query["page"] = [str(page)]
        if sort_order:
            query["sortOrder"] = [str(sort_order)]
        new_query = urlencode(query, doseq=True)

        scheme = parsed.scheme or "https"
        netloc = parsed.netloc or "www.flipkart.com"
        return urlunparse((scheme, netloc, path, parsed.params, new_query, parsed.fragment))

    def _extract_initial_state(self, response):
        """
        Extract Flipkart React hydration state (window.__INITIAL_STATE__).

        Employs multiple extraction strategies:
        1. Explicit script#is_script via xpath/css.
        2. Any script tag containing window.__INITIAL_STATE__.
        3. Global JSON raw_decode to accurately parse complete state JSON
           without regex end-of-string truncation or delimiter failure.
        """
        raw_text = response.text if hasattr(response, "text") else str(response)

        # Strategy 1: Find script#is_script
        script_text = response.xpath('//script[@id="is_script"]/text()').get()
        if not script_text:
            script_text = response.css("script#is_script::text").get()

        # Strategy 2: Find any script tag containing window.__INITIAL_STATE__
        if not script_text:
            for s in response.xpath('//script[contains(text(), "__INITIAL_STATE__")]/text()').getall():
                if "__INITIAL_STATE__" in s:
                    script_text = s
                    break

        if script_text:
            try:
                idx = script_text.find("{")
                if idx != -1:
                    decoder = json.JSONDecoder()
                    obj, _ = decoder.raw_decode(script_text, idx)
                    if isinstance(obj, dict):
                        return obj
            except Exception:
                pass

            try:
                json_str = re.sub(
                    r"^\s*window\.__INITIAL_STATE__\s*=\s*", "", script_text.strip()
                )
                json_str = re.sub(r";\s*$", "", json_str)
                return json.loads(json_str)
            except Exception as e:
                self.logger.warning("Failed to decode __INITIAL_STATE__ JSON: %s", e)

        # Strategy 3: Global search in raw HTML using raw_decode
        match = re.search(r"window\.__INITIAL_STATE__\s*=\s*", raw_text)
        if match:
            try:
                brace_idx = raw_text.find("{", match.end())
                if brace_idx != -1:
                    decoder = json.JSONDecoder()
                    obj, _ = decoder.raw_decode(raw_text, brace_idx)
                    if isinstance(obj, dict):
                        return obj
            except Exception as e:
                self.logger.warning("Failed to raw_decode __INITIAL_STATE__ from HTML: %s", e)

        return None

    def _format_location(self, loc_data):
        """
        Format location dict to a clean string without leading separators.

        BUG FIX B7: Previous code emitted ", city, state" with leading ", "
        which caused empty first element when geo analysis split on ",".

        Returns:
            Clean location string like "Mumbai, Maharashtra" or "" if no data.
        """
        if not isinstance(loc_data, dict):
            return ""
        city = (loc_data.get("city") or "").strip()
        state = (loc_data.get("state") or "").strip()
        if city and state:
            return f"{city}, {state}"
        if city:
            return city
        if state:
            return state
        return ""

    @staticmethod
    def _is_certified_buyer(value):
        """Return True for explicit certified buyer values or badges."""
        if isinstance(value, bool):
            return value
        if isinstance(value, dict):
            if value.get("certifiedBuyer") is True:
                return True
            badge = str(value.get("authorBadge") or "").lower()
            return "certified" in badge
        val_str = str(value).strip().lower()
        return val_str in ("true", "1", "yes", "certified", "certified buyer")

    def _make_review_id(self, val, permalink):
        """
        BUG FIX B3: Generate a stable review identifier.

        Prefer permalink (unique per Flipkart review).
        Fall back to author+title+content hash for deduplication.
        """
        if permalink:
            return permalink.strip()
        author = (val.get("author") or "").strip()
        title = (val.get("title") or "").strip()
        text = (val.get("text") or "").strip()
        # Use first 50 chars of content to create a composite key
        return f"{author}|{title}|{text[:50]}"

    @staticmethod
    def _clean_product_title(title, url=""):
        """Clean generic Flipkart placeholder titles and extract actual product name from URL."""
        generic_substrings = (
            "buy products online",
            "all categories",
            "online at best price",
        )
        title_str = (title or "").strip()
        lower_title = title_str.lower()
        is_generic = (
            not title_str
            or lower_title in ("product reviews", "reviews", "flipkart", "flipkart.com")
            or any(m in lower_title for m in generic_substrings)
        )
        if is_generic and url:
            parsed_path = urlparse(url).path
            parts = [p for p in parsed_path.split("/") if p and p not in ("p", "product-reviews", "reviews") and not p.lower().startswith("itm")]
            if parts:
                slug = parts[0].replace("-", " ").strip()
                if slug:
                    clean_name = " ".join(w.capitalize() for w in slug.split())
                    return f"{clean_name} Reviews"
        return title_str

    def parse_product(self, response):
        """Parses the Flipkart product page to extract product price and navigates to reviews."""
        price = None
        state = self._extract_initial_state(response)

        if state:
            schemas = (
                state.get("multiWidgetState", {})
                .get("pageDataResponse", {})
                .get("seoData", {})
                .get("schema", [])
            )
            for s in schemas:
                if isinstance(s, dict) and "offers" in s:
                    offers = s.get("offers", {})
                    if isinstance(offers, dict) and "price" in offers:
                        price = str(offers["price"])
                        break

            if not price:
                events = (
                    state.get("multiWidgetState", {})
                    .get("pageDataResponse", {})
                    .get("pageContext", {})
                    .get("fdpEventTracking", {})
                    .get("events", {})
                )
                ppd = events.get("psi", {}).get("ppd", {})
                if "finalPrice" in ppd:
                    price = str(ppd["finalPrice"])

        # Check JSON-LD fallback in DOM
        if not price:
            for s_tag in response.xpath('//script[@type="application/ld+json"]/text()').getall():
                try:
                    sd = json.loads(s_tag)
                    if isinstance(sd, list):
                        sd = sd[0] if sd else {}
                    if isinstance(sd, dict) and "offers" in sd:
                        p_val = sd["offers"].get("price")
                        if p_val:
                            price = str(p_val)
                            break
                except Exception:
                    pass

        # Regex fallback on raw HTML for price
        if not price:
            match = re.search(r"₹\s*([0-9,]+)", response.text)
            if match:
                price = match.group(1).replace(",", "")

        if price:
            formatted_price = f"₹{price}" if not str(price).startswith("₹") else price
            yield {
                "product_price": formatted_price,
                "requested_url": self.start_url,
            }

        # Check if page has an explicit link to product reviews
        direct_rev_link = response.xpath(
            '//a[contains(@href, "/product-reviews/")]/@href'
        ).get()
        target_base_url = response.urljoin(direct_rev_link) if direct_rev_link else response.url

        # Navigate to the reviews page
        if self.all_sorts:
            for s_order in self.SORT_ORDERS:
                review_url = self._build_review_url(target_base_url, 1, sort_order=s_order)
                yield scrapy.Request(
                    url=review_url,
                    callback=self.parse_reviews,
                    meta={"current_page": 1, "sort_order": s_order},
                    dont_filter=True,
                )
        else:
            current_page = response.meta.get("current_page", self.start_page)
            review_url = self._build_review_url(
                target_base_url, current_page, sort_order=self.sort
            )
            yield scrapy.Request(
                url=review_url,
                callback=self.parse_reviews,
                meta={"current_page": current_page, "sort_order": self.sort},
                dont_filter=True,
            )

    def parse_reviews(self, response):
        """Parses Flipkart product reviews page and handles pagination if requested."""
        state = self._extract_initial_state(response)
        try:
            current_page = response.meta.get("current_page", self.start_page)
            sort_order = response.meta.get("sort_order", self.sort)
        except AttributeError:
            current_page = self.start_page
            sort_order = self.sort
        has_more = False
        reviews_found_on_page = 0
        new_review_components_on_page = 0

        sort_key = sort_order or "default"
        if sort_key not in self.seen_review_ids_per_sort:
            self.seen_review_ids_per_sort[sort_key] = set()

        if state:
            mws = state.get("multiWidgetState", {})
            slots = mws.get("widgetsData", {}).get("slots", [])
            if not slots:
                slots = mws.get("pageLevelSlots", [])

            pdr = mws.get("pageDataResponse", {})
            has_more = pdr.get("hasMorePages", False)

            # Extract Review Summary (yielded once per crawl)
            if not self.summary_yielded:
                seo = pdr.get("seoData", {}).get("seo", {})
                review_title = (
                    seo.get("keywords")
                    or seo.get("ogTitle")
                    or seo.get("title")
                    or response.xpath("//title/text()").get()
                    or ""
                )
                if ":" in review_title:
                    colon_parts = review_title.split(":", 1)
                    if colon_parts[0].strip().lower() in ("product reviews", "reviews", "flipkart"):
                        review_title = colon_parts[1].strip()
                    else:
                        review_title = colon_parts[0].strip()

                review_title = self._clean_product_title(review_title, response.url or self.start_url)
                summary = {"review_title": review_title}
                rating_found = False

                for slot in slots:
                    sd = slot.get("slotData", {})
                    widget = sd.get("widget", {})
                    if widget.get("type") == "RATING":
                        r_val = widget.get("data", {}).get("rating", {}).get("value", {})
                        avg_rating = r_val.get("average")
                        rating_count = r_val.get("ratingCount")
                        review_count = r_val.get("reviewCount")
                        histogram = r_val.get("histogram", [])

                        summary["avg_rating"] = str(avg_rating) if avg_rating is not None else ""
                        summary["total_ratings"] = (
                            f"{rating_count:,} Ratings" if rating_count is not None else ""
                        )
                        summary["total_reviews"] = (
                            f"{review_count:,} Reviews" if review_count is not None else ""
                        )

                        total_rev = review_count or 0
                        self.total_pages = math.ceil(total_rev / 10) if total_rev else 1
                        summary["total_pages"] = f"Page {current_page} of {self.total_pages:,}"

                        # BUG NOTE A6: Flipkart histogram assumed to be [1-star, 2-star,
                        # 3-star, 4-star, 5-star] order. If Flipkart changes this order,
                        # the star distribution will be inverted.
                        # Verified against existing test fixture: histogram[4]=5-star.
                        if len(histogram) == 5:
                            summary["5"] = f"{histogram[4]:,}"
                            summary["4"] = f"{histogram[3]:,}"
                            summary["3"] = f"{histogram[2]:,}"
                            summary["2"] = f"{histogram[1]:,}"
                            summary["1"] = f"{histogram[0]:,}"
                        rating_found = True
                        break

                if rating_found or summary.get("review_title"):
                    yield summary
                    self.summary_yielded = True

            # Extract Individual Reviews
            for slot in slots:
                sd = slot.get("slotData", {})
                widget = sd.get("widget", {})
                if widget.get("type") == "REVIEWS":
                    components = widget.get("data", {}).get("renderableComponents", [])
                    for comp in components:
                        val = comp.get("value", {})
                        if not val:
                            continue

                        # BUG FIX B3: Add review_id field
                        permalink = (val.get("url") or "").strip()
                        review_id = self._make_review_id(val, permalink)
                        self.seen_page_review_ids.add(review_id)
                        if review_id not in self.seen_review_ids_per_sort[sort_key]:
                            self.seen_review_ids_per_sort[sort_key].add(review_id)
                            new_review_components_on_page += 1

                        author = (val.get("author") or "").strip()
                        title = (val.get("title") or "").strip()
                        text = (val.get("text") or "").strip()

                        # BUG FIX B7: Location formatted without leading ", "
                        loc_data = val.get("location") or {}
                        loc_str = self._format_location(loc_data)

                        up_val = val.get("upvote", {})
                        down_val = val.get("downvote", {})
                        if isinstance(up_val, dict):
                            up = up_val.get("value", {}).get("count")
                        else:
                            up = None
                        if isinstance(down_val, dict):
                            down = down_val.get("value", {}).get("count")
                        else:
                            down = None

                        is_certified = (
                            self._is_certified_buyer(val.get("certifiedBuyer")) or
                            self._is_certified_buyer(val.get("authorBadge")) or
                            self._is_certified_buyer(val)
                        )
                        if self.verified_only and not is_certified:
                            continue

                        # Deduplicate only reviews that are eligible for output.
                        # An unverified duplicate must not block a verified record.
                        if review_id in self.seen_reviews:
                            continue
                        self.seen_reviews.add(review_id)

                        reviews_found_on_page += 1

                        yield {
                            "review_id": review_id,
                            "review_score": str(val.get("rating", "")),
                            "review_title": title,
                            "review_content": text,
                            "reviewer_name": author,
                            "reviewer_location": loc_str,
                            "reviewer_date": (val.get("created") or "").strip(),
                            "permalink": permalink,
                            "certified_buyer": is_certified,
                            "upvote": str(up) if up is not None else "0",
                            "downvote": str(down) if down is not None else "0",
                        }
        else:
            # Fallback DOM Parser
            self.logger.info(
                "__INITIAL_STATE__ not found on page %d. Falling back to DOM parsing. "
                "Note: DOM CSS selectors may be outdated if Flipkart changed its class names.",
                current_page,
            )
            if not self.summary_yielded:
                page_title = response.xpath("//title/text()").get() or ""
                if ":" in page_title:
                    colon_parts = page_title.split(":", 1)
                    if colon_parts[0].strip().lower() in ("product reviews", "reviews", "flipkart"):
                        page_title = colon_parts[1].strip()
                    else:
                        page_title = colon_parts[0].strip()
                page_title = self._clean_product_title(page_title, response.url or self.start_url)
                summary = {"review_title": page_title}
                match = re.search(
                    r"([0-9,]+)\s+ratings\s+and\s+([0-9,]+)\s+reviews",
                    response.text,
                    re.IGNORECASE,
                )
                if match:
                    summary["total_ratings"] = re.sub(r"[\s&]+$", "", f"{match.group(1)} Ratings").strip()
                    summary["total_reviews"] = re.sub(r"^[\s&]+", "", f"{match.group(2)} Reviews").strip()
                yield summary
                self.summary_yielded = True

            # Extract review cards from HTML DOM
            # NOTE: Flipkart's CSS class names change frequently.
            # These selectors may need updating if classes change.
            # Last verified: November 2024
            dom_reviews = response.xpath(
                '//div[contains(@class, "col") and '
                './/div[contains(@class, "X3daEc") or contains(@class, "hGK4qE") '
                'or contains(@class, "_3LWZlK") or contains(@class, "EKF0fL")]]'
            )
            if not dom_reviews:
                dom_reviews = response.css(
                    "div._27M-vq, div._1AtVbE div.cPHDOP, div.col._2wzgFH"
                )

            for card in dom_reviews:
                score = card.xpath(
                    './/div[contains(@class, "X3daEc") or contains(@class, "_3LWZlK") '
                    'or contains(@class, "hGK4qE")]/text()'
                ).get()
                if not score:
                    continue
                score = score.strip()
                title = (
                    card.xpath(
                        './/p[contains(@class, "z9E0IG") or contains(@class, "_2-N8zT")]/text()'
                    ).get()
                    or ""
                )
                content = " ".join(
                    card.xpath(
                        './/div[contains(@class, "ZmyHeo") or contains(@class, "t-ZTKy")]//text()'
                    ).getall()
                ).strip()
                author = (
                    card.xpath(
                        './/p[contains(@class, "_2NsDsF") or contains(@class, "_2sc7ZR")]/text()'
                    ).get()
                    or ""
                )
                loc_text = (
                    card.xpath(
                        './/p[contains(@class, "_2mcClG") or contains(@class, "_2V5EHH")]/text()'
                    ).get()
                    or ""
                )
                date_text = (
                    card.xpath(
                        './/p[contains(@class, "_2NsDsF") or contains(@class, "_2sc7ZR")][last()]/text()'
                    ).get()
                    or ""
                )

                is_certified = bool(
                    card.xpath(
                        './/*[contains(translate(normalize-space(text()), '
                        '"CERTIFIED BUYER", "certified buyer"), "certified buyer")]'
                    ).get()
                )
                if self.verified_only and not is_certified:
                    continue

                # BUG FIX B7: No leading ", " prefix in location
                loc_clean = loc_text.strip()

                rid = f"{author.strip()}|{title.strip()}|{content[:50]}"
                self.seen_page_review_ids.add(rid)
                if rid not in self.seen_review_ids_per_sort[sort_key]:
                    self.seen_review_ids_per_sort[sort_key].add(rid)
                    new_review_components_on_page += 1
                if rid in self.seen_reviews:
                    continue
                self.seen_reviews.add(rid)
                reviews_found_on_page += 1

                yield {
                    "review_id": rid,
                    "review_score": score,
                    "review_title": title.strip(),
                    "review_content": content,
                    "reviewer_name": author.strip(),
                    "reviewer_location": loc_clean,
                    "reviewer_date": date_text.strip(),
                    "permalink": "",
                    "certified_buyer": is_certified,
                    "upvote": "0",
                    "downvote": "0",
                }

        # Pagination check
        if self.scrape_all:
            can_continue = (
                self.target_max_pages is None or current_page < self.target_max_pages
            )
            # Flipkart caps each sort order at page 10.
            # Continue pagination as long as new reviews were discovered in THIS sort order
            # and Flipkart indicates more pages exist (has_more or current_page < total_pages).
            if can_continue and current_page < 10 and new_review_components_on_page > 0 and (
                has_more
                or current_page < self.total_pages
            ):
                next_page = current_page + 1
                next_url = self._build_review_url(
                    response.url, next_page, sort_order=sort_order
                )
                self.logger.info(
                    "Paginating to page %s (sort=%s): %s",
                    next_page,
                    sort_order or "default",
                    next_url,
                )
                yield scrapy.Request(
                    url=next_url,
                    callback=self.parse_reviews,
                    meta={"current_page": next_page, "sort_order": sort_order},
                    dont_filter=True,
                )
