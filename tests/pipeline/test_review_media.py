"""Review media links retain source order and attach each image once."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from app.pipeline import review_media


class ReviewMediaTests(unittest.TestCase):
    def test_discovery_preserves_order_and_deduplicates_links(self):
        media_url = "https://astro4edu.org/resources/media/ABC/"
        diagram_url = "https://astro4edu.org/resources/diagram/DEF/"
        sections = [
            {"links": [
                {"url": media_url},
                {"url": "https://astro4edu.org/oae-reviews/"},
                {"url": diagram_url},
                {"url": media_url},
            ], "media_ids": []},
            {"links": [{"url": diagram_url}, {"url": media_url}], "media_ids": []},
        ]

        def lookup(_data: Path, path: str, _glossary: dict) -> dict:
            return {"id": path}

        with (
            patch.object(review_media, "media_record", side_effect=lookup) as read,
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            media = review_media.review_media(Path("."), sections, pool, {})

        expected = ["/resources/media/ABC", "/resources/diagram/DEF"]
        self.assertEqual([item["id"] for item in media], expected)
        self.assertEqual(sections[0]["media_ids"], expected)
        self.assertEqual(sections[1]["media_ids"], expected[::-1])
        self.assertEqual(read.call_count, 2)

    def test_no_resource_links_leave_existing_media_untouched(self):
        sections = [{"links": [{"url": "https://astro4edu.org/oae-reviews/"}],
                     "media_ids": ["existing"]}]
        with ThreadPoolExecutor(max_workers=1) as pool:
            media = review_media.review_media(Path("."), sections, pool, {})
        self.assertEqual(media, [])
        self.assertEqual(sections[0]["media_ids"], ["existing"])
