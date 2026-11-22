"""Wikimedia file discovery preserves the existing request contract."""

import unittest
from unittest.mock import patch

from scripts.covers import sources


class WikimediaSourceTests(unittest.TestCase):
    def test_file_search_uses_one_commons_query(self):
        pages = [{"title": "File:Star.jpg"}]
        with patch.object(sources, "api", return_value={"query": {"pages": pages}}) as request:
            self.assertEqual(sources.files("Star"), pages)

        request.assert_called_once_with(
            "commons.wikimedia.org",
            generator="search",
            gsrsearch='intitle:"Star"',
            gsrnamespace=6,
            gsrlimit=20,
            prop="imageinfo",
            iiprop="size",
            redirects=1,
        )

    def test_file_names_reject_logos_but_accept_photographs(self):
        self.assertTrue(sources.file_ok("Starfield.JPG"))
        self.assertFalse(sources.file_ok("Star Logo.JPG"))
        self.assertFalse(sources.file_ok("Star.txt"))
