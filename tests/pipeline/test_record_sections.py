"""Source page links and media IDs retain their original order."""

import unittest
from unittest.mock import Mock

from app.pipeline.records import merge_sections, page_sections


class SourceSectionTests(unittest.TestCase):
    def test_page_sections_deduplicates_media_without_removing_links(self):
        html = """
        <div class="main">
          <section>
            <h2 class="module-title">Related media</h2>
            <a href="/resources/media/abc/">First</a>
            <a href="https://www.astro4edu.org/resources/diagram/def/">Diagram</a>
            <a href="/resources/media/abc/">First again</a>
            <a href="/other/">Other</a>
          </section>
        </div>
        """
        media = [
            {"id": "media:abc", "record_url": "https://www.astro4edu.org/resources/media/abc/"},
            {"id": "diagram:def", "record_url": "https://www.astro4edu.org/resources/diagram/def/"},
        ]

        sections = page_sections(html, media)

        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0]["type"], "media_collection")
        self.assertEqual([link["text"] for link in sections[0]["links"]], ["First", "Diagram", "First again"])
        self.assertEqual(sections[0]["media_ids"], ["media:abc", "diagram:def"])

    def test_unchanged_section_keeps_translation_and_link_text(self):
        link = {"url": "https://astro4edu.org/a", "text": "Read", "text_ru": "Читать"}
        previous = {
            "sections": [{
                "id": "definition", "heading": "Glossary term", "body": "A star",
                "heading_ru": "Термин", "body_ru": "Звезда", "links": [link],
            }],
        }
        sections = [{
            "id": "definition", "heading": "Glossary term", "body": "A star",
            "links": [{"url": link["url"], "text": link["text"]}],
        }]
        keep_stale = Mock()

        changed = merge_sections(sections, previous, keep_stale)

        self.assertFalse(changed)
        self.assertEqual(sections[0]["heading_ru"], "Термин")
        self.assertEqual(sections[0]["body_ru"], "Звезда")
        self.assertEqual(sections[0]["links"][0]["text_ru"], "Читать")
        keep_stale.assert_not_called()
