import json

from app.pipeline import importer
from tests.pipeline.fixtures import ImporterCase
from tests.support.catalog import make_term, write_term


class RecordRefreshTests(ImporterCase):
    def test_removed_section_and_media_reset_review_without_losing_translation(self):
        raw_term = self.root / "raw/terms/1.json"
        raw_page = self.root / "raw/pages/1.html"
        raw_media = self.root / "raw/multimedia/1.json"
        raw_diagrams = self.root / "raw/diagrams/1.json"
        for path in (raw_term, raw_page, raw_media, raw_diagrams):
            path.parent.mkdir(parents=True, exist_ok=True)
        raw_term.write_text(
            json.dumps(
                {"results": [{"term_number": 1, "term_name": "Star", "categories": [], "linked_terms": []}]}
            )
        )
        raw_page.write_text(
            '<div class="main"><section><h2 class="module-title">Glossary term</h2><p>Description: A star.</p></section></div>'
        )
        raw_media.write_text('{"results": []}')
        raw_diagrams.write_text('{"results": []}')
        processed = self.root / "processed/terms/1.json"
        prior = make_term(1, "Star", "Звезда", "A star.", "Звезда.", categories=())
        prior["source"]["fetched_at"] = "2026-11-14T00:00:00Z"
        prior["review"] = {"status": "reviewed", "reviewer": "editor"}
        prior["sections"][0]["heading"] = "Glossary term"
        prior["sections"].append(
            {
                "id": "details",
                "type": "source_block",
                "heading": "Details",
                "heading_ru": "Подробности",
                "body": "Old text",
                "body_ru": "Старый текст",
                "media_ids": [],
                "links": [],
            }
        )
        prior["locales"]["en"]["blocks"]["definition"]["heading"] = "Glossary term"
        prior["locales"]["ru"]["blocks"]["definition"]["heading"] = "Термин"
        prior["locales"]["en"]["blocks"]["details"] = {"heading": "Details", "body": "Old text"}
        prior["locales"]["ru"]["blocks"]["details"] = {"heading": "Подробности", "body": "Старый текст"}
        prior["media"].append(
            {
                "id": "media:old",
                "title_en": "Old image",
                "title_ru": "Старое изображение",
                "caption_en": "Old caption",
                "caption_ru": "Старая подпись",
            }
        )
        write_term(self.root, prior)
        importer.import_term(1)
        result = json.loads(processed.read_text())
        self.assertEqual(result["review"]["status"], "draft")
        self.assertEqual(result["source"]["fetched_at"], "2026-11-14T00:00:00Z")
        self.assertEqual([section["id"] for section in result["sections"]], ["definition"])
        self.assertEqual(result["media"], [])
        stale = [entry["previous_ru"] for entry in result["stale_translations"]]
        self.assertIn("Старый текст", stale)
        self.assertIn("Старая подпись", stale)
