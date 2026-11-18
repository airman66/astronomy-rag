import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.pipeline import importer


class ImporterRefreshTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "data"
        self.previous = importer.DATA, importer.REFRESH
        importer.DATA = self.root
        importer.REFRESH = False

    def tearDown(self):
        importer.DATA, importer.REFRESH = self.previous
        self.temp.cleanup()

    def test_refresh_preserves_raw_snapshots_and_reads_current_version(self):
        path = self.root / "raw/terms/1.json"
        first = b'{"results": [1]}'
        second = b'{"results": [2]}'
        with patch.object(importer, "fetch", side_effect=[first, second, first]) as fetch:
            self.assertEqual(importer.raw_json(path, "https://example.test/1")["results"], [1])
            importer.REFRESH = True
            self.assertEqual(importer.raw_json(path, "https://example.test/1")["results"], [2])
            self.assertEqual(fetch.call_count, 2)
            self.assertNotEqual(importer.current_snapshot(path), path)
            self.assertEqual(importer.raw_json(path, "https://example.test/1")["results"], [1])
        self.assertEqual(path.read_bytes(), first)
        revision = importer.current_snapshot(path)
        self.assertEqual(revision, path)
        self.assertTrue(path.with_name(f"1.{hashlib.sha256(second).hexdigest()}.json").exists())
        importer.REFRESH = False
        self.assertEqual(importer.raw_json(path, "https://example.test/1")["results"], [1])

    def test_invalid_refreshed_json_does_not_replace_current_snapshot(self):
        path = self.root / "raw/terms/1.json"
        importer.save_snapshot(path, b'{"results": [1]}')
        importer.REFRESH = True
        with patch.object(importer, "fetch", return_value=b"not json"), self.assertRaises(json.JSONDecodeError):
            importer.raw_json(path, "https://example.test/1")
        self.assertEqual(importer.current_snapshot(path).read_bytes(), b'{"results": [1]}')

    def test_removed_section_and_media_reset_review_without_losing_translation(self):
        raw_term = self.root / "raw/terms/1.json"
        raw_page = self.root / "raw/pages/1.html"
        raw_media = self.root / "raw/multimedia/1.json"
        raw_diagrams = self.root / "raw/diagrams/1.json"
        for path in (raw_term, raw_page, raw_media, raw_diagrams):
            path.parent.mkdir(parents=True, exist_ok=True)
        raw_term.write_text(json.dumps({"results": [{"term_number": 1, "term_name": "Star", "categories": [], "linked_terms": []}]}))
        raw_page.write_text('<div class="main"><section><h2 class="module-title">Glossary term</h2><p>Description: A star.</p></section></div>')
        raw_media.write_text('{"results": []}')
        raw_diagrams.write_text('{"results": []}')
        processed = self.root / "processed/terms/1.json"
        processed.parent.mkdir(parents=True)
        prior = {
            "id": "term:1",
            "source": {"term_number": 1, "url": "https://astro4edu.org/resources/glossary/term/1/", "fetched_at": "2026-11-14T00:00:00Z"},
            "metadata": {"categories": []},
            "sections": [
                {"id": "definition", "type": "definition", "heading": "Glossary term", "body": "A star.", "media_ids": [], "links": []},
                {"id": "details", "type": "source_block", "heading": "Details", "heading_ru": "Подробности", "body": "Old text", "body_ru": "Старый текст", "media_ids": [], "links": []},
            ],
            "locales": {
                "en": {"title": "Star", "aliases": [], "blocks": {"definition": {"heading": "Glossary term", "body": "A star."}, "details": {"heading": "Details", "body": "Old text"}}},
                "ru": {"title": "Звезда", "aliases": [], "blocks": {"definition": {"heading": "Термин", "body": "Звезда."}, "details": {"heading": "Подробности", "body": "Старый текст"}}},
            },
            "media": [{"id": "media:old", "title_en": "Old image", "title_ru": "Старое изображение", "caption_en": "Old caption", "caption_ru": "Старая подпись"}],
            "review": {"status": "reviewed", "reviewer": "editor"},
        }
        processed.write_text(json.dumps(prior, ensure_ascii=False))
        importer.import_term(1)
        result = json.loads(processed.read_text())
        self.assertEqual(result["review"]["status"], "draft")
        self.assertEqual(result["source"]["fetched_at"], "2026-11-14T00:00:00Z")
        self.assertEqual([section["id"] for section in result["sections"]], ["definition"])
        self.assertEqual(result["media"], [])
        stale = [entry["previous_ru"] for entry in result["stale_translations"]]
        self.assertIn("Старый текст", stale)
        self.assertIn("Старая подпись", stale)

    def test_changed_image_gets_new_file_and_invalidates_russian_version(self):
        processed = self.root / "processed/terms/1.json"
        processed.parent.mkdir(parents=True)
        original = self.root / "media/original/media_example.png"
        russian = self.root / "media/ru/media_example.png"
        original.parent.mkdir(parents=True)
        russian.parent.mkdir(parents=True)
        original.write_bytes(b"old original")
        russian.write_bytes(b"old russian")
        record = {
            "id": "term:1",
            "source": {"term_number": 1},
            "media": [{"id": "media:example", "source_url": "https://example.test/image.png", "file_type": "img", "original_path": str(original), "sha256": hashlib.sha256(original.read_bytes()).hexdigest(), "ru_path": str(russian), "ru_sha256": hashlib.sha256(russian.read_bytes()).hexdigest(), "review": {"status": "reviewed", "reviewer": "editor"}}],
            "review": {"status": "reviewed", "reviewer": "editor"},
        }
        processed.write_text(json.dumps(record))
        second = copy.deepcopy(record)
        second["id"] = "term:2"
        second["source"]["term_number"] = 2
        (processed.parent / "2.json").write_text(json.dumps(second))
        with patch.object(importer, "fetch", return_value=b"new original"):
            downloaded, failed = importer.download_media(workers=1, refresh=True, preferred_ids={1})
        self.assertEqual((downloaded, failed), (1, 0))
        updated = json.loads(processed.read_text())
        item = updated["media"][0]
        self.assertEqual(original.read_bytes(), b"old original")
        self.assertEqual(Path(item["original_path"]).read_bytes(), b"new original")
        self.assertIsNone(item["ru_path"])
        self.assertIsNone(item["ru_sha256"])
        self.assertEqual(item["review"]["status"], "draft")
        self.assertEqual(updated["review"]["status"], "draft")
        shared = json.loads((processed.parent / "2.json").read_text())
        self.assertEqual(shared["media"][0]["original_path"], item["original_path"])
        self.assertIsNone(shared["media"][0]["ru_path"])
        self.assertEqual(shared["review"]["status"], "draft")


if __name__ == "__main__":
    unittest.main()
