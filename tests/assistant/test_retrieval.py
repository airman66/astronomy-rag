import unittest
from unittest.mock import patch

from app.assistant import retrieval
from app.assistant import service as assistant
from app.catalog import storage
from tests.support.catalog import seed_star, use_catalog


class AssistantRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = use_catalog(self)
        seed_star(self.root)

    def test_assistant_uses_definition_even_when_search_ranks_another_block(self):
        unrelated = {
            "record_id": "term:1",
            "section_id": "section_4",
            "kind": "term",
            "title": "Звезда класса A",
            "excerpt": "Не определение",
            "url": "/api/v1/terms/1",
            "review_status": "draft",
            "score": 1.0,
        }
        with patch.object(retrieval, "search_with_mode", return_value=([unrelated], "lexical")):
            candidate = assistant.candidates_for("Что такое звезда?", "ru")[0]
        self.assertEqual(candidate["section_id"], "definition")
        self.assertEqual(candidate["excerpt"], "Горячая звезда.")

    def test_assistant_keeps_matched_section_for_explanations(self):
        record = storage.get_term(1)
        record["sections"].append(
            {
                "id": "spectra",
                "type": "source_block",
                "heading": "Spectra",
                "body": "Blue light.",
                "media_ids": [],
                "links": [],
            }
        )
        record["locales"]["ru"]["blocks"]["spectra"] = {
            "heading": "Спектр",
            "body": "Голубой свет определяется температурой.",
        }
        match = {
            "record_id": "term:1",
            "section_id": "spectra",
            "kind": "term",
            "title": "Звезда класса A",
            "excerpt": "Голубой свет",
            "url": "/api/v1/terms/1",
            "review_status": "draft",
            "score": 1.0,
        }
        with (
            patch.object(retrieval, "search_with_mode", return_value=([match], "lexical")),
            patch.object(storage, "get_term", return_value=record),
        ):
            candidate = assistant.candidates_for("Почему голубой свет?", "ru")[0]
        self.assertEqual(candidate["section_id"], "definition")
        self.assertEqual(candidate["matched_section_id"], "spectra")
        self.assertIn("температурой", assistant.matched_context(candidate)["excerpt"])

    def test_candidates_stop_at_eight_and_reuse_the_same_section_image(self):
        base = storage.get_term(1)
        hits = [
            {"record_id": f"term:{number}", "section_id": "definition", "title": "Star", "score": 1}
            for number in range(1, 81)
        ]
        with (
            patch.object(retrieval, "search_with_mode", return_value=(hits, "lexical")),
            patch.object(storage, "get_term", return_value=base) as get_term,
            patch.object(retrieval, "image_for", return_value={"url": "/illustration"}) as image_for,
        ):
            candidates = retrieval.candidates_for("Star", "en")
        self.assertEqual([item["record_id"] for item in candidates], [f"term:{n}" for n in range(1, 9)])
        self.assertEqual(get_term.call_count, 8)
        self.assertEqual(image_for.call_count, 8)
        self.assertEqual(candidates[0]["image"], candidates[0]["matched_image"])

    def test_assistant_can_show_record_image_when_definition_has_no_media(self):
        record = storage.get_term(1)
        record["sections"][0]["media_ids"] = []
        image = retrieval.image_for(record, record["sections"][0], "ru")
        self.assertEqual(image["url"], "/api/v1/media/media:example/file?locale=ru")
