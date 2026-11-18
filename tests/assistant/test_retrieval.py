import unittest
from unittest.mock import patch

from app.assistant import retrieval
from app.catalog import storage
from tests.support.catalog import seed_star, use_catalog


class AssistantRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = use_catalog(self)
        seed_star(self.root)

    def test_canonical_definition_survives_a_non_definition_hit(self):
        hit = {"record_id": "term:1", "section_id": "section_4"}
        with patch.object(retrieval, "search_with_mode", return_value=([hit], "lexical")):
            fragments = retrieval.evidence_for("Что такое звезда?", "ru")
        self.assertEqual(fragments["e1"].section["id"], "definition")
        self.assertEqual(fragments["e1"].canonical_definition, "Горячая звезда.")

    def test_retrieval_preserves_the_exact_chunk_and_ignores_activities(self):
        record = storage.get_term(1)
        assert record is not None
        for key, heading, body in (
            ("spectra", "Spectra", "Ранний абзац. Поздний найденный фрагмент."),
            ("activities", "Educational activities", "Учебные активности."),
        ):
            record["sections"].append({"id": key, "type": "source_block", "heading": heading})
            record["locales"]["ru"]["blocks"][key] = {"heading": heading, "body": body}
        hit = {"record_id": "term:1", "section_id": "spectra", "chunk_id": "late",
               "evidence_text": "Поздний найденный фрагмент."}
        activities = {"record_id": "term:1", "section_id": "activities"}
        with (
            patch.object(retrieval, "search_with_mode", return_value=([activities, hit], "hybrid")),
            patch.object(storage, "get_term", return_value=record),
        ):
            fragments = retrieval.evidence_for("Почему голубой свет?", "ru")
        self.assertEqual([item.section["id"] for item in fragments.values()], ["definition", "spectra"])
        self.assertEqual(fragments["e2"].text, hit["evidence_text"])
        self.assertEqual(fragments["e2"].chunk_id, "late")

    def test_assistant_can_show_record_image_when_definition_has_no_media(self):
        record = storage.get_term(1)
        assert record is not None
        record["sections"][0]["media_ids"] = []
        image = retrieval.image_for(record, record["sections"][0], "ru")
        assert image is not None
        self.assertEqual(image["url"], "/api/v1/media/media:example/file?locale=ru")
