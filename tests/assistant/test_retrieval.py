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

    def test_reduced_fragment_does_not_claim_the_full_chunk(self):
        record = storage.get_term(1)
        assert record is not None
        section = {"id": "spectra", "type": "source_block"}
        canonical = "Первый абзац.\n\n" + ("Второй абзац. " * 160).strip()
        record["sections"].append(section)
        record["locales"]["ru"]["blocks"]["spectra"] = {"body": canonical}
        hit = {"record_id": "term:1", "section_id": "spectra", "chunk_id": "long", "evidence_text": canonical}
        with (
            patch.object(retrieval, "search_with_mode", return_value=([hit], "hybrid")),
            patch.object(storage, "get_term", return_value=record),
        ):
            fragments = retrieval.evidence_for("Спектр", "ru")
        self.assertEqual(fragments["e2"].text, "Первый абзац.")
        self.assertIsNone(fragments["e2"].chunk_id)

    def test_caption_uses_its_media_without_falling_back_to_a_cover(self):
        record = storage.get_term(1)
        assert record is not None
        record["illustration"] = {"url": "/api/v1/terms/1/illustration"}
        section = {"id": "media:example", "type": "caption", "media_ids": ["media:example"]}
        with patch.object(storage, "get_illustration", side_effect=AssertionError("caption uses source media")):
            image = retrieval.image_for(record, section, "ru")
        assert image is not None
        self.assertEqual(image["url"], "/api/v1/media/media:example/file?locale=ru")
        section["media_ids"] = ["media:missing"]
        self.assertIsNone(retrieval.image_for(record, section, "ru"))

    def test_page_hint_provides_real_definition_without_lexical_matches(self):
        with patch.object(retrieval, "search_with_mode", return_value=([], "lexical")):
            fragments = retrieval.evidence_for("Объясни это", "ru", context_record_id="term:1")
        self.assertEqual(list(fragments), ["e1"])
        self.assertEqual(fragments["e1"].record["id"], "term:1")
        self.assertEqual(fragments["e1"].canonical_definition, "Горячая звезда.")

    def test_invalid_or_missing_page_hint_is_not_used_as_evidence(self):
        with patch.object(retrieval, "search_with_mode", return_value=([], "lexical")):
            for invalid in ("term:999999999999999999999", "article:../../secret", "term:xyz", "article:missing"):
                with self.subTest(hint=invalid):
                    self.assertEqual(retrieval.evidence_for("this", "en", context_record_id=invalid), {})
