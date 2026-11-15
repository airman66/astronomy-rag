import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import assistant, storage, vector
from app.main import app


class FakeLimiter:
    retry = 0

    def check(self, request, category):
        return self.retry

    def acquire_call(self, token):
        return True

    def release_call(self, token):
        pass


class FakeDeepSeek:
    async def complete(self, system, payload):
        if "classify" in system:
            scope = "out_of_scope" if "борщ" in payload["question"].lower() else "in_scope"
            return {"scope": scope, "mode": "term", "record_ids": ["term:1"], "reason": "definition"}
        return {"answer": "Звезда класса A — горячая звезда.", "used_record_ids": ["term:1"]}


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "processed/terms").mkdir(parents=True)
        (self.root / "processed/articles").mkdir(parents=True)
        (self.root / "media/original").mkdir(parents=True)
        self.image = self.root / "media/original/example.png"
        self.image.write_bytes(b"not-a-real-png")
        record = {
            "id": "term:1", "kind": "term_page",
            "source": {"type": "astro4edu_glossary", "term_number": 1, "url": "https://astro4edu.org/resources/glossary/term/1/", "fetched_at": "2026-11-14T00:00:00Z"},
            "metadata": {"categories": ["Stars"], "related_term_numbers": []},
            "sections": [{"id": "definition", "type": "definition", "heading": "A-type Star", "body": "A hot star.", "media_ids": ["media:example"], "links": []}],
            "locales": {
                "en": {"title": "A-type Star", "aliases": [], "blocks": {"definition": {"heading": "Definition", "body": "A hot star."}}},
                "ru": {"title": "Звезда класса A", "aliases": [], "blocks": {"definition": {"heading": "Определение", "body": "Горячая звезда."}}},
            },
            "media": [{"id": "media:example", "kind": "media", "file_type": "img", "original_path": str(self.image), "title_en": "Example", "title_ru": "Пример", "caption_en": "A star.", "caption_ru": "Звезда.", "alt_en": "Star", "alt_ru": "Звезда", "review": {"status": "draft", "reviewer": None}}],
            "review": {"status": "draft", "reviewer": None},
        }
        (self.root / "processed/terms/1.json").write_text(json.dumps(record, ensure_ascii=False))
        self.previous = storage.DATA, storage.DB, vector.DATA, vector._loaded, app.state.rate_limiter, app.state.deepseek
        storage.DATA = vector.DATA = self.root
        storage.DB = self.root / "glossary.db"
        vector._loaded = None
        storage.rebuild_database()
        app.state.rate_limiter = FakeLimiter()
        app.state.deepseek = FakeDeepSeek()
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        storage.DATA, storage.DB, vector.DATA, vector._loaded, app.state.rate_limiter, app.state.deepseek = self.previous
        self.temp.cleanup()

    def test_read_routes_and_media(self):
        self.assertEqual(self.client.get("/api/v1/terms").json()["total"], 1)
        term = self.client.get("/api/v1/terms/1")
        self.assertEqual(term.status_code, 200)
        self.assertEqual(term.json()["locales"]["ru"]["title"], "Звезда класса A")
        self.assertEqual(self.client.get("/api/v1/articles").json()["total"], 0)
        self.assertEqual(self.client.get("/api/v1/media/media:example").json()["caption_ru"], "Звезда.")
        self.assertEqual(self.client.get("/api/v1/media/media:example/file").content, b"not-a-real-png")
        self.assertEqual(self.client.get("/api/v1/terms/999").json()["code"], "term_not_found")

    def test_bilingual_search_and_openapi(self):
        self.assertEqual(self.client.get("/api/v1/search", params={"q": "Звезда", "locale": "ru"}).json()["items"][0]["record_id"], "term:1")
        self.assertEqual(self.client.get("/api/v1/search", params={"q": "star", "locale": "en"}).json()["items"][0]["record_id"], "term:1")
        schema = self.client.get("/openapi.json").json()
        self.assertIn("/api/v1/assistant/answer", schema["paths"])
        self.assertIn("AnswerResponse", schema["components"]["schemas"])

    def test_assistant_modes_and_errors(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "unit-test"}):
            response = self.client.post("/api/v1/assistant/answer", json={"question": "Что такое звезда?"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["mode"], "term")
            self.assertEqual(response.json()["sources"][0]["record_id"], "term:1")
            self.assertEqual(response.json()["images"][0]["url"], "/api/v1/media/media:example/file?locale=ru")
            general = self.client.post("/api/v1/assistant/answer", json={"question": "Почему у планет есть орбиты?", "mode": "general"})
            self.assertEqual(general.json()["grounding"], "model")
            self.assertEqual(general.json()["sources"], [])
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            self.assertEqual(self.client.post("/api/v1/assistant/answer", json={"question": "Что такое звезда?"}).json()["code"], "assistant_unavailable")

    def test_assistant_uses_definition_even_when_search_ranks_another_block(self):
        unrelated = {"record_id": "term:1", "section_id": "section_4", "kind": "term", "title": "Звезда класса A", "excerpt": "Не определение", "url": "/api/v1/terms/1", "review_status": "draft", "score": 1.0}
        with patch.object(storage, "search", return_value=[unrelated]):
            candidate = assistant.candidates_for("Что такое звезда?", "ru")[0]
        self.assertEqual(candidate["section_id"], "definition")
        self.assertEqual(candidate["excerpt"], "Горячая звезда.")

    def test_assistant_can_show_record_image_when_definition_has_no_media(self):
        record = storage.get_term(1)
        record["sections"][0]["media_ids"] = []
        image = assistant.image_for(record, record["sections"][0], "ru")
        self.assertEqual(image["url"], "/api/v1/media/media:example/file?locale=ru")

    def test_assistant_does_not_attach_unused_sources(self):
        class EmptyCitations:
            async def complete(self, system, payload):
                if "classify" in system:
                    return {"scope": "in_scope", "mode": "term", "record_ids": ["term:1"]}
                if "general model knowledge" in system:
                    return {"answer": "Ответ без ссылки.", "used_record_ids": []}
                return {"answer": "Источник не использован.", "used_record_ids": []}

        app.state.deepseek = EmptyCitations()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "unit-test"}):
            response = self.client.post("/api/v1/assistant/answer", json={"question": "Что такое звезда?", "mode": "term"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["mode"], "general")
        self.assertEqual(response.json()["grounding"], "model")
        self.assertEqual(response.json()["sources"], [])

    def test_assistant_rejects_off_topic_in_every_mode(self):
        class ScopeOnly:
            async def complete(self, system, payload):
                if "classify" not in system:
                    raise AssertionError("Off-topic question must not reach answer generation")
                return {"scope": "out_of_scope", "mode": "general", "record_ids": []}

        app.state.deepseek = ScopeOnly()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "unit-test"}):
            for mode in ("auto", "term", "explain", "general"):
                with self.subTest(mode=mode):
                    response = self.client.post("/api/v1/assistant/answer", json={
                        "question": "Как приготовить борщ?", "mode": mode,
                        "context": {"planet": "Proxima Cen b"},
                    })
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json()["mode"], "out_of_scope")
                    self.assertEqual(response.json()["grounding"], "policy")
                    self.assertIn("астрономии", response.json()["answer"])
                    self.assertEqual(response.json()["sources"], [])

    def test_rate_limit_shape(self):
        app.state.rate_limiter.retry = 7
        response = self.client.post("/api/v1/assistant/answer", json={"question": "Что такое звезда?"})
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["Retry-After"], "7")
        self.assertEqual(response.json()["code"], "rate_limited")


if __name__ == "__main__":
    unittest.main()
