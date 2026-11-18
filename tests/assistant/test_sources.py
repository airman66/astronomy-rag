from unittest.mock import patch

from app.assistant import service as assistant
from tests.support.api import ANSWER_PATH, ApiCase
from tests.support.catalog import seed_star


class AssistantSourceTests(ApiCase):
    def setUp(self) -> None:
        super().setUp()
        seed_star(self.root)

    def test_article_mode_selects_article_below_term(self):
        term = {
            "record_id": "term:1",
            "section_id": "definition",
            "kind": "term",
            "title": "Звезда",
            "excerpt": "Определение",
            "url": "/api/v1/terms/1",
            "review_status": "draft",
            "score": 1,
            "image": None,
        }
        article = {
            "record_id": "article:stars",
            "section_id": "intro",
            "kind": "article",
            "title": "Звёзды",
            "excerpt": "Введение",
            "matched_section_id": "details",
            "matched_excerpt": "Подробный раздел",
            "url": "/api/v1/articles/article:stars",
            "review_status": "reviewed",
            "score": 0.9,
            "image": None,
        }

        class ArticleClient:
            async def complete(self, system, payload):
                if "classify" in system:
                    return {"scope": "in_scope", "mode": "term", "record_ids": ["term:1"]}
                self.sources = payload["sources"]
                return {"answer": "Подробный ответ.", "used_record_ids": ["article:stars"]}

        client = ArticleClient()
        self.app.state.llm_client = client
        with patch.object(assistant, "candidates_for", return_value=[term, article]):
            response = self.client.post(
                ANSWER_PATH, json={"question": "Расскажи подробно о звёздах", "mode": "article"}
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["mode"], "article")
        self.assertEqual(response.json()["sources"][0]["section_id"], "details")
        self.assertEqual(client.sources[0]["excerpt"], "Подробный раздел")

    def test_assistant_does_not_attach_unused_sources(self):
        class EmptyCitations:
            async def complete(self, system, payload):
                if "classify" in system:
                    return {"scope": "in_scope", "mode": "term", "record_ids": ["term:1"]}
                if "general model knowledge" in system:
                    return {"answer": "Ответ без ссылки.", "used_record_ids": []}
                return {"answer": "Источник не использован.", "used_record_ids": []}

        self.app.state.llm_client = EmptyCitations()
        response = self.client.post(ANSWER_PATH, json={"question": "Что такое звезда?", "mode": "term"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["mode"], "general")
        self.assertEqual(response.json()["grounding"], "model")
        self.assertEqual(response.json()["sources"], [])
