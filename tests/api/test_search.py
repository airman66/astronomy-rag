from unittest.mock import patch

from app.catalog import vector
from tests.support.api import ApiCase
from tests.support.catalog import seed_search_terms, seed_star


class BilingualSearchTests(ApiCase):
    def setUp(self) -> None:
        super().setUp()
        seed_star(self.root)

    def test_bilingual_search_and_openapi(self):
        self.assertEqual(
            self.client.get("/api/v1/search", params={"q": "Звезда", "locale": "ru"}).json()["items"][0][
                "record_id"
            ],
            "term:1",
        )
        self.assertEqual(
            self.client.get("/api/v1/search", params={"q": "star", "locale": "en"}).json()["items"][0][
                "record_id"
            ],
            "term:1",
        )
        schema = self.client.get("/openapi.json").json()
        self.assertIn("/api/v1/assistant/answer", schema["paths"])
        self.assertIn("AnswerResponse", schema["components"]["schemas"])


class SearchFallbackTests(ApiCase):
    def setUp(self) -> None:
        super().setUp()
        seed_search_terms(self.root)

    def test_search_falls_back_to_lexical_when_embeddings_fail(self):
        with patch.object(vector, "search", side_effect=ConnectionError("embedding server down")):
            response = self.client.get("/api/v1/search", params={"q": "дыра", "locale": "ru"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["mode"], "lexical")
        self.assertEqual(response.json()["items"][0]["record_id"], "term:1")

    def test_russian_search_ignores_case_yo_and_word_forms(self):
        self.assertEqual(self.client.get("/api/v1/terms", params={"q": "звезды"}).json()["total"], 1)
        self.assertEqual(self.client.get("/api/v1/terms", params={"q": "ЧЕРНАЯ"}).json()["total"], 1)
        found = self.client.get("/api/v1/search", params={"q": "черные дыры", "locale": "ru"}).json()
        self.assertEqual(found["items"][0]["record_id"], "term:1")
        star = self.client.get("/api/v1/search", params={"q": "что такое звезда", "locale": "ru"}).json()
        self.assertEqual(star["items"][0]["record_id"], "term:2")
        self.assertLess(star["total"], 3, "служебные слова не должны находить весь корпус")

    def test_hybrid_pagination_reuses_pool_until_index_changes(self):
        index_root = self.root / "index"
        index_root.mkdir()
        manifest = index_root / "index-manifest.json"
        manifest.write_text("{}")
        with (
            patch.object(vector, "available", return_value=True),
            patch.object(vector, "search", return_value=[]) as embed,
        ):
            self.client.get("/api/v1/search", params={"q": "black hole", "locale": "en", "page": 1})
            self.client.get("/api/v1/search", params={"q": "black  hole", "locale": "en", "page": 2})
            self.assertEqual(embed.call_count, 1)
            manifest.write_text('{"updated":true}')
            self.client.get("/api/v1/search", params={"q": "black hole", "locale": "en", "page": 2})
            self.assertEqual(embed.call_count, 2)
