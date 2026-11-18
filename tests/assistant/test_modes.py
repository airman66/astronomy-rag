from unittest.mock import patch

from app.assistant import service as assistant
from tests.support.api import ANSWER_PATH, ApiCase
from tests.support.catalog import seed_star


class AssistantModeTests(ApiCase):
    def setUp(self) -> None:
        super().setUp()
        seed_star(self.root)

    def test_auto_mode_returns_term_and_image(self):
        response = self.client.post(ANSWER_PATH, json={"question": "Что такое звезда?"})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["mode"], "term")
        self.assertEqual(result["sources"][0]["record_id"], "term:1")
        self.assertEqual(result["images"][0]["url"], "/api/v1/media/media:example/file?locale=ru")

    def test_general_mode_uses_model_without_sources(self):
        response = self.client.post(
            ANSWER_PATH, json={"question": "Почему у планет есть орбиты?", "mode": "general"}
        )
        result = response.json()
        self.assertEqual(result["grounding"], "model")
        self.assertEqual(result["sources"], [])

    def test_general_mode_skips_retrieval(self):
        class General:
            async def complete(self, system, payload):
                if "classify" in system:
                    return {"scope": "in_scope", "mode": "general", "record_ids": []}
                return {"answer": "Ответ.", "used_record_ids": []}

        self.app.state.llm_client = General()
        with patch.object(assistant, "candidates_for", side_effect=AssertionError("retrieval must not run")):
            response = self.client.post(
                ANSWER_PATH, json={"question": "Почему звёзды светят?", "mode": "general"}
            )
        self.assertEqual(response.status_code, 200, response.text)

    def test_assistant_rejects_off_topic_in_every_mode(self):
        class ScopeOnly:
            async def complete(self, system, payload):
                if "classify" not in system:
                    raise AssertionError("Off-topic question must not reach answer generation")
                return {"scope": "out_of_scope", "mode": "general", "record_ids": []}

        self.app.state.llm_client = ScopeOnly()
        for mode in ("auto", "term", "explain", "general"):
            with self.subTest(mode=mode):
                response = self.client.post(
                    ANSWER_PATH,
                    json={
                        "question": "Как приготовить борщ?",
                        "mode": mode,
                        "context": {"planet": "Proxima Cen b"},
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertEqual(result["mode"], "out_of_scope")
                self.assertEqual(result["grounding"], "policy")
                self.assertIn("астрономии", result["answer"])
                self.assertEqual(result["sources"], [])
