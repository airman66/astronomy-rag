import os
from unittest.mock import AsyncMock, Mock, patch

import redis

from app.assistant.client import AssistantNotConfigured, Completion, LLMClient
from tests.support.api import ANSWER_PATH, ApiCase
from tests.support.catalog import seed_search_terms


class AssistantErrorTests(ApiCase):
    def setUp(self) -> None:
        super().setUp()
        seed_search_terms(self.root)

    def test_missing_api_key_returns_unavailable(self):
        self.app.state.llm_client = LLMClient()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            response = self.client.post(ANSWER_PATH, json={"question": "Что такое звезда?"})
        self.assertEqual(response.json()["code"], "assistant_unavailable")

    def test_assistant_rejects_malformed_model_json(self):
        self.app.state.llm_client = Mock(complete=AsyncMock(return_value=Completion([], "test-model")))
        response = self.client.post(ANSWER_PATH, json={"question": "Что такое чёрная дыра?"})
        self.assertEqual(response.status_code, 502, response.text)
        self.assertEqual(response.json()["code"], "assistant_error")

    def test_grounded_mode_with_missing_definition_does_not_generate_a_general_answer(self):
        complete = AsyncMock(return_value=Completion({
            "mode": "term", "coverage": "complete", "article_fallback": "default", "definition_id": "e6", "blocks": [],
        }, "test-model"))
        self.app.state.llm_client = Mock(complete=complete)
        response = self.client.post(ANSWER_PATH, json={"question": "Что такое чёрная дыра?"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["coverage"], "insufficient")
        self.assertEqual(response.json()["sources"], [])
        complete.assert_awaited_once()

    def test_exhausted_deepseek_balance_is_reported_as_unavailable(self):
        error = AssistantNotConfigured("На аккаунте DeepSeek закончился баланс", 402)
        self.app.state.llm_client = Mock(complete=AsyncMock(side_effect=error))
        response = self.client.post(ANSWER_PATH, json={"question": "Что такое чёрная дыра?"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], "assistant_unavailable")
        self.assertIn("баланс", response.json()["message"])

    def test_invalid_deadline_is_reported_before_acquiring_a_slot(self):
        for value in ("nan", "0", "150", "invalid"):
            with self.subTest(value=value), patch.dict(os.environ, {"ASSISTANT_DEADLINE": value}), \
                    patch.object(self.limiter, "acquire_call") as acquire:
                response = self.client.post(ANSWER_PATH, json={"question": "Что такое звезда?"})
                self.assertEqual(response.status_code, 503, response.text)
                self.assertEqual(response.json()["code"], "assistant_unavailable")
                acquire.assert_not_called()

    def test_redis_release_failure_does_not_replace_answer(self):
        with (
            patch.object(self.limiter, "release_call", side_effect=redis.ConnectionError("unavailable")),
            self.assertLogs("app.assistant.service", level="WARNING"),
        ):
            response = self.client.post(ANSWER_PATH, json={"question": "Что такое звезда?"})
        self.assertEqual(response.status_code, 200, response.text)
