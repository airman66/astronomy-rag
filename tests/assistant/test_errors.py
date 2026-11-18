import os
from unittest.mock import AsyncMock, Mock, patch

from app.assistant.client import AssistantNotConfigured
from tests.support.api import ANSWER_PATH, ApiCase
from tests.support.catalog import seed_search_terms


class AssistantErrorTests(ApiCase):
    def setUp(self) -> None:
        super().setUp()
        seed_search_terms(self.root)

    def test_missing_api_key_returns_unavailable(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            response = self.client.post(ANSWER_PATH, json={"question": "Что такое звезда?"})
        self.assertEqual(response.json()["code"], "assistant_unavailable")

    def test_assistant_rejects_malformed_model_json(self):
        self.app.state.llm_client = Mock(complete=AsyncMock(return_value=["not", "an", "object"]))
        response = self.client.post(ANSWER_PATH, json={"question": "Что такое чёрная дыра?"})
        self.assertEqual(response.status_code, 502, response.text)
        self.assertEqual(response.json()["code"], "assistant_error")

    def test_grounded_mode_with_missing_definition_does_not_generate_a_general_answer(self):
        complete = AsyncMock(return_value={
            "mode": "term", "coverage": "complete", "definition_id": "e6", "blocks": [],
        })
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
