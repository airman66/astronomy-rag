import json
import unittest

import httpx

from app.assistant.client import AssistantUnavailable, LLMClient
from app.infra.config import LLMSettings


class CompletionTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_qwen_has_independent_thinking_and_token_settings(self):
        def handle(request: httpx.Request) -> httpx.Response:
            self.assertTrue(request.url.path.endswith("/qwen3-32b-awq/openai/v1/chat/completions"))
            payload = json.loads(request.content)
            self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
            self.assertEqual(payload["max_tokens"], 768)
            self.assertNotIn("reasoning_effort", payload)
            self.assertNotIn("evidence", payload["messages"][0]["content"])
            self.assertIn("evidence", payload["messages"][1]["content"])
            return httpx.Response(200, json={
                "choices": [{"finish_reason": "stop", "message": {"content": '{"mode":"term"}'}}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 12},
            })

        settings = LLMSettings("runpod", "Qwen/Qwen3-32B-AWQ", "test", "qwen3-32b-awq", thinking="disabled")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            model = LLMClient(http, settings)
            completion = await model.complete("system", {"evidence": []})
            self.assertEqual(completion.data, {"mode": "term"})
            self.assertEqual(completion.model, settings.model)
            self.assertEqual(completion.usage["completion_tokens"], 12)

    async def test_truncated_completion_is_not_published(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
            "choices": [{"finish_reason": "length", "message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 1536},
        }))
        settings = LLMSettings("runpod", "glm-5.3", "test", "test-endpoint")
        async with httpx.AsyncClient(transport=transport) as http:
            with self.assertRaises(AssistantUnavailable) as raised:
                await LLMClient(http, settings).complete("system", {})
        self.assertEqual(raised.exception.usage["completion_tokens"], 1536)
