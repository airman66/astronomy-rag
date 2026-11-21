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

    async def test_deepseek_request_disables_thinking(self):
        def handle(request: httpx.Request) -> httpx.Response:
            self.assertEqual(str(request.url), "https://api.deepseek.com/chat/completions")
            payload = json.loads(request.content)
            self.assertEqual(payload["thinking"], {"type": "disabled"})
            self.assertEqual(payload["max_tokens"], 768)
            self.assertNotIn("chat_template_kwargs", payload)
            return httpx.Response(200, json={
                "model": "deepseek-test",
                "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
            })

        settings = LLMSettings("deepseek", "deepseek-flash", "test")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            completion = await LLMClient(http, settings).complete("system", {})
        self.assertEqual(completion.model, "deepseek-test")
        self.assertEqual(completion.usage, {})

    async def test_glm_uses_its_own_generation_defaults(self):
        def handle(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            self.assertEqual(payload["max_tokens"], 1536)
            self.assertEqual(payload["reasoning_effort"], "low")
            self.assertEqual(payload["temperature"], 1.0)
            self.assertEqual(payload["top_p"], 0.95)
            return httpx.Response(200, json={
                "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
            })

        settings = LLMSettings("runpod", "glm-5.3", "test", "test-endpoint")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            completion = await LLMClient(http, settings).complete("system", {})
        self.assertEqual(completion.model, "glm-5.3")
