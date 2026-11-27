"""OpenAI-compatible requests for the two benchmark candidates."""

import json
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

DEEPSEEK = "deepseek"
GIGACHAT = "gigachat"
MODELS = {
    DEEPSEEK: "deepseek-flash",
    GIGACHAT: "GigaChat/GigaChat-4-708B-A41",
}
URLS = {
    DEEPSEEK: "https://api.deepseek.com/chat/completions",
    GIGACHAT: "https://foundation-models.api.cloud.ru/v1/chat/completions",
}
KEYS = {DEEPSEEK: "DEEPSEEK_API_KEY", GIGACHAT: "CLOUD_RU_API_KEY"}


@dataclass(frozen=True)
class Candidate:
    name: str
    max_tokens: int
    api_key: str = field(repr=False)

    @classmethod
    def from_env(cls, name: str, max_tokens: int) -> "Candidate":
        if not (key := os.getenv(KEYS[name])):
            raise SystemExit(f"Missing {KEYS[name]}")
        return cls(name, max_tokens, key)

    def configuration(self) -> dict[str, Any]:
        options = (
            {
                "thinking": {"type": "enabled"},
                "reasoning_effort": "max",
            }
            if self.name == DEEPSEEK
            else {
                "reasoning_effort": "high",
                "provider": {"sort": "price"},
            }
        )
        return {
            "model": MODELS[self.name],
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "stream": False,
            **options,
        }

    async def call(self, http: httpx.AsyncClient, system: str, payload: dict) -> dict:
        request = {
            **self.configuration(),
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": json.dumps(
                        payload, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            ],
        }
        try:
            response = await http.post(
                URLS[self.name],
                json=request,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return {
                "error": f"HTTP {exc.response.status_code}",
                "http_status": exc.response.status_code,
            }
        except httpx.HTTPError as exc:
            return {"error": type(exc).__name__}
        try:
            body = response.json()
            choice = body["choices"][0]
            usage = body.get("usage")
            usage = usage if isinstance(usage, dict) else {}
            message = choice["message"]
            reasoning_details = usage.get("completion_tokens_details") or {}
            reasoning_tokens = reasoning_details.get("reasoning_tokens", 0)
            reasoning_visible = any(
                isinstance(message.get(key), str) and bool(message[key])
                for key in ("reasoning_content", "reasoning")
            )
            return {
                "reported_model": body.get("model"),
                "reported_provider": body.get("provider"),
                "finish_reason": choice.get("finish_reason"),
                "usage": usage,
                "reasoning_observed": reasoning_visible
                or (isinstance(reasoning_tokens, int) and reasoning_tokens > 0),
                "content": message.get("content"),
            }
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return {"error": "completion envelope"}
