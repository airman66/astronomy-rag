"""JSON completions via the configured model provider."""

import json
from typing import Any, Protocol

import httpx

from app.infra.config import LLMSettings


class CompletionClient(Protocol):
    async def complete(self, system: str, payload: dict[str, Any]) -> Any: ...


class AssistantUnavailable(Exception):
    pass


class AssistantNotConfigured(AssistantUnavailable):
    """The model provider rejected the account: not a transient failure."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


class LLMClient:
    async def complete(self, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        settings = LLMSettings.from_env()
        runpod = settings.provider == "runpod"
        if settings.provider not in ("deepseek", "runpod"):
            raise AssistantUnavailable("Неизвестный провайдер LLM")
        if not settings.api_key:
            raise AssistantUnavailable("Ключ LLM не настроен")
        if runpod and not settings.endpoint_id:
            raise AssistantUnavailable("RunPod endpoint не настроен")
        request = {
            "model": settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 4096 if runpod else 1800,
            "stream": False,
        }
        if runpod:
            request.update({"temperature": 1.0, "top_p": 0.95, "reasoning_effort": "low"})
        else:
            request["thinking"] = {"type": "disabled"}
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(150.0 if runpod else 45.0, connect=10.0)
            ) as client:
                response = await client.post(
                    f"https://api.runpod.ai/v2/{settings.endpoint_id}/openai/v1/chat/completions"
                    if runpod
                    else "https://api.deepseek.com/chat/completions",
                    headers={"Authorization": f"Bearer {settings.api_key}"},
                    json=request,
                )
                response.raise_for_status()
            return json.loads(response.json()["choices"][0]["message"]["content"])
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 402:
                raise AssistantNotConfigured("На аккаунте провайдера LLM закончился баланс", status) from exc
            if status in (401, 403):
                raise AssistantNotConfigured("Провайдер LLM отклонил ключ API", status) from exc
            raise AssistantUnavailable(f"LLM request failed: HTTP {status}") from exc
        except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError) as exc:
            raise AssistantUnavailable(f"LLM request failed: {type(exc).__name__}") from exc
