"""JSON completions via the configured model provider."""

import json
from contextlib import nullcontext
from typing import Any, Protocol

import httpx

from app.infra.config import LLMSettings


class CompletionClient(Protocol):
    async def complete(self, system: str, payload: dict[str, Any]) -> Any: ...


class AssistantUnavailable(Exception):
    pass


class AssistantNotConfigured(AssistantUnavailable):
    """The model provider rejected the account."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


class LLMClient:
    def __init__(self, http: httpx.AsyncClient | None = None, settings: LLMSettings | None = None) -> None:
        self.http = http
        self.settings = settings
        self.usage: dict = {}

    async def complete(self, system: str, payload: dict[str, Any]) -> Any:
        settings = self.settings or LLMSettings.from_env()
        runpod = settings.provider == "runpod"
        if settings.provider not in ("deepseek", "runpod"):
            raise AssistantUnavailable("Неизвестный провайдер LLM")
        if not settings.api_key:
            raise AssistantUnavailable("Ключ LLM не настроен")
        if runpod and not settings.endpoint_id:
            raise AssistantUnavailable("RunPod endpoint не настроен")

        model = settings.model.casefold()
        glm = runpod and "glm" in model
        qwen = runpod and "qwen" in model
        request: dict[str, Any] = {
            "model": settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": settings.max_tokens or (1536 if glm else 768),
            "stream": False,
        }
        if glm:
            request.update(temperature=1.0, top_p=0.95, reasoning_effort=settings.reasoning_effort or "low")
        if qwen and settings.thinking != "default":
            request["chat_template_kwargs"] = {"enable_thinking": settings.thinking == "enabled"}
        if not runpod:
            request["thinking"] = {"type": "disabled"}

        url = (
            f"https://api.runpod.ai/v2/{settings.endpoint_id}/openai/v1/chat/completions"
            if runpod else "https://api.deepseek.com/chat/completions"
        )
        timeout = httpx.Timeout(90.0 if runpod else 35.0, connect=5.0)
        self.usage = {}
        try:
            connection = httpx.AsyncClient() if self.http is None else nullcontext(self.http)
            async with connection as http:
                response = await http.post(url, headers={"Authorization": f"Bearer {settings.api_key}"},
                                           json=request, timeout=timeout)
            response.raise_for_status()
            body = response.json()
            usage = body.get("usage")
            self.usage = usage if isinstance(usage, dict) else {}
            choice = body["choices"][0]
            if choice.get("finish_reason") not in (None, "stop"):
                raise AssistantUnavailable("Модель не завершила ответ в заданном бюджете")
            return json.loads(choice["message"]["content"])
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 402:
                raise AssistantNotConfigured("На аккаунте провайдера LLM закончился баланс", status) from exc
            if status in (401, 403):
                raise AssistantNotConfigured("Провайдер LLM отклонил ключ API", status) from exc
            raise AssistantUnavailable(f"LLM request failed: HTTP {status}") from exc
        except (httpx.HTTPError, AttributeError, KeyError, IndexError, ValueError, TypeError) as exc:
            raise AssistantUnavailable(f"LLM request failed: {type(exc).__name__}") from exc
