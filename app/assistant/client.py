"""JSON completions via the configured model provider."""

import json
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from app.infra.config import LLMSettings


class ChatTransport(Protocol):
    async def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: httpx.Timeout,
    ) -> httpx.Response: ...


@dataclass(frozen=True)
class Completion:
    data: Any
    model: str
    usage: dict[str, Any] = field(default_factory=dict)


class CompletionClient(Protocol):
    async def complete(self, system: str, payload: dict[str, Any]) -> Completion: ...


class AssistantUnavailable(Exception):
    def __init__(self, message: str, *, usage: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.usage = usage or {}


class AssistantNotConfigured(AssistantUnavailable):
    """Missing local configuration or a rejected provider account."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class LLMClient:
    def __init__(
        self,
        http: ChatTransport | None = None,
        settings: LLMSettings | None = None,
        read_timeout: float | None = None,
    ) -> None:
        self.http = http
        self.settings = settings
        self.read_timeout = read_timeout

    def configured_settings(self) -> LLMSettings:
        try:
            settings = self.settings or LLMSettings.from_env()
        except ValueError as exc:
            raise AssistantNotConfigured("Некорректный бюджет токенов помощника") from exc
        if settings.provider not in ("deepseek", "runpod"):
            raise AssistantNotConfigured("Неизвестный провайдер LLM")
        if not settings.api_key:
            raise AssistantNotConfigured("Ключ LLM не настроен")
        if settings.provider == "runpod" and not settings.endpoint_id:
            raise AssistantNotConfigured("RunPod endpoint не настроен")
        return settings

    async def complete(self, system: str, payload: dict[str, Any]) -> Completion:
        settings = self.configured_settings()
        is_runpod = settings.provider == "runpod"
        is_glm = is_runpod and "glm" in settings.model.casefold()
        request: dict[str, Any] = {
            "model": settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": settings.max_tokens or (1536 if is_glm else 768),
            "stream": False,
        }
        if is_glm:
            request.update(temperature=1.0, top_p=0.95, reasoning_effort=settings.reasoning_effort or "low")
        if is_runpod and settings.thinking != "default":
            request["chat_template_kwargs"] = {"enable_thinking": settings.thinking == "enabled"}
        if not is_runpod:
            request["thinking"] = {"type": "disabled"}

        url = (
            f"https://api.runpod.ai/v2/{settings.endpoint_id}/openai/v1/chat/completions"
            if is_runpod else "https://api.deepseek.com/chat/completions"
        )
        timeout = httpx.Timeout(self.read_timeout or (90.0 if is_runpod else 35.0), connect=5.0)
        usage: dict[str, Any] = {}
        try:
            connection = httpx.AsyncClient() if self.http is None else nullcontext(self.http)
            async with connection as http:
                response = await http.post(
                    url,
                    headers={"Authorization": f"Bearer {settings.api_key}"},
                    json=request,
                    timeout=timeout,
                )
            response.raise_for_status()
            body = response.json()
            reported_usage = body.get("usage")
            if isinstance(reported_usage, dict):
                usage = reported_usage
            choice = body["choices"][0]
            if choice.get("finish_reason") not in (None, "stop"):
                raise AssistantUnavailable("Модель не завершила ответ в заданном бюджете", usage=usage)
            reported_model = body.get("model")
            if not isinstance(reported_model, str) or not reported_model:
                reported_model = settings.model
            return Completion(json.loads(choice["message"]["content"]), reported_model, usage)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 402:
                raise AssistantNotConfigured("На аккаунте провайдера LLM закончился баланс", status) from exc
            if status in (401, 403):
                raise AssistantNotConfigured("Провайдер LLM отклонил ключ API", status) from exc
            raise AssistantUnavailable(f"LLM request failed: HTTP {status}") from exc
        except (httpx.HTTPError, AttributeError, KeyError, IndexError, ValueError, TypeError) as exc:
            raise AssistantUnavailable(f"LLM request failed: {type(exc).__name__}", usage=usage) from exc
