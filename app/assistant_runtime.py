"""Run one assistant request under the shared model concurrency limit."""

import asyncio
import uuid

from fastapi.concurrency import run_in_threadpool

from app.assistant import answer
from app.config import LLMSettings, assistant_deadline
from app.llm import CompletionClient
from app.rate_limit import RedisLimiter


class AssistantConfigurationError(Exception):
    pass


class AssistantOverloaded(Exception):
    pass


async def run_answer(
    question: str,
    mode: str,
    locale: str,
    context: dict | None,
    client: CompletionClient,
    limiter: RedisLimiter,
) -> dict:
    settings = LLMSettings.from_env()
    if settings.provider not in ("deepseek", "runpod"):
        raise AssistantConfigurationError("Неизвестный провайдер LLM")
    if not settings.api_key:
        raise AssistantConfigurationError("Ключ LLM не настроен")

    deadline = assistant_deadline()
    token = str(uuid.uuid4())
    if not await run_in_threadpool(limiter.acquire_call, token):
        raise AssistantOverloaded("Помощник перегружен")
    try:
        return await asyncio.wait_for(answer(question, mode, locale, context, client), timeout=deadline)
    finally:
        await run_in_threadpool(limiter.release_call, token)
