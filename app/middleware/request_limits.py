import logging
from collections.abc import Awaitable, Callable
import time

import redis
from fastapi import Request
from fastapi.concurrency import run_in_threadpool
from starlette.responses import Response

from app.api.errors import error
from app.api.paths import API_PREFIX, ASSISTANT_ANSWER_PATH
from app.middleware.client_identity import client_identity

log = logging.getLogger("glossary.limits")
_last_warning = float("-inf")


def warn_limiter_down() -> None:
    global _last_warning

    now = time.monotonic()
    if now - _last_warning > 60:
        log.warning("Redis unavailable: read limits are not enforced, assistant is closed")
        _last_warning = now


def request_category(path: str) -> str:
    if path == API_PREFIX + ASSISTANT_ANSWER_PATH:
        return "assistant"
    if path.endswith(("/file", "/illustration", "/source.pdf", "/translation.pdf")):
        return "media"
    return "read"


async def request_limits(
    request: Request, call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    path = request.url.path
    if not path.startswith(API_PREFIX + "/"):
        return await call_next(request)
    category = request_category(path)
    identity, ip = client_identity(request)
    try:
        retry = await run_in_threadpool(request.app.state.rate_limiter.check, identity, ip, category)
    except redis.RedisError:
        if category == "assistant":
            return error(
                503,
                "rate_limit_unavailable",
                "Сервис ограничений временно недоступен",
                headers={"Retry-After": "10"},
            )
        warn_limiter_down()
        return await call_next(request)

    if retry:
        return error(
            429,
            "rate_limited",
            "Лимит запросов превышен",
            headers={"Retry-After": str(min(10, max(1, retry)))},
        )
    return await call_next(request)
