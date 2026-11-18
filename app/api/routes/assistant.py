import asyncio

import redis
from fastapi import APIRouter, Body, Request

from app.api.errors import error
from app.api.paths import ASSISTANT_ANSWER_PATH
from app.api.responses import RETRY_HEADER
from app.api.schemas.assistant import AnswerRequest, AnswerResponse
from app.api.schemas.common import ErrorResponse
from app.assistant.client import AssistantNotConfigured, AssistantUnavailable
from app.assistant.service import AssistantOverloaded, run_answer

router = APIRouter()


@router.post(
    ASSISTANT_ANSWER_PATH,
    response_model=AnswerResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Некорректный Content-Length или тело запроса"},
        413: {"model": ErrorResponse, "description": "Тело запроса превышает 16 КиБ"},
        429: {
            "model": ErrorResponse,
            "headers": RETRY_HEADER,
            "content": {
                "application/json": {
                    "example": {"code": "rate_limited", "message": "Лимит запросов превышен", "details": None}
                }
            },
        },
        503: {
            "model": ErrorResponse,
            "headers": RETRY_HEADER,
            "content": {
                "application/json": {
                    "example": {
                        "code": "assistant_unavailable",
                        "message": "Ключ LLM не настроен",
                        "details": None,
                    }
                }
            },
        },
        502: {
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {"code": "assistant_error", "message": "Сбой запроса LLM", "details": None}
                }
            },
        },
        504: {
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {
                        "code": "assistant_timeout",
                        "message": "Помощник не уложился во время ответа",
                        "details": None,
                    }
                }
            },
        },
    },
)
async def assistant_answer(
    request: Request,
    body: AnswerRequest = Body(
        openapi_examples={
            "auto": {
                "summary": "Автоматический выбор",
                "value": {"question": "Что такое звезда?", "mode": "auto", "locale": "ru"},
            },
            "term": {
                "summary": "Краткое определение",
                "value": {"question": "Что такое звезда?", "mode": "term", "locale": "ru"},
            },
            "article": {
                "summary": "Статья, если доступна",
                "value": {"question": "Расскажи подробно о звёздах", "mode": "article", "locale": "ru"},
            },
            "explain": {
                "summary": "Составное объяснение",
                "value": {"question": "Чем звезда отличается от планеты?", "mode": "explain", "locale": "ru"},
            },
            "general": {
                "summary": "Общий ответ по астрономии без справочника",
                "value": {
                    "question": "Почему планеты вращаются вокруг звёзд?",
                    "mode": "general",
                    "locale": "ru",
                },
            },
        }
    ),
):
    try:
        result = await run_answer(
            body.question,
            body.mode,
            body.locale,
            body.context,
            request.app.state.llm_client,
            request.app.state.rate_limiter,
        )
    except AssistantOverloaded as exc:
        return error(503, "assistant_overloaded", str(exc), headers={"Retry-After": "10"})
    except asyncio.TimeoutError:
        return error(504, "assistant_timeout", "Помощник не уложился во время ответа")
    except AssistantNotConfigured as exc:
        details = {"upstream_status": exc.status} if exc.status is not None else None
        return error(503, "assistant_unavailable", str(exc), details)
    except AssistantUnavailable as exc:
        return error(502, "assistant_error", str(exc))
    except redis.RedisError:
        return error(
            503, "assistant_overloaded", "Помощник временно недоступен", headers={"Retry-After": "10"}
        )
    return AnswerResponse.model_validate(result)
