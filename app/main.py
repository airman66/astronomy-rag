"""Read-only glossary API and model assistant."""
from __future__ import annotations

import asyncio
import mimetypes
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import redis
from fastapi import Body, FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import storage, vector
from app.assistant import AssistantNotConfigured, AssistantUnavailable, DeepSeekClient, answer
from app.rate_limit import RedisLimiter

mimetypes.add_type("image/webp", ".webp")

# Must stay below the model slot lease (150 s).
ASSISTANT_DEADLINE = float(os.getenv("ASSISTANT_DEADLINE", "100"))
SEARCH_POOL = 200


class HealthResponse(BaseModel):
    status: str
    service: str


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Any = None


class ListItem(BaseModel):
    id: str
    title: str
    title_en: str | None = None
    url: str
    review_status: Literal["draft", "reviewed"]
    categories: list[str]
    categories_ru: list[str] = Field(default_factory=list)
    illustration_url: str | None = None


class ListResponse(BaseModel):
    items: list[ListItem]
    total: int
    page: int
    page_size: int


class FacetLetter(BaseModel):
    letter: str
    count: int


class FacetCategory(BaseModel):
    id: str
    title: str
    count: int


class FacetsResponse(BaseModel):
    locale: Literal["ru", "en"]
    total: int
    letters: list[FacetLetter]
    categories: list[FacetCategory]


class ReviewInfo(BaseModel):
    status: Literal["draft", "reviewed"]
    reviewer: str | None = None
    notes: str | None = None


class Section(BaseModel):
    id: str
    type: str
    heading: str
    body: str
    heading_ru: str | None = None
    body_ru: str | None = None
    links: list[dict[str, str]] = Field(default_factory=list)
    media_ids: list[str] = Field(default_factory=list)
    source_html: str | None = None


class LocaleContent(BaseModel):
    title: str
    aliases: list[str] = Field(default_factory=list)
    blocks: dict[str, dict[str, str]]


class MediaDetail(BaseModel):
    id: str
    kind: str
    file_type: str | None = None
    source_url: str | None = None
    record_url: str | None = None
    original_path: str | None = None
    ru_path: str | None = None
    ru_sha256: str | None = None
    sha256: str | None = None
    size: int | None = None
    title_en: str = ""
    title_ru: str = ""
    short_caption_en: str = ""
    short_caption_ru: str = ""
    caption_en: str = ""
    caption_ru: str = ""
    alt_en: str = ""
    alt_ru: str = ""
    credit: str | None = None
    credit_url: str | None = None
    license: str | None = None
    license_url: str | None = None
    term_numbers: list[int] = Field(default_factory=list)
    source_language: str = "en"
    review: ReviewInfo


class IllustrationInfo(BaseModel):
    """Cover image for a term, stored apart from Astro4Edu media."""

    term_number: int
    url: str
    width: int | None = None
    height: int | None = None
    alt_ru: str = ""
    alt_en: str = ""
    caption_ru: str = ""
    kind: Literal["conceptual", "diagram"] = "conceptual"
    status: Literal["generated", "approved"]
    generator: str = ""
    source_type: str = "generated"
    file_page: str | None = None
    article_url: str | None = None
    author: str | None = None
    license: str | None = None
    license_url: str | None = None
    modified: str | None = None


class RecordDetail(BaseModel):
    id: str
    kind: str
    source: dict[str, Any]
    metadata: dict[str, Any]
    sections: list[Section]
    locales: dict[str, LocaleContent]
    media: list[MediaDetail] = Field(default_factory=list)
    illustration: IllustrationInfo | None = None
    review: ReviewInfo


class SearchItem(BaseModel):
    record_id: str
    section_id: str
    kind: str
    title: str
    excerpt: str
    url: str
    review_status: str
    score: float


class SearchResponse(BaseModel):
    items: list[SearchItem]
    total: int
    page: int
    page_size: int
    mode: Literal["hybrid", "lexical"] = Field("hybrid", description="lexical: семантический индекс или сервер эмбеддингов недоступен, выдача только по FTS")


class AnswerRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    mode: Literal["auto", "term", "article", "explain", "general"] = "auto"
    locale: Literal["ru", "en"] = "ru"
    context: dict[str, Any] | None = None


class AnswerSource(BaseModel):
    record_id: str
    section_id: str
    url: str
    review_status: str


class AnswerImage(BaseModel):
    record_id: str
    url: str
    alt: str
    caption: str
    kind: Literal["illustration", "source_media"]
    credit: str | None = None
    credit_url: str | None = None
    license: str | None = None
    license_url: str | None = None
    modified: str | None = None


class AnswerResponse(BaseModel):
    mode: Literal["term", "article", "explain", "general", "out_of_scope"]
    grounding: Literal["knowledge_base", "model", "mixed", "policy"]
    answer: str
    primary_url: str | None
    sources: list[AnswerSource]
    images: list[AnswerImage] = Field(default_factory=list)
    review_status: str | None
    diagnostics: dict[str, str]


def error(status: int, code: str, message: str, details: Any = None, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content=ErrorResponse(code=code, message=message, details=details).model_dump(), headers=headers)


RETRY_HEADER = {"Retry-After": {"description": "Секунды до повторной попытки", "schema": {"type": "integer"}}}
LIMITS = {
    429: {"model": ErrorResponse, "description": "Лимит запросов превышен", "headers": RETRY_HEADER},
    503: {"model": ErrorResponse, "description": "Сервис ограничений временно недоступен", "headers": RETRY_HEADER},
}
IMAGE_TYPES = {"image/png": {}, "image/jpeg": {}, "image/webp": {}, "image/gif": {}, "image/svg+xml": {}}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Rebuild a stale SQLite DB before serving and verify the vector index in the background.
    await run_in_threadpool(storage.ensure_database, True)
    vector.warm_up()
    yield


app = FastAPI(title="Astronomy Glossary and RAG", version="1.1.0", responses={422: {"model": ErrorResponse}}, lifespan=lifespan)
origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"], allow_credentials=True)
app.state.rate_limiter = RedisLimiter()
app.state.deepseek = DeepSeekClient()


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    return error(422, "validation_error", "Некорректный запрос", exc.errors())


@app.exception_handler(StarletteHTTPException)
async def http_error(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return error(exc.status_code, "not_found" if exc.status_code == 404 else "http_error", str(exc.detail))


_limiter_warned = [0.0]


def limiter_down_warning() -> None:
    import logging
    import time

    if time.monotonic() - _limiter_warned[0] > 60:
        logging.getLogger("glossary.limits").warning("Redis unavailable: read limits are not enforced, assistant is closed")
        _limiter_warned[0] = time.monotonic()


@app.middleware("http")
async def request_limits(request: Request, call_next):
    path = request.url.path
    if not path.startswith("/api/v1/"):
        return await call_next(request)
    category = "assistant" if path == "/api/v1/assistant/answer" else "media" if path.endswith(("/file", "/illustration")) else "read"
    if category == "assistant":
        if int(request.headers.get("content-length", "0") or "0") > 16384 or len(await request.body()) > 16384:
            return error(413, "body_too_large", "Тело запроса превышает 16 КиБ")
    try:
        # Redis calls are blocking; run them off the event loop.
        retry = await run_in_threadpool(request.app.state.rate_limiter.check, request, category)
    except redis.RedisError:
        if category == "assistant":
            return error(503, "rate_limit_unavailable", "Сервис ограничений временно недоступен", headers={"Retry-After": "10"})
        # Reading the glossary must not depend on Redis: without shared counters only the
        # The assistant spends model calls; reads and media stay available.
        limiter_down_warning()
        retry = 0
    if retry:
        return error(429, "rate_limited", "Лимит запросов превышен", headers={"Retry-After": str(min(10, max(1, retry)))})
    return await call_next(request)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="glossary-rag")


@app.get("/api/v1/terms", response_model=ListResponse, responses=LIMITS)
def terms(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100), q: str = Query("", max_length=200), category: str = "", letter: str = Query("", max_length=1), status: Literal["", "draft", "reviewed"] = "", locale: Literal["ru", "en"] = "ru") -> ListResponse:
    """Список терминов. Поиск по названию и алиасам и фильтр по букве не зависят от регистра и «ё»; сортировка — по названию на языке `locale`."""
    items, total = storage.list_records("term", page=page, page_size=page_size, locale=locale, query=q, category=category, letter=letter, status=status)
    return ListResponse(items=items, total=total, page=page, page_size=page_size)


@app.get("/api/v1/terms/facets", response_model=FacetsResponse, responses=LIMITS)
def term_facets(locale: Literal["ru", "en"] = "ru") -> FacetsResponse:
    """Первые буквы названий и категории с числом терминов — для алфавитного указателя и фильтров."""
    letters = storage.letters(locale)
    return FacetsResponse(locale=locale, total=sum(item["count"] for item in letters), letters=letters, categories=storage.categories(locale))


@app.get("/api/v1/terms/{term_number}", response_model=RecordDetail, responses={404: {"model": ErrorResponse}, **LIMITS})
def term_detail(term_number: int):
    record = storage.get_term(term_number)
    return RecordDetail.model_validate(record) if record else error(404, "term_not_found", "Термин не найден")


@app.get("/api/v1/terms/{term_number}/illustration", response_class=FileResponse, responses={200: {"content": IMAGE_TYPES, "description": "Иллюстрация термина"}, 404: {"model": ErrorResponse}, **LIMITS})
def term_illustration(term_number: int):
    """Файл иллюстрации команды; 404 `illustration_not_found`, пока она не сгенерирована и не проверена."""
    item = storage.get_illustration(term_number)
    if not item:
        return error(404, "illustration_not_found", "Иллюстрация ещё не готова")
    path = (storage.DATA / item["path"]).resolve() if not Path(item["path"]).is_absolute() else Path(item["path"]).resolve()
    if not path.is_relative_to(storage.DATA.resolve()) or not path.is_file():
        return error(404, "illustration_not_found", "Файл иллюстрации не найден")
    return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/v1/articles", response_model=ListResponse, responses=LIMITS)
def articles(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100), q: str = Query("", max_length=200), status: Literal["", "draft", "reviewed"] = "", locale: Literal["ru", "en"] = "ru") -> ListResponse:
    items, total = storage.list_records("article", page=page, page_size=page_size, locale=locale, query=q, status=status)
    return ListResponse(items=items, total=total, page=page, page_size=page_size)


@app.get("/api/v1/articles/{article_id}", response_model=RecordDetail, responses={404: {"model": ErrorResponse}, **LIMITS})
def article_detail(article_id: str):
    record = storage.get_article(article_id)
    return RecordDetail.model_validate(record) if record else error(404, "article_not_found", "Статья не найдена")


@app.get("/api/v1/media/{media_id}", response_model=MediaDetail, responses={404: {"model": ErrorResponse}, **LIMITS})
def media_detail(media_id: str):
    item = storage.get_media(media_id)
    return MediaDetail.model_validate(item) if item else error(404, "media_not_found", "Медиа не найдено")


@app.get("/api/v1/media/{media_id}/file", response_class=FileResponse, responses={200: {"content": IMAGE_TYPES, "description": "Файл изображения на выбранном языке"}, 404: {"model": ErrorResponse}, **LIMITS})
def media_file(media_id: str, locale: Literal["ru", "en"] = "ru"):
    item = storage.get_media(media_id)
    if not item:
        return error(404, "media_not_found", "Медиа не найдено")
    filename = item.get("ru_path") if locale == "ru" and item.get("ru_path") else item.get("original_path")
    if not filename:
        return error(404, "media_file_not_found", "Файл медиа не найден")
    path = Path(filename).resolve()
    if not path.is_relative_to(storage.DATA.resolve()) or not path.is_file():
        return error(404, "media_file_not_found", "Файл медиа не найден")
    return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/v1/search", response_model=SearchResponse, responses=LIMITS)
def search(q: str = Query(min_length=1, max_length=200), locale: Literal["ru", "en"] = "ru", page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100)) -> SearchResponse:
    """Гибридный поиск (FTS5 со стеммингом + векторный индекс). `total` — размер пула релевантных результатов (не больше 200)."""
    matches, mode = storage.search_with_mode(q, locale, limit=SEARCH_POOL)
    return SearchResponse(items=matches[(page - 1) * page_size : page * page_size], total=len(matches), page=page, page_size=page_size, mode=mode)


@app.post("/api/v1/assistant/answer", response_model=AnswerResponse, responses={
    429: {"model": ErrorResponse, "headers": RETRY_HEADER, "content": {"application/json": {"example": {"code": "rate_limited", "message": "Лимит запросов превышен", "details": None}}}},
    503: {"model": ErrorResponse, "headers": RETRY_HEADER, "content": {"application/json": {"example": {"code": "assistant_unavailable", "message": "Ключ LLM не настроен", "details": None}}}},
    502: {"model": ErrorResponse, "content": {"application/json": {"example": {"code": "assistant_error", "message": "Сбой запроса LLM", "details": None}}}},
    504: {"model": ErrorResponse, "content": {"application/json": {"example": {"code": "assistant_timeout", "message": "Помощник не уложился во время ответа", "details": None}}}},
})
async def assistant_answer(request: Request, body: AnswerRequest = Body(openapi_examples={
    "auto": {"summary": "Автоматический выбор", "value": {"question": "Что такое звезда?", "mode": "auto", "locale": "ru"}},
    "term": {"summary": "Краткое определение", "value": {"question": "Что такое звезда?", "mode": "term", "locale": "ru"}},
    "article": {"summary": "Статья, если доступна", "value": {"question": "Расскажи подробно о звёздах", "mode": "article", "locale": "ru"}},
    "explain": {"summary": "Составное объяснение", "value": {"question": "Чем звезда отличается от планеты?", "mode": "explain", "locale": "ru"}},
    "general": {"summary": "Общий ответ по астрономии без справочника", "value": {"question": "Почему планеты вращаются вокруг звёзд?", "mode": "general", "locale": "ru"}},
})):
    provider = os.getenv("ASSISTANT_PROVIDER", "deepseek").lower()
    if provider not in ("deepseek", "runpod"):
        return error(503, "assistant_unavailable", "Неизвестный провайдер LLM")
    if not os.getenv("RUNPOD_API_KEY" if provider == "runpod" else "DEEPSEEK_API_KEY"):
        return error(503, "assistant_unavailable", "Ключ LLM не настроен")
    token = str(uuid.uuid4())
    limiter = request.app.state.rate_limiter
    try:
        if not await run_in_threadpool(limiter.acquire_call, token):
            return error(503, "assistant_overloaded", "Помощник перегружен", headers={"Retry-After": "10"})
        try:
            result = await asyncio.wait_for(answer(body.question, body.mode, body.locale, body.context, request.app.state.deepseek), timeout=ASSISTANT_DEADLINE)
        finally:
            await run_in_threadpool(limiter.release_call, token)
    except asyncio.TimeoutError:
        return error(504, "assistant_timeout", "Помощник не уложился во время ответа")
    except AssistantNotConfigured as exc:
        # Ключ без баланса или отклонённый ключ: справочник работает, помощник выключен.
        return error(503, "assistant_unavailable", str(exc), {"upstream_status": exc.status})
    except AssistantUnavailable as exc:
        return error(502, "assistant_error", str(exc))
    except redis.RedisError:
        return error(503, "assistant_overloaded", "Помощник временно недоступен", headers={"Retry-After": "10"})
    return AnswerResponse.model_validate(result)
