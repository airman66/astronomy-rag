import logging
import mimetypes
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException

from app.api.errors import http_error, validation_error
from app.api.paths import API_PREFIX
from app.api.routes import articles, assistant, health, media, search, terms
from app.api.schemas.common import ErrorResponse
from app.assistant.client import LLMClient
from app.catalog import storage, vector
from app.infra.config import cors_origins
from app.infra.redis_limits import RedisLimiter
from app.middleware.body_limit import AssistantBodyLimitMiddleware
from app.middleware.request_limits import request_limits


@asynccontextmanager
async def lifespan(application: FastAPI):
    # A fresh checkout has no SQLite database; building the FTS5 catalogue
    # can dominate startup time, especially for a second smoke-test container.
    cold_start = not storage.DB.exists()
    catalog_started = time.perf_counter()
    await run_in_threadpool(storage.ensure_database, True)
    logging.getLogger("uvicorn.error").info(
        "Catalogue %s completed in %.2f seconds",
        "initial build" if cold_start else "check/rebuild",
        time.perf_counter() - catalog_started,
    )
    vector.warm_up()
    client = application.state.llm_client
    if isinstance(client, LLMClient) and client.http is None:
        async with httpx.AsyncClient() as http:
            client.http = http
            try:
                yield
            finally:
                client.http = None
    else:
        yield


def create_app(*, rate_limiter=None, llm_client=None) -> FastAPI:
    application = FastAPI(
        title="Astronomy Glossary and RAG",
        version="1.1.0",
        responses={422: {"model": ErrorResponse}},
        lifespan=lifespan,
        exception_handlers={RequestValidationError: validation_error, HTTPException: http_error},
    )
    application.state.rate_limiter = RedisLimiter() if rate_limiter is None else rate_limiter
    application.state.llm_client = LLMClient() if llm_client is None else llm_client

    origins = cors_origins()
    if origins:
        application.add_middleware(
            CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"], allow_credentials=True,
        )
    application.middleware("http")(request_limits)
    application.add_middleware(AssistantBodyLimitMiddleware)

    application.include_router(health.router)
    for routes in (terms, articles, media, search, assistant):
        application.include_router(routes.router, prefix=API_PREFIX)
    return application


mimetypes.add_type("image/webp", ".webp")
app = create_app()
