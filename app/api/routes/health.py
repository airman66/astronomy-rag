import os
import sqlite3
import time
from threading import Lock

from fastapi import APIRouter, Response
from fastapi.responses import PlainTextResponse

from app.api.schemas.common import HealthResponse, ReadyResponse
from app.catalog import storage, vector
from app.infra import metrics

router = APIRouter()
_probe_lock = Lock()
_probe: tuple[float, bool] = (float("-inf"), False)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="glossary-rag")


def semantic_ready() -> bool:
    global _probe
    with _probe_lock:
        if time.monotonic() - _probe[0] < 15:
            return _probe[1]
        try:
            vector.search("What is a star?", "en", limit=1, timeout=2.0)
            ready = vector.available()
        except Exception:  # A failed dependency probe must be reported as unavailable.
            ready = False
        _probe = time.monotonic(), ready
        return ready


@router.get("/ready", response_model=ReadyResponse, responses={503: {"model": ReadyResponse}})
def readiness(response: Response) -> ReadyResponse:
    required = os.getenv("REQUIRE_HYBRID") == "1"
    try:
        catalogue = bool(storage.corpus_version())
    except (sqlite3.Error, OSError, ValueError):
        catalogue = False
    semantic = catalogue and semantic_ready()
    if not catalogue or (required and not semantic):
        response.status_code = 503
        status = "unavailable"
    else:
        status = "ready" if semantic else "degraded"
    return ReadyResponse(
        status=status, catalogue="ready" if catalogue else "unavailable",
        semantic="ready" if semantic else "unavailable", hybrid_required=required,
    )


@router.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics() -> PlainTextResponse:
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")
