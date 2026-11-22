import os
import sqlite3
import time
from contextlib import contextmanager
from threading import Event, Thread

from fastapi import APIRouter, Request, Response
from fastapi.responses import PlainTextResponse

from app.api.schemas.common import HealthResponse, ReadyResponse
from app.catalog import storage, vector
from app.infra import metrics

router = APIRouter()
PROBE_INTERVAL = 15.0


class SemanticReadiness:
    def __init__(self) -> None:
        self._result: tuple[float, str | None] = (float("-inf"), None)

    def ready(self, corpus: str) -> bool:
        checked, probed_corpus = self._result
        return corpus == probed_corpus and time.monotonic() - checked < 2 * PROBE_INTERVAL + vector.QUERY_TIMEOUT

    @contextmanager
    def monitor(self):
        stop = Event()
        Thread(target=self._check, args=(stop,), name="semantic-readiness", daemon=True).start()
        try:
            yield
        finally:
            stop.set()

    def _check(self, stop: Event) -> None:
        while not stop.is_set():
            ready_corpus = None
            try:
                corpus = storage.corpus_version()
                if corpus and vector.search("What is a star?", "en", limit=1, corpus_sha=corpus) is not None:
                    ready_corpus = corpus
            except Exception:  # A failed dependency probe must be reported as unavailable.
                pass
            self._result = (time.monotonic(), ready_corpus)
            stop.wait(PROBE_INTERVAL)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="glossary-rag")


@router.get("/ready", response_model=ReadyResponse, responses={503: {"model": ReadyResponse}})
def readiness(request: Request, response: Response) -> ReadyResponse:
    required = os.getenv("REQUIRE_HYBRID") == "1"
    try:
        corpus = storage.corpus_version()
    except (sqlite3.Error, OSError, ValueError):
        corpus = None
    probe: SemanticReadiness | None = getattr(request.app.state, "semantic_readiness", None)
    semantic = bool(corpus and probe and probe.ready(corpus))
    if not corpus or (required and not semantic):
        response.status_code = 503
        status = "unavailable"
    else:
        status = "ready" if semantic else "degraded"
    return ReadyResponse(
        status=status, catalogue="ready" if corpus else "unavailable",
        semantic="ready" if semantic else "unavailable", hybrid_required=required,
    )


@router.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics() -> PlainTextResponse:
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")
