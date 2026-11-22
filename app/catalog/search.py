"""Merge lexical and semantic result lists with reciprocal rank fusion."""

import logging
import os
import time
from collections import OrderedDict
from threading import Lock
from typing import Literal

from app.catalog import storage, vector
from app.catalog.index import MANIFEST_NAME
from app.infra.metrics import count

SEMANTIC_POOL = int(os.getenv("SEMANTIC_POOL") or "40")
log = logging.getLogger("glossary.search")


_last_semantic_error = float("-inf")


def search_with_mode(query: str, locale: str, limit: int = 20) -> tuple[list[dict], Literal["hybrid", "lexical"]]:
    """Hybrid FTS + vector search; falls back to FTS alone when embeddings are unavailable."""
    global _last_semantic_error

    lexical = storage.lexical_search(query, locale, limit)

    try:
        semantic = vector.search(query, locale, min(limit, SEMANTIC_POOL))
    except Exception as exc:  # embedding server/model/index problems must not break search
        if time.monotonic() - _last_semantic_error > 60:
            log.warning("semantic search unavailable, using lexical results: %s: %s", type(exc).__name__, exc)
            _last_semantic_error = time.monotonic()
        count("astronomy_search_lexical_total")
        return lexical, "lexical"
    if not vector.available():
        count("astronomy_search_lexical_total")
        return lexical, "lexical"
    fused: dict[tuple[str, str], dict] = {}
    for ranked in (lexical, semantic):
        seen: set[tuple[str, str]] = set()
        for item in ranked:
            key = item["record_id"], item["section_id"]
            if key in seen:
                continue
            seen.add(key)
            if key not in fused:
                fused[key] = {**item, "score": 0.0}
            if "chunk_id" in item and "chunk_id" not in fused[key]:
                fused[key].update(chunk_id=item["chunk_id"], evidence_text=item["evidence_text"])
            fused[key]["score"] += 1.0 / (60 + len(seen))
    count("astronomy_search_hybrid_total")
    return sorted(fused.values(), key=lambda row: row["score"], reverse=True)[:limit], "hybrid"


_PAGE_CACHE_TTL = 15.0
_PAGE_CACHE_LIMIT = 128
_page_cache: OrderedDict[tuple, tuple[float, list[dict]]] = OrderedDict()
_page_lock = Lock()


def paged_search(query: str, locale: str, limit: int) -> tuple[list[dict], Literal["hybrid", "lexical"]]:
    # Share a pool across pages without changing the case of semantic query terms.
    query = " ".join(query.split())
    try:
        files = (storage.DB, vector.DATA / "index" / MANIFEST_NAME)
        generation = tuple((file.resolve(), stat.st_ino, stat.st_mtime_ns, stat.st_size)
                           for file in files for stat in (file.stat(),))
    except OSError:
        return search_with_mode(query, locale, limit)
    key = (query, locale, limit, generation)
    with _page_lock:
        cached = _page_cache.get(key)
        if cached and time.monotonic() - cached[0] < _PAGE_CACHE_TTL:
            _page_cache.move_to_end(key)
            count("astronomy_search_cache_hits_total")
            return cached[1], "hybrid"
    rows, mode = search_with_mode(query, locale, limit)
    if mode == "hybrid":
        with _page_lock:
            _page_cache[key] = time.monotonic(), rows
            _page_cache.move_to_end(key)
            if len(_page_cache) > _PAGE_CACHE_LIMIT:
                _page_cache.popitem(last=False)
    return rows, mode
