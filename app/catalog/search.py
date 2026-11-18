"""Merge lexical and semantic result lists with reciprocal rank fusion."""

import logging
import os
import time
from typing import Literal

from app.catalog import storage, vector

SEMANTIC_POOL = int(os.getenv("SEMANTIC_POOL", "40"))
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
        return lexical, "lexical"
    if not vector.available():
        return lexical, "lexical"
    fused: dict[tuple[str, str], dict] = {}
    for ranked in (lexical, semantic):
        for rank, item in enumerate(ranked):
            key = item["record_id"], item["section_id"]
            if key not in fused:
                fused[key] = {**item, "score": 0.0}
            fused[key]["score"] += 1.0 / (60 + rank + 1)
    return sorted(fused.values(), key=lambda row: row["score"], reverse=True)[:limit], "hybrid"
