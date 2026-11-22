"""CPU search over a verified Runpod/GPU-built FAISS index.

The index is optional for the service: when the manifest, model file or embedding server is
missing, callers fall back to lexical search. Load failures are cached briefly so a broken
index is not re-verified (4.7 GB model hash) on every request.
"""
from __future__ import annotations

import logging
import os
import time
from threading import Lock
from typing import NamedTuple

from app.catalog import storage
from app.catalog.index import MANIFEST_NAME, load_index
from app.infra.config import data_directory
from app.infra.metrics import count

DATA = data_directory()
QUERY_TIMEOUT = float(os.getenv("EMBEDDING_TIMEOUT") or "15")
# Cosine similarity below which a neighbour is not a match. Calibrated on Qwen3-Embedding-8B: relevant
# Russian hits 0.58–0.83, off-topic queries up to 0.46, nonsense up to 0.54 (scripts/calibrate, 2026-11-14).
MIN_SCORE = float(os.getenv("SEMANTIC_MIN_SCORE") or "0.55")
RETRY_AFTER_FAILURE = 60.0
_lock = Lock()


class _LoadedIndex(NamedTuple):
    generation: str
    artifacts: tuple


_loaded: _LoadedIndex | None = None
_checked_at = 0.0
_failure: dict[str, tuple[float, Exception]] = {}
log = logging.getLogger("glossary.vector")


def _failure_key() -> str:
    manifest = DATA / "index" / MANIFEST_NAME
    try:
        stat = manifest.stat()
    except OSError:
        return f"{DATA.resolve()}|missing"
    return f"{DATA.resolve()}|{stat.st_ino}|{stat.st_mtime_ns}|{stat.st_ctime_ns}|{stat.st_size}"


def available() -> bool:
    return _loaded is not None


def loaded_generation() -> str | None:
    loaded = _loaded
    return loaded.generation if loaded is not None else None


def _changed_on_disk(generation: str) -> bool:
    """A republished index (new manifest) replaces the loaded one; checked at most every 30 s."""
    global _checked_at
    now = time.monotonic()
    if now - _checked_at < 30:
        return False
    _checked_at = now
    return _failure_key() != generation


def load():
    global _loaded, _checked_at
    current = _loaded
    if current is not None and not _changed_on_disk(current.generation):
        return current.artifacts
    root = DATA / "index"
    if not (root / MANIFEST_NAME).exists():
        return current.artifacts if current is not None else None
    key = _failure_key()
    failed = _failure.get(key)
    if failed and time.monotonic() - failed[0] < RETRY_AFTER_FAILURE:
        if current is not None:
            return current.artifacts  # keep serving the previous index while the new one is inconsistent
        raise failed[1]
    with _lock:
        if _loaded is not None and _loaded.generation == key:
            return _loaded.artifacts
        try:
            loaded = load_index(root, DATA / "models")
        except Exception as exc:
            _failure[key] = (time.monotonic(), exc)
            if _loaded is not None:
                log.warning("new vector index rejected, keeping the loaded one: %s", exc)
                return _loaded.artifacts
            raise
        _failure.pop(key, None)
        _loaded = _LoadedIndex(key, loaded)
        _checked_at = time.monotonic()
        log.info("vector index loaded: %s chunks", loaded[0].ntotal)
        return loaded


def search(
    query: str, locale: str, limit: int = 20, min_score: float | None = None,
    *, timeout: float = QUERY_TIMEOUT, corpus_sha: str | None = None,
) -> list[dict]:
    loaded = load()
    if loaded is None:
        return []
    index, mapping, chunks, manifest = loaded
    source_sha = manifest.get("corpus_sha256")
    catalog_sha = corpus_sha if corpus_sha is not None else storage.corpus_version()
    if not source_sha or source_sha != catalog_sha:
        raise ValueError("Vector index does not match the SQLite catalogue")
    import numpy as np
    import requests

    threshold = MIN_SCORE if min_score is None else min_score
    prompt = f"Instruct: {manifest['query_instruction']}\nQuery: {query}"
    url = os.getenv("EMBEDDING_URL", "http://127.0.0.1:8080/v1/embeddings")
    started = time.perf_counter()
    count("astronomy_embedding_requests_total")
    try:
        response = requests.post(
            url, json={"input": prompt, "model": manifest["model"], "encoding_format": "float"}, timeout=timeout,
        )
        response.raise_for_status()
        embedding = np.asarray(response.json()["data"][0]["embedding"], dtype="float32")
        if embedding.shape != (index.d,) or not np.isfinite(embedding).all() or not np.any(embedding):
            raise ValueError("Invalid query embedding for vector index")
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        count("astronomy_embedding_errors_total")
        raise
    finally:
        count("astronomy_embedding_request_seconds_total", time.perf_counter() - started)
    vector = embedding.reshape(1, index.d)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm  # the index stores L2-normalised vectors: scores are cosine similarities
    if limit <= 0 or index.ntotal == 0:
        return []

    # Hits in the other language must not starve matching-language candidates.
    # Expand only when necessary; the query embedding is requested once.
    pool = min(index.ntotal, max(limit * 4, 50))
    while True:
        scores, identifiers = index.search(vector, pool)
        result: list[dict] = []
        for score, identifier in zip(scores[0], identifiers[0]):
            if identifier < 0 or float(score) < threshold:
                continue
            chunk = chunks[mapping[int(identifier)]["chunk_id"]]
            if chunk["locale"] != locale:
                continue
            record_id = chunk["record_id"]
            section_id = chunk["section_id"]
            if chunk.get("media_id") and chunk.get("source_record_id"):
                record_id, section_id = chunk["source_record_id"], chunk["media_id"]
            result.append({"record_id": record_id, "section_id": section_id, "kind": chunk["kind"], "title": chunk["title"], "excerpt": chunk["text"][:400], "chunk_id": chunk["id"], "evidence_text": chunk["text"], "url": chunk["url"], "review_status": chunk["review_status"], "score": float(score)})
            if len(result) >= limit:
                return result
        if pool == index.ntotal or float(scores[0][-1]) < threshold:
            return result
        pool = min(index.ntotal, pool * 2)
