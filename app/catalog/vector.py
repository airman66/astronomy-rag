"""CPU search over a verified Runpod/GPU-built FAISS index.

The index is optional for the service: when the manifest, model file or embedding server is
missing, callers fall back to lexical search. Load failures are cached briefly so a broken
index is not re-verified (4.7 GB model hash) on every request.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from threading import Lock

from app.catalog.index import MANIFEST_NAME, load_index
from app.infra.config import data_directory

DATA = data_directory()
QUERY_TIMEOUT = float(os.getenv("EMBEDDING_TIMEOUT") or "15")
# Cosine similarity below which a neighbour is not a match. Calibrated on Qwen3-Embedding-8B: relevant
# Russian hits 0.58–0.83, off-topic queries up to 0.46, nonsense up to 0.54 (scripts/calibrate, 2026-11-14).
MIN_SCORE = float(os.getenv("SEMANTIC_MIN_SCORE") or "0.55")
RETRY_AFTER_FAILURE = 60.0
_lock = Lock()
_loaded = None
_failure: dict[str, tuple[float, Exception]] = {}
log = logging.getLogger("glossary.vector")


def _failure_key() -> str:
    manifest = DATA / "index" / MANIFEST_NAME
    try:
        stamp = manifest.stat().st_mtime_ns
    except OSError:
        stamp = 0
    return f"{DATA.resolve()}|{stamp}"


def available() -> bool:
    return _loaded is not None


_stamp: dict[str, float | str] = {"key": "", "checked": 0.0}


def _changed_on_disk() -> bool:
    """A republished index (new manifest) replaces the loaded one; checked at most every 30 s."""
    if time.monotonic() - float(_stamp["checked"]) < 30:
        return False
    _stamp["checked"] = time.monotonic()
    return _failure_key() != _stamp["key"]


def load():
    global _loaded
    if _loaded is not None and not _changed_on_disk():
        return _loaded
    root = DATA / "index"
    if not (root / MANIFEST_NAME).exists():
        return _loaded
    key = _failure_key()
    failed = _failure.get(key)
    if failed and time.monotonic() - failed[0] < RETRY_AFTER_FAILURE:
        if _loaded is not None:
            return _loaded  # keep serving the previous index while the new one is inconsistent
        raise failed[1]
    with _lock:
        if _loaded is not None and _stamp["key"] == key:
            return _loaded
        try:
            loaded = load_index(root, DATA / "models")
        except Exception as exc:
            _failure[key] = (time.monotonic(), exc)
            if _loaded is not None:
                log.warning("new vector index rejected, keeping the loaded one: %s", exc)
                return _loaded
            raise
        _failure.pop(key, None)
        _loaded = loaded
        _stamp.update(key=key, checked=time.monotonic())
        log.info("vector index loaded: %s chunks", _loaded[0].ntotal)
        return _loaded


def warm_up() -> None:
    """Verify and load the index in a background thread so requests never wait for it."""

    def run() -> None:
        try:
            load()
        except Exception as exc:
            log.warning("vector index not loaded, lexical search only: %s: %s", type(exc).__name__, exc)

    threading.Thread(target=run, name="vector-warm-up", daemon=True).start()


def search(query: str, locale: str, limit: int = 20, min_score: float | None = None) -> list[dict]:
    loaded = load()
    if loaded is None:
        return []
    index, mapping, chunks, manifest = loaded
    import numpy as np
    import requests

    threshold = MIN_SCORE if min_score is None else min_score
    prompt = f"Instruct: {manifest['query_instruction']}\nQuery: {query}"
    url = os.getenv("EMBEDDING_URL", "http://127.0.0.1:8080/v1/embeddings")
    response = requests.post(url, json={"input": prompt, "model": manifest["model"], "encoding_format": "float"}, timeout=QUERY_TIMEOUT)
    response.raise_for_status()
    embedding = np.asarray(response.json()["data"][0]["embedding"], dtype="float32")
    if embedding.shape != (index.d,):
        raise ValueError("Query embedding dimension differs from index")
    vector = embedding.reshape(1, index.d)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm  # the index stores L2-normalised vectors: scores are cosine similarities
    scores, identifiers = index.search(vector, min(index.ntotal, max(limit * 4, 50)))
    result = []
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
        result.append({"record_id": record_id, "section_id": section_id, "kind": chunk["kind"], "title": chunk["title"], "excerpt": chunk["text"][:400], "url": chunk["url"], "review_status": chunk["review_status"], "score": float(score)})
        if len(result) >= limit:
            break
    return result
