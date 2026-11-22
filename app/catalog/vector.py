"""CPU search over a verified Runpod/GPU-built FAISS index.

The index is optional for the service: when the manifest, model file or embedding server is
missing, callers fall back to lexical search. Load failures are cached briefly so a broken
index is not re-verified (4.7 GB model hash) on every request.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import NamedTuple

from app.catalog import storage
from app.catalog.index import MANIFEST_NAME, artifact_generation, embedding_data, file_generation, load_index, model_path, read_manifest
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
    generation: tuple
    artifacts: tuple


_loaded: _LoadedIndex | None = None
_failure: tuple[tuple, float, str] | None = None
log = logging.getLogger("glossary.vector")


def loaded_generation() -> tuple | None:
    loaded = _loaded
    return loaded.generation if loaded is not None else None


def _generation(root: Path, models: Path) -> tuple:
    stamps = artifact_generation(root)
    try:
        path = model_path(models, read_manifest(root))
        model_stamp = file_generation(path)
    except (OSError, ValueError, AttributeError):
        model_stamp = None
    return root.resolve(), stamps, model_stamp


def load():
    global _loaded, _failure
    root = DATA / "index"
    models = DATA / "models"
    if not _lock.acquire(blocking=False):
        current = _loaded
        return current.artifacts if current is not None else None
    try:
        current = _loaded
        previous = current.artifacts if current is not None else None
        if not (root / MANIFEST_NAME).is_file():
            return previous
        generation = _generation(root, models)
        if current is not None and current.generation == generation:
            return previous
        failed = _failure
        if failed is not None and failed[0] == generation and time.monotonic() - failed[1] < RETRY_AFTER_FAILURE:
            if previous is not None:
                return previous
            raise ValueError(failed[2])
        try:
            loaded = load_index(root, models)
            if generation != _generation(root, models):
                raise ValueError("Index artifacts or model changed during loading")
        except Exception as exc:
            _failure = (generation, time.monotonic(), str(exc))
            if previous is not None:
                log.warning("new vector index rejected, keeping the loaded one: %s", exc)
                return previous
            log.warning("vector index unavailable: %s", exc)
            raise
        _failure = None
        _loaded = _LoadedIndex(generation, loaded)
        log.info("vector index loaded: %s chunks", loaded[0].ntotal)
        return loaded
    finally:
        _lock.release()


def search(
    query: str, locale: str, limit: int = 20, min_score: float | None = None,
    *, timeout: float = QUERY_TIMEOUT, corpus_sha: str | None = None, allow_index_load: bool = True,
) -> list[dict] | None:
    """Return None when no index exists and an empty list after a successful query with no hits."""
    current = _loaded
    loaded = load() if allow_index_load else current.artifacts if current is not None else None
    if loaded is None:
        return None
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
        rows = embedding_data(response.json(), manifest["model"], manifest["model_file"])
        embedding = np.asarray(rows[0]["embedding"], dtype="float32")
        if embedding.shape != (index.d,) or not np.isfinite(embedding).all():
            raise ValueError("Invalid query embedding for vector index")
        norm = float(np.linalg.norm(embedding))
        if not 0 < norm < float("inf"):
            raise ValueError("Invalid query embedding norm for vector index")
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        count("astronomy_embedding_errors_total")
        raise
    finally:
        count("astronomy_embedding_request_seconds_total", time.perf_counter() - started)
    vector = embedding.reshape(1, index.d) / norm
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
