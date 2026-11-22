"""Run on a Runpod GPU Pod: build a reusable FAISS index from chunks.jsonl."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import requests

from app.catalog.index import (
    CHUNK_MANIFEST_NAME,
    MANIFEST_NAME,
    MODEL,
    MODEL_FILE,
    MODEL_REVISION,
    QUERY_INSTRUCTION,
    index_writer,
    model_checksum,
    read_chunks,
)
from app.infra.files import checksum, write_json
from scripts.index.cache import cache_path, read_cache, save_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("index_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--embedding-url", default="http://127.0.0.1:8080/v1/embeddings")
    parser.add_argument("--model-file", type=Path, required=True)
    args = parser.parse_args()
    build(args.index_dir, args.model_file, args.embedding_url, args.batch_size)


def build(root: Path, model_file: Path, embedding_url: str, batch_size: int) -> None:
    if batch_size < 1:
        raise ValueError("Embedding batch size must be positive")
    with index_writer(root):
        _build(root, model_file, embedding_url, batch_size)


def _embedding_matrix(vectors: list) -> np.ndarray:
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim != 2 or not matrix.size:
        raise ValueError("Expected a matrix of embedding vectors")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Embedding vectors must have finite, nonzero norms")
    return matrix


def _build(root: Path, model_file: Path, embedding_url: str, batch_size: int) -> None:
    chunks_file = root / "chunks.jsonl"
    config_file = root / CHUNK_MANIFEST_NAME
    config_bytes = config_file.read_bytes()
    source = json.loads(config_bytes)
    if not source.get("corpus_sha256"):
        raise ValueError("Missing corpus provenance; rebuild chunks first")
    chunks = read_chunks(chunks_file)
    if checksum(chunks_file) != source["chunks_sha256"] or len(chunks) != source["chunk_count"]:
        raise ValueError("Chunk manifest does not match chunks.jsonl")
    if not chunks:
        raise ValueError("No chunks to index")
    model_sha = model_checksum(model_file)
    cache_file = cache_path(root, model_sha)
    cache = read_cache(cache_file)
    keys = [row["content_hash"] for row in chunks]
    missing_by_hash = {}
    for row in chunks:
        key = row["content_hash"]
        if key not in cache:
            missing_by_hash.setdefault(key, row)
    missing = list(missing_by_hash.values())
    checkpoint = 0
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        response = requests.post(
            embedding_url,
            json={"input": [row["search_text"] for row in batch], "model": MODEL, "encoding_format": "float"},
            timeout=180,
        )
        response.raise_for_status()
        encoded = sorted(response.json()["data"], key=lambda item: item["index"])
        if [item["index"] for item in encoded] != list(range(len(batch))):
            raise ValueError("Embedding server returned unexpected vector indices")
        vectors = _embedding_matrix([item["embedding"] for item in encoded])
        previous = next(iter(cache.values()), None)
        if previous is not None and previous.shape != vectors.shape[1:]:
            raise ValueError("Embedding server returned a different vector dimension")
        for row, vector in zip(batch, vectors):
            cache[row["content_hash"]] = vector
        if start + len(batch) - checkpoint >= 128:
            save_cache(cache_file, cache)
            checkpoint = start + len(batch)
        print(f"embedded {min(start + len(batch), len(missing))}/{len(missing)}", flush=True)
    matrix = _embedding_matrix([cache[key] for key in keys])
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    mapping = [
        {"vector_id": i, "chunk_id": row["id"], "content_hash": row["content_hash"]}
        for i, row in enumerate(chunks)
    ]
    with tempfile.TemporaryDirectory(dir=root) as directory:
        staging = Path(directory)
        index_temp = staging / "index.faiss"
        map_temp = staging / "vector-map.json"
        faiss.write_index(index, str(index_temp))
        write_json(map_temp, mapping)
        save_cache(cache_file, {key: cache[key] for key in set(keys)})
        manifest = {
            "schema_version": 1,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "model_file": MODEL_FILE,
            "model_sha256": model_sha,
            "quantization": "Q4_K_M",
            "pooling": "last",
            "query_instruction": QUERY_INSTRUCTION,
            "dimension": int(matrix.shape[1]),
            "chunk_count": len(chunks),
            "chunk_config": source,
            "corpus_sha256": source["corpus_sha256"],
            "chunks_sha256": source["chunks_sha256"],
            "index_sha256": checksum(index_temp),
            "vector_map_sha256": checksum(map_temp),
            "embedding_cache_sha256": checksum(cache_file),
            "normalization": "L2",
            "metric": "inner_product",
        }
        write_json(staging / MANIFEST_NAME, manifest)
        if config_file.read_bytes() != config_bytes or checksum(chunks_file) != source["chunks_sha256"]:
            raise ValueError("Chunks changed during index build")
        for name in (index_temp.name, map_temp.name, MANIFEST_NAME):
            (staging / name).replace(root / name)
    print(
        json.dumps(
            {"chunks": len(chunks), "dimension": int(matrix.shape[1]), "model_revision": MODEL_REVISION}
        )
    )
