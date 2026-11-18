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
    MANIFEST_NAME,
    MODEL,
    MODEL_FILE,
    MODEL_REVISION,
    QUERY_INSTRUCTION,
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
    chunks_file = root / "chunks.jsonl"
    chunks = read_chunks(chunks_file)
    source = json.loads((root / "chunk-manifest.json").read_text())
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
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        response = requests.post(
            embedding_url,
            json={"input": [row["search_text"] for row in batch], "model": MODEL, "encoding_format": "float"},
            timeout=180,
        )
        response.raise_for_status()
        encoded = sorted(response.json()["data"], key=lambda item: item["index"])
        if len(encoded) != len(batch):
            raise ValueError("Embedding server returned an unexpected vector count")
        vectors = [item["embedding"] for item in encoded]
        for row, vector in zip(batch, vectors):
            cache[row["content_hash"]] = np.asarray(vector, dtype="float32")
        print(f"embedded {min(start + len(batch), len(missing))}/{len(missing)}", flush=True)
    matrix = np.stack([cache[key] for key in keys]).astype("float32")
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
        index_temp.replace(root / index_temp.name)
        map_temp.replace(root / map_temp.name)
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
        "chunks_sha256": checksum(chunks_file),
        "index_sha256": checksum(root / "index.faiss"),
        "vector_map_sha256": checksum(root / "vector-map.json"),
        "embedding_cache_sha256": checksum(cache_file),
        "normalization": "L2",
        "metric": "inner_product",
    }
    write_json(root / MANIFEST_NAME, manifest)
    print(
        json.dumps(
            {"chunks": len(chunks), "dimension": int(matrix.shape[1]), "model_revision": MODEL_REVISION}
        )
    )
