"""Run on a Runpod GPU Pod: build a reusable FAISS index from chunks.jsonl."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import requests

MODEL = "Qwen/Qwen3-Embedding-8B-GGUF"
MODEL_REVISION = "69d0e58a13e463cd99a9b83e3f5fee7c10265fab"
MODEL_FILE = "Qwen3-Embedding-8B-Q4_K_M.gguf"
QUERY_INSTRUCTION = "Retrieve relevant astronomy glossary definitions, explanations, and image captions for the user question."


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("index_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--embedding-url", default="http://127.0.0.1:8080/v1/embeddings")
    parser.add_argument("--model-file", type=Path, required=True)
    args = parser.parse_args()
    root = args.index_dir
    chunks_file = root / "chunks.jsonl"
    chunks = [json.loads(line) for line in chunks_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    source = json.loads((root / "chunk-manifest.json").read_text())
    if checksum(chunks_file) != source["chunks_sha256"] or len(chunks) != source["chunk_count"]:
        raise ValueError("Chunk manifest does not match chunks.jsonl")
    if not chunks:
        raise ValueError("No chunks to index")
    try:  # reuse the "<sha> <size> <mtime>" marker written after a full check of the 4.7 GB file
        from app.vector import model_checksum as verified_checksum

        model_checksum = verified_checksum(args.model_file)
    except ImportError:
        model_checksum = checksum(args.model_file)
    cache_path = root / f"embedding-cache-{model_checksum[:16]}.npz"
    cache = {}
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as stored:
            cache = {key: stored[key] for key in stored.files}
    keys = [row["content_hash"] for row in chunks]
    missing = [row for row in chunks if row["content_hash"] not in cache]
    for start in range(0, len(missing), args.batch_size):
        batch = missing[start : start + args.batch_size]
        response = requests.post(args.embedding_url, json={"input": [row["search_text"] for row in batch], "model": MODEL, "encoding_format": "float"}, timeout=180)
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
    mapping = [{"vector_id": i, "chunk_id": row["id"], "content_hash": row["content_hash"]} for i, row in enumerate(chunks)]
    index_temp = root / "index.faiss.tmp"
    map_temp = root / "vector-map.json.tmp"
    cache_temp = root / "embedding-cache.npz.tmp"
    faiss.write_index(index, str(index_temp))
    map_temp.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n")
    with cache_temp.open("wb") as target:
        np.savez_compressed(target, **{key: cache[key] for key in set(keys)})
    os.replace(index_temp, root / "index.faiss")
    os.replace(map_temp, root / "vector-map.json")
    os.replace(cache_temp, cache_path)
    manifest = {
        "schema_version": 1,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "model_file": MODEL_FILE,
        "model_sha256": model_checksum,
        "quantization": "Q4_K_M",
        "pooling": "last",
        "query_instruction": QUERY_INSTRUCTION,
        "dimension": int(matrix.shape[1]),
        "chunk_count": len(chunks),
        "chunk_config": source,
        "chunks_sha256": checksum(chunks_file),
        "index_sha256": checksum(root / "index.faiss"),
        "vector_map_sha256": checksum(root / "vector-map.json"),
        "embedding_cache_sha256": checksum(cache_path),
        "normalization": "L2",
        "metric": "inner_product",
    }
    temporary = root / "index-manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, root / "index-manifest.json")
    print(json.dumps({"chunks": len(chunks), "dimension": int(matrix.shape[1]), "model_revision": MODEL_REVISION}))


if __name__ == "__main__":
    main()
