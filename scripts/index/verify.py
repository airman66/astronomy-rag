"""Verify shipped FAISS artefacts without downloading the large GGUF model.

This catches a stale or partial index push, corrupted vector mapping and
incorrect manifest versions; it does not assess semantic answer quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np

from app.catalog.corpus import corpus_digest
from app.catalog.index import MODEL, read_chunks, read_manifest
from app.infra.files import checksum
from app.pipeline.chunks import CHUNK_VERSION


def verify_index(root: Path, *, require_current_chunking: bool = False) -> dict[str, int]:
    manifest = read_manifest(root)
    if manifest.get("model") != MODEL or manifest.get("pooling") != "last":
        raise ValueError("Unexpected embedding model or pooling in index manifest")
    if manifest.get("normalization", "L2") != "L2":
        raise ValueError("Index must use L2-normalized vectors")
    for filename, field in (
        ("chunks.jsonl", "chunks_sha256"),
        ("vector-map.json", "vector_map_sha256"),
        ("index.faiss", "index_sha256"),
    ):
        if checksum(root / filename) != manifest[field]:
            raise ValueError(f"Index file checksum mismatch: {filename}")
    chunk_manifest_path = root / "chunk-manifest.json"
    if chunk_manifest_path.exists():
        chunk_manifest = json.loads(chunk_manifest_path.read_text())
        if (chunk_manifest["chunks_sha256"] != manifest["chunks_sha256"]
                or chunk_manifest["chunk_count"] != manifest["chunk_count"]):
            raise ValueError("Chunk manifest does not match index manifest")
        if require_current_chunking and (
            chunk_manifest.get("chunk_version") != CHUNK_VERSION
            or manifest.get("chunk_config") != chunk_manifest
        ):
            raise ValueError(f"Rebuild the index with chunking version {CHUNK_VERSION}")
    elif require_current_chunking:
        raise ValueError("Missing chunk manifest")
    if require_current_chunking and manifest.get("corpus_sha256") != corpus_digest(root.parent):
        raise ValueError("Index does not match the canonical corpus")
    rows = read_chunks(root / "chunks.jsonl")
    row_by_id = {row["id"]: row for row in rows}
    mapping = json.loads((root / "vector-map.json").read_text())
    index = faiss.read_index(str(root / "index.faiss"))
    count = manifest["chunk_count"]
    if (len(rows) != count or len(row_by_id) != count
            or len(mapping) != count or index.ntotal != count or index.d != manifest["dimension"]):
        raise ValueError("Index dimension, count or chunk IDs mismatch")
    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("FAISS index must use inner-product metric")
    mapped_ids: set[str] = set()
    for position, item in enumerate(mapping):
        chunk_id = item.get("chunk_id")
        chunk = row_by_id.get(chunk_id)
        if chunk is None:
            raise ValueError(f"Vector {position} references an unknown chunk")
        if chunk_id in mapped_ids:
            raise ValueError(f"Vector {position} references a duplicate chunk ID: {chunk_id}")
        mapped_ids.add(chunk_id)
        if "vector_id" in item and item["vector_id"] != position:
            raise ValueError(f"Vector {position} has a non-sequential ID")
        if "content_hash" in item and item["content_hash"] != chunk.get("content_hash"):
            raise ValueError(f"Vector {position} hash disagrees with its chunk")
    for position in {0, count // 2, count - 1} if count else set():
        norm = float(np.linalg.norm(index.reconstruct(position)))
        if not 0.99 <= norm <= 1.01:
            raise ValueError(f"Index vector {position} is not L2 normalized")
    return {"chunks": count, "dimensions": index.d, "vectors": index.ntotal}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_dir", type=Path)
    parser.add_argument("--require-current-chunking", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify_index(args.index_dir, require_current_chunking=args.require_current_chunking)))


if __name__ == "__main__":
    main()
