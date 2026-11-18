"""Seed build_index.py's embedding cache from the published FAISS index.

The cache (embedding-cache-<model sha[:16]>.npz, not in Git) maps chunk content hashes to vectors.
After a translation change only chunks with a new content hash need the embedding model; the
vectors of unchanged chunks are recovered from index.faiss (IndexFlatIP stores them exactly;
they are already L2-normalised, and normalising again is a no-op).

  python -m scripts.seed_embedding_cache data/index
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss

# Preserve direct script invocation from a checkout.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.index_files import read_manifest
from scripts.embedding_cache import cache_path, read_cache, save_cache


def main(root: Path) -> None:
    manifest = read_manifest(root)
    mapping = json.loads((root / "vector-map.json").read_text())
    index = faiss.read_index(str(root / "index.faiss"))
    if index.ntotal != len(mapping):
        raise SystemExit("index and vector map disagree")
    cache_file = cache_path(root, manifest["model_sha256"])
    cache = read_cache(cache_file)
    added = 0
    for row in mapping:
        if row["content_hash"] not in cache:
            cache[row["content_hash"]] = index.reconstruct(int(row["vector_id"])).astype("float32")
            added += 1
    save_cache(cache_file, cache)
    print(json.dumps({"cache": cache_file.name, "vectors": len(cache), "added": added}))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/index"))
