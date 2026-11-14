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
import numpy as np


def main(root: Path) -> None:
    manifest = json.loads((root / "index-manifest.json").read_text())
    mapping = json.loads((root / "vector-map.json").read_text())
    index = faiss.read_index(str(root / "index.faiss"))
    if index.ntotal != len(mapping):
        raise SystemExit("index and vector map disagree")
    cache_path = root / f"embedding-cache-{manifest['model_sha256'][:16]}.npz"
    cache: dict[str, np.ndarray] = {}
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as stored:
            cache = {key: stored[key] for key in stored.files}
    added = 0
    for row in mapping:
        if row["content_hash"] not in cache:
            cache[row["content_hash"]] = index.reconstruct(int(row["vector_id"])).astype("float32")
            added += 1
    temporary = cache_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as target:
        np.savez_compressed(target, **cache)
    temporary.replace(cache_path)
    print(json.dumps({"cache": cache_path.name, "vectors": len(cache), "added": added}))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/index"))
