"""Read and publish the cache shared by the embedding build and seed commands."""

import argparse
import json
import uuid
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import faiss
import numpy as np

from app.catalog.index import read_manifest


def cache_path(root: Path, model_sha256: str) -> Path:
    return root / f"embedding-cache-{model_sha256[:16]}.npz"


def read_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def save_cache(path: Path, vectors: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
            for key, vector in vectors.items():
                with archive.open(f"{key}.npy", "w", force_zip64=True) as entry:
                    np.save(entry, vector, allow_pickle=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def seed(root: Path) -> None:
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


def seed_main() -> None:
    parser = argparse.ArgumentParser(description="Seed the embedding cache from the published FAISS index.")
    parser.add_argument("index_dir", type=Path, nargs="?", default=Path("data/index"))
    seed(parser.parse_args().index_dir)
