"""Read and publish the cache shared by the embedding build and seed commands."""

import uuid
from pathlib import Path

import numpy as np


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
        with temporary.open("wb") as target:
            np.savez_compressed(target, **vectors)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
