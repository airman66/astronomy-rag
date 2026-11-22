"""The on-disk format shared by index generation and search."""

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path

from app.infra.files import checksum

MODEL = "Qwen/Qwen3-Embedding-8B-GGUF"
MODEL_REVISION = "69d0e58a13e463cd99a9b83e3f5fee7c10265fab"
MODEL_FILE = "Qwen3-Embedding-8B-Q4_K_M.gguf"
QUERY_INSTRUCTION = "Retrieve relevant astronomy glossary definitions, explanations, and image captions for the user question."
MANIFEST_NAME = "index-manifest.json"
CHUNK_MANIFEST_NAME = "chunk-manifest.json"
ARTIFACT_HASH_FIELDS = {
    "chunks.jsonl": "chunks_sha256",
    "index.faiss": "index_sha256",
    "vector-map.json": "vector_map_sha256",
}
INDEX_FILES = (MANIFEST_NAME, CHUNK_MANIFEST_NAME, *ARTIFACT_HASH_FIELDS)


@contextmanager
def index_writer(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".index-write.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def file_generation(path: Path) -> tuple[int, ...] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def artifact_generation(root: Path) -> tuple:
    return tuple(file_generation(root / name) for name in INDEX_FILES)


def read_manifest(root: Path) -> dict:
    return json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))


def read_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def check_chunk_manifest(root: Path, manifest: dict, *, required: bool = True) -> dict:
    try:
        chunks = json.loads((root / CHUNK_MANIFEST_NAME).read_text(encoding="utf-8"))
    except FileNotFoundError:
        if required:
            raise ValueError("Missing chunk manifest") from None
        return {}
    if (chunks["chunks_sha256"] != manifest["chunks_sha256"]
            or chunks["chunk_count"] != manifest["chunk_count"]
            or chunks.get("corpus_sha256") != manifest.get("corpus_sha256")
            or manifest.get("chunk_config", chunks) != chunks):
        raise ValueError("Chunk manifest does not match index manifest")
    if required and (not manifest.get("corpus_sha256") or "chunk_config" not in manifest):
        raise ValueError("Missing index corpus provenance")
    return chunks


def model_checksum(path: Path) -> str:
    """Hash the GGUF bytes rather than trusting a stale/user-writable marker.

    This runs when an index is loaded (in a background startup task), not per query.
    """
    return checksum(path)


def load_artifacts(root: Path, *, require_provenance: bool = True):
    import faiss
    import numpy as np

    generation = artifact_generation(root)
    manifest = read_manifest(root)
    for name, field in ARTIFACT_HASH_FIELDS.items():
        if checksum(root / name) != manifest[field]:
            raise ValueError(f"Index file checksum mismatch: {name}")
    config = check_chunk_manifest(root, manifest, required=require_provenance)
    if manifest.get("model") != MODEL or manifest.get("pooling") != "last":
        raise ValueError("Unsupported embedding model or pooling")
    if manifest.get("normalization", "L2") != "L2":
        raise ValueError("Index must use L2-normalized vectors")

    index = faiss.read_index(str(root / "index.faiss"))
    mapping = json.loads((root / "vector-map.json").read_text(encoding="utf-8"))
    rows = read_chunks(root / "chunks.jsonl")
    chunks = {row["id"]: row for row in rows}
    count = manifest["chunk_count"]
    if not (len(rows) == len(chunks) == len(mapping) == index.ntotal == count) or index.d != manifest["dimension"]:
        raise ValueError("Index dimension, count or chunk IDs mismatch")
    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("FAISS index must use inner-product metric")
    mapped_ids = set()
    for position, item in enumerate(mapping):
        chunk_id = item["chunk_id"]
        chunk = chunks.get(chunk_id)
        if chunk is None:
            raise ValueError(f"Vector {position} references an unknown chunk")
        if chunk_id in mapped_ids:
            raise ValueError(f"Vector {position} references a duplicate chunk ID: {chunk_id}")
        mapped_ids.add(chunk_id)
        if item.get("vector_id", position) != position:
            raise ValueError(f"Vector {position} has a non-sequential ID")
        if item.get("content_hash", chunk.get("content_hash")) != chunk.get("content_hash"):
            raise ValueError(f"Vector {position} hash disagrees with its chunk")
    for position in {0, count // 2, count - 1} if count else set():
        if not 0.99 <= float(np.linalg.norm(index.reconstruct(position))) <= 1.01:
            raise ValueError(f"Index vector {position} is not L2 normalized")
    if artifact_generation(root) != generation:
        raise ValueError("Index artifacts changed during loading")
    return index, mapping, chunks, manifest, config


def model_path(models: Path, manifest: dict) -> Path:
    filename = manifest.get("model_file")
    if not isinstance(filename, str) or not filename or Path(filename).name != filename or filename == "..":
        raise ValueError("Embedding model must be a local filename")
    return models / filename


def embedding_data(payload: object, model: str, filename: str) -> list[dict]:
    if not isinstance(payload, dict):
        raise ValueError("Invalid embedding server response")
    reported = payload.get("model")
    if not isinstance(reported, str) or (reported != model and Path(reported).name != filename):
        raise ValueError("Embedding server returned a different or missing model ID")
    return payload["data"]


def load_index(root: Path, models: Path):
    index, mapping, chunks, manifest, _ = load_artifacts(root)
    if model_checksum(model_path(models, manifest)) != manifest["model_sha256"]:
        raise ValueError("Embedding model checksum mismatch")
    return index, mapping, chunks, manifest
