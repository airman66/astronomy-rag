"""The on-disk format shared by index generation and search."""

import json
from pathlib import Path

from app.infra.files import checksum, write_bytes

MODEL = "Qwen/Qwen3-Embedding-8B-GGUF"
MODEL_REVISION = "69d0e58a13e463cd99a9b83e3f5fee7c10265fab"
MODEL_FILE = "Qwen3-Embedding-8B-Q4_K_M.gguf"
QUERY_INSTRUCTION = "Retrieve relevant astronomy glossary definitions, explanations, and image captions for the user question."
MANIFEST_NAME = "index-manifest.json"


def read_manifest(root: Path) -> dict:
    return json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))


def read_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def model_checksum(path: Path) -> str:
    """SHA-256 of the GGUF; reuses "<sha> <size> <mtime>" written after an earlier full check."""
    marker = path.with_name(path.name + ".verified")
    stat = path.stat()
    if marker.exists():
        parts = marker.read_text(encoding="utf-8").split()
        if len(parts) == 3 and parts[1] == str(stat.st_size) and parts[2] == str(int(stat.st_mtime)):
            return parts[0]
    digest = checksum(path)
    try:
        write_bytes(marker, f"{digest} {stat.st_size} {int(stat.st_mtime)}\n".encode())
    except OSError:
        pass
    return digest


def load_index(root: Path, models: Path):
    import faiss

    manifest = read_manifest(root)
    for name, expected in (
        ("chunks.jsonl", manifest["chunks_sha256"]),
        ("index.faiss", manifest["index_sha256"]),
        ("vector-map.json", manifest["vector_map_sha256"]),
    ):
        if checksum(root / name) != expected:
            raise ValueError(f"Index file checksum mismatch: {name}")
    index = faiss.read_index(str(root / "index.faiss"))
    mapping = json.loads((root / "vector-map.json").read_text())
    chunks = {row["id"]: row for row in read_chunks(root / "chunks.jsonl")}
    if (
        index.ntotal != manifest["chunk_count"]
        or len(mapping) != index.ntotal
        or index.d != manifest["dimension"]
    ):
        raise ValueError("Index manifest count/dimension mismatch")
    model_path = models / manifest["model_file"]
    if model_checksum(model_path) != manifest["model_sha256"]:
        raise ValueError("Embedding model checksum mismatch")
    if manifest["model"] != MODEL or manifest["pooling"] != "last":
        raise ValueError("Unsupported embedding model or pooling")
    return index, mapping, chunks, manifest
