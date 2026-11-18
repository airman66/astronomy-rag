"""A two-document FAISS index for testing vector search without an embedding server."""

import hashlib
import json
from pathlib import Path


def seed_vector_index(root: Path) -> None:
    import faiss
    import numpy as np

    index_dir = root / "index"
    model_dir = root / "models"
    index_dir.mkdir()
    model_dir.mkdir()
    (model_dir / "small.gguf").write_bytes(b"test-model")
    chunks = [
        {
            "id": "term:1/definition/ru/0",
            "record_id": "term:1",
            "section_id": "definition",
            "locale": "ru",
            "kind": "term",
            "title": "Солнце",
            "text": "Звезда.",
            "url": "/api/v1/terms/1",
            "review_status": "draft",
        },
        {
            "id": "term:2/definition/ru/0",
            "record_id": "term:2",
            "section_id": "definition",
            "locale": "ru",
            "kind": "term",
            "title": "Луна",
            "text": "Спутник.",
            "url": "/api/v1/terms/2",
            "review_status": "draft",
        },
    ]
    (index_dir / "chunks.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chunks)
    )
    (index_dir / "vector-map.json").write_text(json.dumps([{"chunk_id": row["id"]} for row in chunks]))
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1, 0], [0, 1]], dtype="float32"))
    faiss.write_index(index, str(index_dir / "index.faiss"))

    def digest(name: str) -> str:
        return hashlib.sha256((index_dir / name).read_bytes()).hexdigest()

    (index_dir / "index-manifest.json").write_text(
        json.dumps(
            {
                "model": "Qwen/Qwen3-Embedding-8B-GGUF",
                "model_file": "small.gguf",
                "model_sha256": hashlib.sha256(b"test-model").hexdigest(),
                "pooling": "last",
                "query_instruction": "Find astronomy definitions.",
                "dimension": 2,
                "chunk_count": 2,
                "chunks_sha256": digest("chunks.jsonl"),
                "index_sha256": digest("index.faiss"),
                "vector_map_sha256": digest("vector-map.json"),
            }
        )
    )
