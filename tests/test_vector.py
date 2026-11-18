"""Verify that the searcher uses the same model contract as the GPU index."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import index_files, vector


@unittest.skipUnless(importlib.util.find_spec("faiss"), "faiss-cpu is not installed")
class VectorContractTests(unittest.TestCase):
    def setUp(self):
        import faiss
        import numpy as np

        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        index_dir = self.root / "index"
        model_dir = self.root / "models"
        index_dir.mkdir()
        model_dir.mkdir()
        (model_dir / "small.gguf").write_bytes(b"test-model")
        chunks = [
            {"id": "term:1/definition/ru/0", "record_id": "term:1", "section_id": "definition", "locale": "ru", "kind": "term", "title": "Солнце", "text": "Звезда.", "url": "/api/v1/terms/1", "review_status": "draft"},
            {"id": "term:2/definition/ru/0", "record_id": "term:2", "section_id": "definition", "locale": "ru", "kind": "term", "title": "Луна", "text": "Спутник.", "url": "/api/v1/terms/2", "review_status": "draft"},
        ]
        (index_dir / "chunks.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chunks))
        (index_dir / "vector-map.json").write_text(json.dumps([{"chunk_id": row["id"]} for row in chunks]))
        index = faiss.IndexFlatIP(2)
        index.add(np.array([[1, 0], [0, 1]], dtype="float32"))
        faiss.write_index(index, str(index_dir / "index.faiss"))
        digest = lambda name: hashlib.sha256((index_dir / name).read_bytes()).hexdigest()
        (index_dir / "index-manifest.json").write_text(json.dumps({
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
        }))
        self.previous = vector.DATA, vector._loaded
        vector.DATA, vector._loaded = self.root, None

    def tearDown(self):
        vector.DATA, vector._loaded = self.previous
        self.temp.cleanup()

    def test_query_prompt_and_index_agree(self):
        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {"data": [{"embedding": [1.0, 0.0]}]}

        with patch("requests.post", return_value=Response()) as post:
            result = vector.search("Что такое Солнце?", "ru", 2)
        self.assertEqual(result[0]["record_id"], "term:1")
        self.assertEqual(post.call_args.kwargs["json"]["input"], "Instruct: Find astronomy definitions.\nQuery: Что такое Солнце?")

    def test_model_checksum_is_required(self):
        (self.root / "models/small.gguf").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "model checksum"):
            vector.load()


class VectorFailureCacheTests(unittest.TestCase):
    def test_broken_index_is_not_reverified_on_every_request(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "index").mkdir()
            (root / "index/index-manifest.json").write_text(json.dumps({"chunks_sha256": "x", "index_sha256": "x", "vector_map_sha256": "x"}))
            (root / "index/chunks.jsonl").write_text("")
            previous = vector.DATA, vector._loaded
            vector.DATA, vector._loaded = root, None
            try:
                with patch.object(index_files, "checksum", wraps=index_files.checksum) as checksum:
                    for _ in range(3):
                        with self.assertRaises(Exception):
                            vector.load()
                self.assertEqual(checksum.call_count, 1, "a failed load is cached instead of re-hashing files")
            finally:
                vector.DATA, vector._loaded = previous


if __name__ == "__main__":
    unittest.main()
