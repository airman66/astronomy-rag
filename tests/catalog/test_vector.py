"""Verify that the searcher uses the same model contract as the GPU index."""

import importlib.util
import json
import unittest
from unittest.mock import Mock, patch

from app.catalog import index as index_files
from app.catalog import search, vector
from tests.catalog.fixtures import seed_vector_index
from tests.support.catalog import temporary_dir, use_vector_root

requires_faiss = unittest.skipUnless(importlib.util.find_spec("faiss"), "faiss-cpu is not installed")


class RankFusionTests(unittest.TestCase):
    def test_multiple_vector_chunks_do_not_outvote_an_agreement_between_searches(self):
        agreed = {"record_id": "term:1", "section_id": "definition"}
        repeated = {"record_id": "article:other", "section_id": "overview"}
        semantic = [
            {**repeated, "chunk_id": "first", "evidence_text": "First passage"},
            {**repeated, "chunk_id": "second", "evidence_text": "Second passage"},
            {**agreed, "chunk_id": "agreed", "evidence_text": "Definition"},
        ]
        with (
            patch.object(search.storage, "lexical_search", return_value=[agreed]),
            patch.object(vector, "search", return_value=semantic),
            patch.object(vector, "available", return_value=True),
        ):
            results, mode = search.search_with_mode("definition", "en")
        self.assertEqual(mode, "hybrid")
        self.assertEqual([row["chunk_id"] for row in results], ["agreed", "first"])


@requires_faiss
class VectorContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = temporary_dir(self)
        use_vector_root(self, self.root)
        seed_vector_index(self.root)

    def test_query_prompt_and_index_agree(self):
        response = Mock()
        response.json.return_value = {"data": [{"embedding": [1.0, 0.0]}]}
        with patch("requests.post", return_value=response) as post:
            result = vector.search("Что такое Солнце?", "ru", 2)
        self.assertEqual(result[0]["record_id"], "term:1")
        self.assertEqual(
            post.call_args.kwargs["json"]["input"],
            "Instruct: Find astronomy definitions.\nQuery: Что такое Солнце?",
        )

    def test_model_checksum_is_required(self):
        (self.root / "models/small.gguf").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "model checksum"):
            vector.load()


@requires_faiss
class VectorFailureCacheTests(unittest.TestCase):
    def test_broken_index_is_not_reverified_on_every_request(self):
        root = temporary_dir(self)
        use_vector_root(self, root)
        (root / "index").mkdir()
        (root / "index/index-manifest.json").write_text(
            json.dumps(
                {
                    "chunks_sha256": "x",
                    "index_sha256": "x",
                    "vector_map_sha256": "x",
                }
            )
        )
        (root / "index/chunks.jsonl").write_text("")
        with patch.object(index_files, "checksum", wraps=index_files.checksum) as checksum:
            for _ in range(3):
                with self.assertRaisesRegex(ValueError, "Index file checksum mismatch"):
                    vector.load()
        self.assertEqual(checksum.call_count, 1, "a failed load is cached instead of re-hashing files")
