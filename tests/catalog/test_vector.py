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

class ModelChecksumTests(unittest.TestCase):
    def test_stale_verified_marker_does_not_bypass_hash(self):
        import os
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "small.gguf"
            model.write_bytes(b"original")
            stat = model.stat()
            marker = model.with_name(model.name + ".verified")
            marker.write_text(f"fakehash {stat.st_size} {int(stat.st_mtime)}\\n")
            model.write_bytes(b"replaced")
            os.utime(model, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            self.assertEqual(index_files.model_checksum(model), index_files.checksum(model))
            self.assertNotEqual(index_files.model_checksum(model), "fakehash")


@requires_faiss
class VectorLocaleCoverageTests(unittest.TestCase):
    def test_mixed_language_shortlist_is_expanded(self):
        import numpy as np

        class FakeIndex:
            d = 2
            ntotal = 52

            def __init__(self):
                self.pools = []

            def search(self, embedding, pool):
                self.pools.append(pool)
                return (
                    np.array([[0.9] * pool], dtype="float32"),
                    np.array([list(range(pool))], dtype="int64"),
                )

        index = FakeIndex()
        mapping = [{"chunk_id": str(i)} for i in range(index.ntotal)]
        chunks = {
            str(i): {
                "id": str(i), "locale": "ru" if i == 51 else "en",
                "record_id": f"term:{i}", "section_id": "definition",
                "kind": "term", "title": "Star", "text": "A star",
                "url": f"/api/v1/terms/{i}", "review_status": "draft",
            }
            for i in range(index.ntotal)
        }
        response = Mock()
        response.json.return_value = {"data": [{"embedding": [1.0, 0.0]}]}
        manifest = {"query_instruction": "Find astronomy definitions.", "model": "test"}
        with (
            patch.object(vector, "load", return_value=(index, mapping, chunks, manifest)),
            patch("requests.post", return_value=response),
        ):
            hits = vector.search("звезда", "ru", limit=1)
        self.assertEqual(index.pools, [50, 52])
        self.assertEqual(hits[0]["record_id"], "term:51")
