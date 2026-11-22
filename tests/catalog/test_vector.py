"""Verify that the searcher uses the same model contract as the GPU index."""

import importlib.util
import json
import unittest
from unittest.mock import Mock, patch

from app.catalog import index as index_files
from app.catalog import search, storage, vector
from app.catalog.corpus import corpus_digest
from tests.catalog.fixtures import bind_vector_index, seed_vector_index
from tests.support.catalog import make_term, temporary_dir, use_catalog, use_vector_root, write_term

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
            patch.object(search.storage, "lexical_search_snapshot", return_value=([agreed], "test-corpus")),
            patch.object(vector, "search", return_value=semantic),
        ):
            results, mode = search.search_with_mode("definition", "en")
        self.assertEqual(mode, "hybrid")
        self.assertEqual([row["chunk_id"] for row in results], ["agreed", "first"])


@requires_faiss
class CorpusSnapshotTests(unittest.TestCase):
    def test_sqlite_replacement_during_search_never_fuses_different_corpora(self):
        root = use_catalog(self)
        seed_vector_index(root)
        write_term(root, make_term(1, "Sun", "Солнце", "Moon.", "Луна."))
        storage.build_database(root, storage.DB)
        (root / "processed/terms/1.json").unlink()
        write_term(root, make_term(2, "Moon", "Луна", "Satellite.", "Спутник."))
        replacement = root / "replacement.db"
        storage.build_database(root, replacement)
        bind_vector_index(root, corpus_digest(root))
        original_load = vector.load
        response = Mock()
        response.json.return_value = {"data": [{"embedding": [0.0, 1.0]}]}

        def load_after_replacement():
            replacement.replace(storage.DB)
            return original_load()

        with (
            patch.object(storage, "READ_ONLY", True),
            patch("requests.post", return_value=response) as post,
        ):
            with patch.object(vector, "load", side_effect=load_after_replacement):
                hits, mode = search.search_with_mode("Луна", "ru")
            self.assertEqual(mode, "lexical")
            self.assertEqual({hit["record_id"] for hit in hits}, {"term:1"})
            post.assert_not_called()

            hits, mode = search.search_with_mode("Луна", "ru")
            self.assertEqual(mode, "hybrid")
            self.assertEqual({hit["record_id"] for hit in hits}, {"term:2"})

    def test_page_cache_drops_old_results_after_index_reload(self):
        root = use_catalog(self)
        seed_vector_index(root)
        write_term(root, make_term(1, "Sun", "Солнце", "A star.", "Звезда."))
        write_term(root, make_term(2, "Moon", "Луна", "Satellite.", "Спутник."))
        storage.ensure_database()
        manifest_path = root / "index/index-manifest.json"
        bind_vector_index(root, corpus_digest(root))
        manifest = json.loads(manifest_path.read_text())
        vector.load()

        manifest["query_instruction"] = "Updated query instruction."
        manifest_path.write_text(json.dumps(manifest))
        old = Mock()
        old.json.return_value = {"data": [{"embedding": [1.0, 0.0]}]}
        new = Mock()
        new.json.return_value = {"data": [{"embedding": [0.0, 1.0]}]}

        with patch("requests.post", side_effect=[old, new]) as post:
            hits, mode = search.paged_search("объект", "ru", 200)
            self.assertEqual((hits[0]["record_id"], mode), ("term:1", "hybrid"))
            vector.load()

            for _ in range(2):
                hits, mode = search.paged_search("объект", "ru", 200)
                self.assertEqual((hits[0]["record_id"], mode), ("term:2", "hybrid"))
            self.assertEqual(post.call_count, 2)


@requires_faiss
class VectorContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = temporary_dir(self)
        use_vector_root(self, self.root)
        seed_vector_index(self.root)

    def test_query_prompt_and_index_agree(self):
        response = Mock()
        response.json.return_value = {"data": [{"embedding": [1.0, 0.0]}]}
        with (
            patch.object(vector.storage, "corpus_version", return_value="test-corpus"),
            patch("requests.post", return_value=response) as post,
        ):
            result = vector.search("Что такое Солнце?", "ru", 2)
        assert result is not None
        self.assertEqual(result[0]["record_id"], "term:1")
        self.assertEqual(
            post.call_args.kwargs["json"]["input"],
            "Instruct: Find astronomy definitions.\nQuery: Что такое Солнце?",
        )

    def test_model_checksum_is_required(self):
        (self.root / "models/small.gguf").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "model checksum"):
            vector.load()

    def test_failed_generation_is_retried_when_index_file_is_repaired(self):
        path = self.root / "index/vector-map.json"
        intact = path.read_bytes()
        path.write_bytes(b"[]")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            vector.load()
        path.write_bytes(intact)
        self.assertIsNotNone(vector.load())

    def test_model_arrival_invalidates_cached_missing_file_failure(self):
        path = self.root / "models/small.gguf"
        content = path.read_bytes()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            vector.load()
        path.write_bytes(content)
        self.assertIsNotNone(vector.load())

    def test_file_replaced_during_load_is_not_admitted(self):
        path = self.root / "index/vector-map.json"
        original_read = index_files.read_chunks

        def replace_mapping(chunks):
            replacement = path.with_suffix(".tmp")
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
            return original_read(chunks)

        with patch.object(index_files, "read_chunks", side_effect=replace_mapping):
            with self.assertRaisesRegex(ValueError, "artifacts changed during loading"):
                vector.load()
        self.assertIsNone(vector.loaded_generation())
        self.assertIsNotNone(vector.load())


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
        manifest = {"query_instruction": "Find astronomy definitions.", "model": "test", "corpus_sha256": "test-corpus"}
        with (
            patch.object(vector, "load", return_value=(index, mapping, chunks, manifest)),
            patch.object(vector.storage, "corpus_version", return_value="test-corpus"),
            patch("requests.post", return_value=response),
        ):
            hits = vector.search("звезда", "ru", limit=1)
        assert hits is not None
        self.assertEqual(index.pools, [50, 52])
        self.assertEqual(hits[0]["record_id"], "term:51")
