"""The committed vector mapping must agree with all shipped index files."""

import hashlib
import json
import unittest
from unittest.mock import Mock, patch

from app.catalog.index import MODEL, read_chunks
from app.infra.files import checksum
from scripts.index.build import build
from scripts.index.cache import seed
from scripts.index.verify import verify_index
from tests.catalog.fixtures import seed_vector_index
from tests.support.catalog import temporary_dir


class IndexIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.root = temporary_dir(self)
        seed_vector_index(self.root)
        self.index_dir = self.root / "index"

    def test_valid_small_index(self):
        self.assertEqual(verify_index(self.index_dir)["chunks"], 2)

    def test_checksum_mismatch_detected(self):
        with (self.index_dir / "vector-map.json").open("a") as out:
            out.write("\\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verify_index(self.index_dir)

    def _replace_first_chunk_id(self, chunk_id: str):
        path = self.index_dir / "vector-map.json"
        rows = json.loads(path.read_text())
        rows[0]["chunk_id"] = chunk_id
        path.write_text(json.dumps(rows))
        manifest_path = self.index_dir / "index-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["vector_map_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))

    def test_bad_reference_rejected_even_with_updated_checksum(self):
        self._replace_first_chunk_id("missing")
        with self.assertRaisesRegex(ValueError, "unknown chunk"):
            verify_index(self.index_dir)

    def test_duplicate_reference_rejected_even_with_updated_checksum(self):
        self._replace_first_chunk_id("term:2/definition/ru/0")
        with self.assertRaisesRegex(ValueError, "duplicate chunk ID"):
            verify_index(self.index_dir)

    def test_current_chunking_check_rejects_a_consistent_older_index(self):
        manifest_path = self.index_dir / "index-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        config = {
            "chunk_version": "2",
            "chunk_count": manifest["chunk_count"],
            "chunks_sha256": manifest["chunks_sha256"],
            "corpus_sha256": manifest["corpus_sha256"],
        }
        manifest["chunk_config"] = config
        manifest_path.write_text(json.dumps(manifest))
        (self.index_dir / "chunk-manifest.json").write_text(json.dumps(config))

        self.assertEqual(verify_index(self.index_dir)["chunks"], 2)
        with self.assertRaisesRegex(ValueError, "Rebuild the index with chunking version"):
            verify_index(self.index_dir, require_current_chunking=True)

    def test_cache_seed_rejects_changed_mapping_before_reading_vectors(self):
        (self.index_dir / "vector-map.json").write_text("[]")
        with self.assertRaisesRegex(ValueError, "Index file checksum mismatch"):
            seed(self.index_dir)
        self.assertEqual(list(self.index_dir.glob("embedding-cache-*.npz")), [])


class IndexBuildTests(unittest.TestCase):
    def setUp(self):
        self.root = temporary_dir(self)
        seed_vector_index(self.root)
        self.index_dir = self.root / "index"
        self.model = self.root / "models/small.gguf"
        self.chunks = self.index_dir / "chunks.jsonl"
        self.config = self.index_dir / "chunk-manifest.json"
        rows = read_chunks(self.chunks)
        for row in rows:
            row["search_text"] = row["text"]
            row["content_hash"] = row["id"].replace("/", "-")
        self.chunks.write_text("".join(json.dumps(row) + "\n" for row in rows))
        config = json.loads(self.config.read_text())
        config["chunks_sha256"] = checksum(self.chunks)
        self.config.write_text(json.dumps(config))
        self.published = {
            self.index_dir / name: checksum(self.index_dir / name)
            for name in ("index-manifest.json", "index.faiss", "vector-map.json")
        }

    def assert_publication_unchanged(self):
        self.assertEqual({path: checksum(path) for path in self.published}, self.published)

    def test_chunk_replacement_during_embedding_preserves_previous_index(self):
        response = Mock()
        response.json.return_value = {"model": MODEL, "data": [
            {"index": 0, "embedding": [1.0, 1.0]}, {"index": 1, "embedding": [1.0, 0.0]},
        ]}

        def change_corpus(*args, **kwargs):
            config = json.loads(self.config.read_text())
            config["corpus_sha256"] = "different"
            self.config.write_text(json.dumps(config))
            return response

        with patch("requests.post", side_effect=change_corpus):
            with self.assertRaisesRegex(ValueError, "Chunks changed during index build"):
                build(self.index_dir, self.model, "http://embeddings.test", 2)
        self.assert_publication_unchanged()

    def test_bad_embedding_batches_do_not_write_artifacts_or_pollute_cache(self):
        for model, first, index, message in (
            (MODEL, [1.0, 1.0], 0, "unexpected vector indices"),
            (MODEL, [0.0, 0.0], 1, "finite, nonzero norms"),
            ("different-model", [1.0, 1.0], 1, "different or missing model ID"),
        ):
            response = Mock()
            response.json.return_value = {"model": model, "data": [
                {"index": 0, "embedding": first}, {"index": index, "embedding": [1.0, 0.0]},
            ]}
            with self.subTest(message=message), patch("requests.post", return_value=response):
                with self.assertRaisesRegex(ValueError, message):
                    build(self.index_dir, self.model, "http://embeddings.test", 2)
                self.assert_publication_unchanged()
                self.assertEqual(list(self.index_dir.glob("embedding-cache-*.npz")), [])
