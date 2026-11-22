"""The committed vector mapping must agree with all shipped index files."""

import hashlib
import json
import unittest

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
        }
        manifest["chunk_config"] = config
        manifest_path.write_text(json.dumps(manifest))
        (self.index_dir / "chunk-manifest.json").write_text(json.dumps(config))

        self.assertEqual(verify_index(self.index_dir)["chunks"], 2)
        with self.assertRaisesRegex(ValueError, "Rebuild the index with chunking version"):
            verify_index(self.index_dir, require_current_chunking=True)
