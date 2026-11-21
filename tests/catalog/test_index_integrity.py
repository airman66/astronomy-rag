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

    def test_bad_reference_rejected_even_with_updated_checksum(self):
        path = self.index_dir / "vector-map.json"
        rows = json.loads(path.read_text())
        rows[0]["chunk_id"] = "missing"
        path.write_text(json.dumps(rows))
        manifest_path = self.index_dir / "index-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["vector_map_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "unknown chunk"):
            verify_index(self.index_dir)
