"""Media deployment report distinguishes missing files from corrupted files."""

import hashlib
import unittest
from pathlib import Path

from scripts.media.check_paths import inspect
from tests.support.catalog import make_term, temporary_dir, write_term


class DeploymentMediaTests(unittest.TestCase):
    def setUp(self):
        self.root = temporary_dir(self)
        term = make_term(1, "Star", "Звезда", "Hot", "Горячая")
        term["media"] = [{
            "id": "media:example", "original_path": "data/media/original/example.png",
            "sha256": hashlib.sha256(b"original").hexdigest(), "ru_path": None,
        }]
        write_term(self.root, term)
        self.target = self.root / "media/original/example.png"

    def test_missing_original_is_reported(self):
        result = inspect(self.root)
        self.assertEqual(result["missing_count"], 1)
        self.assertEqual(result["present_files"], 0)
        self.assertEqual(result["missing"][0]["media_id"], "media:example")

    def test_present_and_corrupt_originals_are_distinguished(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b"original")
        valid = inspect(self.root, verify_hashes=True)
        self.assertEqual((valid["missing_count"], valid["checksum_errors"]), (0, 0))
        self.target.write_bytes(b"tampered")
        invalid = inspect(self.root, verify_hashes=True)
        self.assertEqual(invalid["checksum_errors"], 1)
