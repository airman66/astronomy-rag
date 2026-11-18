import hashlib
import json
from unittest.mock import patch

from app.pipeline import importer
from tests.pipeline.fixtures import ImporterCase


class SnapshotRefreshTests(ImporterCase):
    def test_refresh_preserves_raw_snapshots_and_reads_current_version(self):
        path = self.root / "raw/terms/1.json"
        first = b'{"results": [1]}'
        second = b'{"results": [2]}'
        with patch.object(importer, "fetch", side_effect=[first, second, first]) as fetch:
            self.assertEqual(importer.raw_json(path, "https://example.test/1")["results"], [1])
            importer.REFRESH = True
            self.assertEqual(importer.raw_json(path, "https://example.test/1")["results"], [2])
            self.assertEqual(fetch.call_count, 2)
            self.assertNotEqual(importer.current_snapshot(path), path)
            self.assertEqual(importer.raw_json(path, "https://example.test/1")["results"], [1])
        self.assertEqual(path.read_bytes(), first)
        revision = importer.current_snapshot(path)
        self.assertEqual(revision, path)
        self.assertTrue(path.with_name(f"1.{hashlib.sha256(second).hexdigest()}.json").exists())
        importer.REFRESH = False
        self.assertEqual(importer.raw_json(path, "https://example.test/1")["results"], [1])

    def test_invalid_refreshed_json_does_not_replace_current_snapshot(self):
        path = self.root / "raw/terms/1.json"
        importer.save_snapshot(path, b'{"results": [1]}')
        importer.REFRESH = True
        with (
            patch.object(importer, "fetch", return_value=b"not json"),
            self.assertRaises(json.JSONDecodeError),
        ):
            importer.raw_json(path, "https://example.test/1")
        self.assertEqual(importer.current_snapshot(path).read_bytes(), b'{"results": [1]}')
