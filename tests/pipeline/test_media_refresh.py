import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from app.pipeline import importer
from tests.pipeline.fixtures import ImporterCase


class MediaRefreshTests(ImporterCase):
    def test_changed_image_gets_new_file_and_invalidates_russian_version(self):
        processed = self.root / "processed/terms/1.json"
        processed.parent.mkdir(parents=True)
        original = self.root / "media/original/media_example.png"
        russian = self.root / "media/ru/media_example.png"
        original.parent.mkdir(parents=True)
        russian.parent.mkdir(parents=True)
        original.write_bytes(b"old original")
        russian.write_bytes(b"old russian")
        record = {
            "id": "term:1",
            "source": {"term_number": 1},
            "media": [
                {
                    "id": "media:example",
                    "source_url": "https://example.test/image.png",
                    "file_type": "img",
                    "original_path": str(original),
                    "sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                    "ru_path": str(russian),
                    "ru_sha256": hashlib.sha256(russian.read_bytes()).hexdigest(),
                    "review": {"status": "reviewed", "reviewer": "editor"},
                }
            ],
            "review": {"status": "reviewed", "reviewer": "editor"},
        }
        processed.write_text(json.dumps(record))
        second = copy.deepcopy(record)
        second["id"] = "term:2"
        second["source"]["term_number"] = 2
        (processed.parent / "2.json").write_text(json.dumps(second))
        with patch.object(importer, "fetch", return_value=b"new original"):
            downloaded, failed = importer.download_media(workers=1, refresh=True, preferred_ids={1})
        self.assertEqual((downloaded, failed), (1, 0))
        updated = json.loads(processed.read_text())
        item = updated["media"][0]
        self.assertEqual(original.read_bytes(), b"old original")
        stored = Path(item["original_path"])
        self.assertEqual(stored.parent, Path("media/original"))
        self.assertEqual((self.root / stored).read_bytes(), b"new original")
        self.assertIsNone(item["ru_path"])
        self.assertIsNone(item["ru_sha256"])
        self.assertEqual(item["review"]["status"], "draft")
        self.assertEqual(updated["review"]["status"], "draft")
        shared = json.loads((processed.parent / "2.json").read_text())
        self.assertEqual(shared["media"][0]["original_path"], item["original_path"])
        self.assertIsNone(shared["media"][0]["ru_path"])
        self.assertEqual(shared["review"]["status"], "draft")

        with patch.object(importer, "fetch") as fetch:
            self.assertEqual(importer.download_media(workers=1), (0, 0))
            fetch.assert_not_called()
