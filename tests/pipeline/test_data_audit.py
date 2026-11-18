import unittest
from unittest.mock import patch

from app.pipeline import audit
from scripts.covers import manifest
from tests.support.catalog import seed_star, temporary_dir, use_catalog


class DataAuditTests(unittest.TestCase):
    def test_audit_fails_when_catalog_is_missing(self):
        with patch.object(audit, "DATA", temporary_dir(self)):
            self.assertTrue(audit.check()["errors"])

    def test_complete_cover_audit_fails_when_catalog_is_missing(self):
        root = temporary_dir(self)
        with (
            patch.multiple(manifest, TERMS=root / "terms", FILES=root / "files", MANIFEST=root / "manifest.json"),
            patch("builtins.print"),
        ):
            self.assertEqual(manifest.verify(complete=True), 1)

    def test_clean_checkout_can_skip_downloaded_originals(self):
        root = use_catalog(self)
        seed_star(root).unlink()
        with patch.object(audit, "DATA", root):
            without_original = "media media:example: missing/changed original image"
            self.assertIn(without_original, audit.check()["errors"])
            self.assertNotIn(without_original, audit.check(skip_original_media=True)["errors"])
