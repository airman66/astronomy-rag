import json
import unittest
from unittest.mock import patch

from app.catalog import storage
from tests.support.catalog import make_term, use_catalog, write_term


class CatalogRebuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = use_catalog(self)
        self.term = make_term(1, "Star", "Звезда", "Hot.", "Горячая.")
        write_term(self.root, self.term)
        self.record = self.root / "processed/terms/1.json"

    def test_invalid_update_keeps_catalog_and_retries_after_interval(self):
        storage.ensure_database()
        self.record.write_text("{")
        storage._checked[str(storage.DB)] -= storage.RECHECK_SECONDS
        with (
            patch.object(storage, "rebuild_database", wraps=storage.rebuild_database) as rebuild,
            self.assertLogs(storage.log, "ERROR"),
        ):
            self.assertEqual(storage.get_term(1), self.term)
            self.assertEqual(storage.get_term(1), self.term)
            rebuild.assert_called_once()

        self.term["locales"]["ru"]["title"] = "Звезда (исправлено)"
        write_term(self.root, self.term)
        storage._checked[str(storage.DB)] -= storage.RECHECK_SECONDS
        self.assertEqual(storage.get_term(1), self.term)
        self.assertEqual(list(self.root.glob("*.building")), [])

    def test_invalid_initial_catalog_fails_without_leaving_temporary_database(self):
        self.record.write_text("{")
        with self.assertRaises(json.JSONDecodeError):
            storage.ensure_database()
        self.assertFalse(storage.DB.exists())
        self.assertEqual(list(self.root.glob("*.building")), [])
