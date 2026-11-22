import json
import sqlite3
from contextlib import closing
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

    def test_empty_update_keeps_catalog(self):
        storage.ensure_database()
        self.record.unlink()

        with self.assertLogs(storage.log, "ERROR"):
            storage.ensure_database(force_check=True)

        self.assertEqual(storage.get_term(1), self.term)
        self.assertEqual(list(self.root.glob("*.building")), [])

    def test_empty_initial_catalog_fails_without_creating_database(self):
        self.record.unlink()

        with self.assertRaisesRegex(ValueError, "No canonical term records"):
            storage.ensure_database(force_check=True)

        self.assertFalse(storage.DB.exists())
        self.assertEqual(list(self.root.glob("*.building")), [])

    def test_existing_empty_database_is_not_accepted_at_startup(self):
        storage.ensure_database()
        with closing(sqlite3.connect(storage.DB)) as db, db:
            self.record.unlink()
            db.execute("DELETE FROM terms")
            db.execute("UPDATE meta SET value=? WHERE key='fingerprint'", (storage.fingerprint(),))

        with self.assertRaisesRegex(ValueError, "No canonical term records"):
            storage.ensure_database(force_check=True)

    def test_category_filter_is_exact_even_for_sql_wildcards(self):
        self.term["metadata"]["categories"] = ["Star%Class", "Solar_System"]
        write_term(self.root, self.term)
        storage.rebuild_database()
        def total(category: str) -> int:
            return storage.list_records("term", page=1, page_size=10, locale="en", category=category)[1]
        self.assertEqual(total("Star%Class"), 1)
        self.assertEqual(total("Solar_System"), 1)
        self.assertEqual(total("StarXClass"), 0)
        self.assertEqual(total("SolarXSystem"), 0)

    def test_title_and_alias_filters_treat_wildcards_as_text(self):
        self.term["locales"]["en"]["title"] = "A%Star"
        self.term["locales"]["en"]["aliases"] = ["Bright_Sky"]
        write_term(self.root, self.term)
        storage.rebuild_database()

        def total(query: str) -> int:
            return storage.list_records("term", page=1, page_size=10, locale="en", query=query)[1]

        self.assertEqual(total("A%Star"), 1)
        self.assertEqual(total("Bright_Sky"), 1)
        self.assertEqual(total("AXStar"), 0)
        self.assertEqual(total("BrightXSky"), 0)
