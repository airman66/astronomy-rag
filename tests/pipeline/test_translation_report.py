"""The translation audit report must describe every field of every canonical record (machine check)."""

import json
import unittest

from app.pipeline.translation_fields import fields, get_path
from scripts.translation_review.common import (
    REPORT,
    REVIEW_CORPUS,
    load_records,
    manual_media_edits,
    manual_page_edits,
)

VERDICTS = {"ok", "manual_ok", "fix", "source_error", "uncertain", "pending", "not_covered"}


@unittest.skipUnless(REPORT.exists(), "отчёт ревью переводов ещё не построен")
class TranslationAuditReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.records = load_records()

    def test_every_record_is_reported(self):
        self.assertEqual(set(self.report["records"]), set(self.records))
        if not REVIEW_CORPUS:
            self.assertEqual(len(self.records), 501)

    def test_every_translatable_field_has_a_verdict(self):
        for record_id, record in self.records.items():
            listed = {row[0]: row[2] for row in self.report["records"][record_id]["fields_checked"]}
            expected = {field["ru_path"] for field in fields(record)}
            self.assertEqual(set(listed), expected, record_id)
            self.assertTrue(set(listed.values()) <= VERDICTS, record_id)

    def test_fixes_match_canonical_text(self):
        for record_id, entry in self.report["records"].items():
            record = self.records[record_id]
            if record["review"]["status"] == "reviewed":
                continue  # a human editor may have changed it after the audit
            for fix in entry["fixed"]:
                self.assertEqual(get_path(record, fix["path"]), fix["new"], f"{record_id} {fix['path']}")

    def test_automatic_review_does_not_claim_human_confirmation(self):
        if not REVIEW_CORPUS:
            self.assertIn("Не является подтверждением человеком", self.report["method"])
        for record_id, entry in self.report["records"].items():
            self.assertEqual(entry["review_status"], self.records[record_id]["review"]["status"])

    @unittest.skipUnless(REVIEW_CORPUS, "ручная сверка OAE Reviews хранится в отдельном отчёте")
    def test_manual_verdicts_match_reviewed_source_and_translation(self):
        manual = {
            (record_id, path)
            for edits in (manual_media_edits(self.records), manual_page_edits(self.records))
            for record_id, path, _ in edits
        }
        reported = {
            (record_id, path)
            for record_id, entry in self.report["records"].items()
            for path, _, verdict in entry["fields_checked"]
            if verdict == "manual_ok"
        }
        self.assertTrue(manual)
        self.assertEqual(reported, manual)
