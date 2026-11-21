"""The translation audit report must describe every field of every canonical record (machine check)."""

import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from scripts.translation_review import apply, common
from scripts.translation_review.common import (
    REVIEW_PROMPT_VERSION,
    VERIFY_PROMPT_VERSION,
)


class TranslationAuditApplyTests(unittest.TestCase):
    def test_current_manual_article_review_protects_both_copies_and_links(self):
        data = Path(self.enterContext(tempfile.TemporaryDirectory()))
        ledger = data / "manual_pages.json"
        record_id = "article:review"
        link = {"url": "https://example.org/a", "text": "Resource", "text_ru": "Материал"}
        section = {
            "id": "intro",
            "heading": "Introduction",
            "heading_ru": "Введение",
            "body": "Source text",
            "body_ru": "Проверенный текст",
            "links": [link],
        }
        record = {
            "id": record_id,
            "sections": [section],
            "locales": {
                "en": {"blocks": {"intro": {"heading": section["heading"], "body": section["body"]}}},
                "ru": {"blocks": {"intro": {"heading": section["heading_ru"], "body": section["body_ru"]}}},
            },
            "review": {"status": "draft"},
        }
        records = {record_id: record}
        ledger.write_text(json.dumps({"records": {record_id: {"intro": {
            "en_sha256": sha256(section["body"].encode()).hexdigest(),
            "ru_sha256": sha256(section["body_ru"].encode()).hexdigest(),
            "heading_links_en_sha256": sha256(b'["Introduction",[["https://example.org/a","Resource"]]]').hexdigest(),
            "heading_links_ru_sha256": sha256('["Введение",[["https://example.org/a","Материал"]]]'.encode()).hexdigest(),
        }}}}))
        expected = {
            "/sections/0/heading_ru": "Введение",
            "/locales/ru/blocks/intro/heading": "Введение",
            "/sections/0/body_ru": "Проверенный текст",
            "/locales/ru/blocks/intro/body": "Проверенный текст",
            "/sections/0/links/0/text_ru": "Материал",
        }

        with patch.object(common, "MANUAL_PAGES", ledger):
            self.assertEqual({path: value for _, path, value in common.manual_page_edits(records)}, expected)
            link["url"] = "https://example.org/b"
            self.assertEqual(
                {path for _, path, _ in common.manual_page_edits(records)},
                {"/sections/0/body_ru", "/locales/ru/blocks/intro/body"},
            )
            section["body"] = "New source text"
            self.assertEqual(list(common.manual_page_edits(records)), [])

    def test_previous_edits_survive_the_legacy_reports_move(self):
        data = Path(self.enterContext(tempfile.TemporaryDirectory()))
        eval_dir = data / "eval"
        legacy = eval_dir / "translation_legacy"
        legacy.mkdir(parents=True)
        resolutions = eval_dir / "translation_resolutions.json"
        numeric = eval_dir / "numeric_patches.json"
        resolutions.write_text(
            json.dumps({"results": {"original-pair": {"valid": True, "corrected_ru": "Текст"}}})
        )
        numeric.write_text(json.dumps([{"valid": True, "old": "100", "new": "10"}]))

        expected = (set(), {"original-pair"}, {"100"})
        with patch.multiple(apply, DATA=data, EDITORIAL=eval_dir / "editorial_sync.json"):
            self.assertEqual(apply.previous_edits(), expected)
            resolutions.rename(legacy / resolutions.name)
            numeric.rename(legacy / numeric.name)
            self.assertEqual(apply.previous_edits(), expected)

    def test_requires_verification_and_preserves_earlier_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            journal = data / "applied.json"
            review = data / "review.jsonl"
            verify = data / "verify.jsonl"
            editorial = data / "editorial_sync.json"
            title = "/locales/ru/title"
            journal.write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "applied": [
                                    {
                                        "record_id": "term:2",
                                        "path": title,
                                        "old": "Старая версия",
                                        "new": "Исправлено",
                                    },
                                ]
                            }
                        ]
                    }
                )
            )
            editorial.write_text(json.dumps({"changes": [{"new": "Редактура"}]}))
            inputs = [
                ("Оригинал 1", "Черновик", "Без проверки"),
                ("Оригинал 2", "Исправлено", "Старая версия"),
                ("Оригинал 3", "Редактура", "Замена"),
                ("Оригинал 4", "Старый вариант", "Подтверждено"),
            ]
            records = {}
            units = {}
            reviews = []
            verifications = []
            for number, (source, current, proposal) in enumerate(inputs, 1):
                record_id = f"term:{number}"
                unit = str(number)
                records[record_id] = {
                    "source": {"term_number": number},
                    "locales": {"ru": {"title": current}},
                    "sections": [],
                    "review": {"status": "draft"},
                }
                units[unit] = {"en": source, "ru": current, "occurrences": [(record_id, title)]}
                reviews.append(
                    {
                        "unit": unit,
                        "verdict": "fix",
                        "corrected_ru": proposal,
                        "issues": [],
                        "prompt_version": REVIEW_PROMPT_VERSION,
                    }
                )
                if number != 1:
                    verifications.append(
                        {
                            "unit": unit,
                            "decision": "accept",
                            "prompt_version": VERIFY_PROMPT_VERSION,
                        }
                    )
            for path, rows in ((review, reviews), (verify, verifications)):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with patch.multiple(
                apply,
                DATA=data,
                TERMS=data,
                APPLIED=journal,
                REVIEW=review,
                VERIFY=verify,
                EDITORIAL=editorial,
            ):
                result = apply.run_apply(units, records)
                repeated = apply.run_apply(units, records)
            self.assertEqual([row["record_id"] for row in result["applied"]], ["term:4"])
            self.assertEqual(
                [row["reason"] for row in result["disputed"]],
                ["verifier_missing", "reverts_earlier_fix", "manual_editorial_edit"],
            )
            self.assertEqual(repeated["applied"], [])
            self.assertEqual(json.loads((data / "4.json").read_text())["review"]["status"], "draft")
