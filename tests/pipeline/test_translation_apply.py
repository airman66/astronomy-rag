"""The translation audit report must describe every field of every canonical record (machine check)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.translation_review import apply
from scripts.translation_review.common import (
    REVIEW_PROMPT_VERSION,
    VERIFY_PROMPT_VERSION,
)


class TranslationAuditApplyTests(unittest.TestCase):
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
