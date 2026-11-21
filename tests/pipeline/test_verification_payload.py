"""The independent verifier receives identical fields from every execution path."""

import unittest

from scripts.translation_review.review import verification_item


class VerificationPayloadTests(unittest.TestCase):
    def test_online_and_offline_verification_share_the_review_contract(self):
        unit_id = "0123456789abcdef0123456789abcdef"
        units = {
            unit_id: {
                "context_record": "term:1",
                "kinds": ["definition", "heading"],
                "en": "Star",
                "ru": "Стар",
                "target_ru_title": "Звезда",
            }
        }
        records = {"term:1": {"locales": {"en": {"title": "Star"}}}}
        review = {
            "unit": unit_id,
            "corrected_ru": "Звезда",
            "issues": [{"type": "terminology", "note": "Incorrect term"}],
        }

        self.assertEqual(
            verification_item(review, units, records),
            {
                "id": unit_id[:16],
                "kind": "definition/heading",
                "term_en": "Star",
                "en": "Star",
                "ru_current": "Стар",
                "ru_proposed": "Звезда",
                "reviewer_issues": review["issues"],
                "target_ru_title": "Звезда",
            },
        )

    def test_optional_target_title_is_included_as_null(self):
        units = {
            "full-id": {"context_record": "term:1", "kinds": ["heading"], "en": "Sky", "ru": "Небо"}
        }
        records = {"term:1": {"locales": {"en": {"title": "Star"}}}}
        review = {"unit": "full-id", "corrected_ru": "Свод", "issues": []}

        self.assertIsNone(verification_item(review, units, records)["target_ru_title"])
