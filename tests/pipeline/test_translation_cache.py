"""Translation cache must not conflate identical strings in different contexts."""

import json
import unittest
from unittest.mock import Mock, patch

from app.pipeline import translate


class ScopedTranslationCacheTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(translate, "CACHE", {}))
        self.enterContext(patch.object(translate, "write_json"))
        self.enterContext(patch.object(translate.time, "sleep"))

    def test_same_source_in_different_fields_is_translated_separately(self):
        requests_seen = []

        def post(_url, **kwargs):
            rows = json.loads(kwargs["json"]["messages"][1]["content"])
            requests_seen.extend(rows)
            response = Mock()
            response.json.return_value = {"choices": [{"message": {"content": json.dumps({
                row["id"]: "Светило" if "stars" in row["context"] else "Звезда"
                for row in rows
            }, ensure_ascii=False)}}]}
            return response

        with patch.object(translate.requests, "post", side_effect=post) as http:
            items = [
                {"id": "a", "text": "Star", "context": "article:stars | heading"},
                {"id": "b", "text": "Star", "context": "term:star | title"},
            ]
            result = translate.translate_batch(items, "unit-test-key")
            self.assertEqual(result, {"a": "Светило", "b": "Звезда"})
            again = translate.translate_batch(items, "unit-test-key")
        self.assertEqual(result, again)
        self.assertEqual(http.call_count, 1)
        self.assertEqual(len(requests_seen), 2)
        self.assertEqual(len(translate.CACHE), 2)

    def test_legacy_text_only_cache_entries_are_not_trusted(self):
        translate.CACHE["Star"] = "Устаревший перевод"
        self.assertNotIn(translate.translation_cache_key("Star", "term:star"), translate.CACHE)
        self.assertNotEqual(
            translate.translation_cache_key("Star", "term:1"),
            translate.translation_cache_key("Star", "term:2"),
        )
