import json
import re
import unittest
from collections import Counter
from unittest.mock import patch

from app.infra.config import ROOT
from scripts.evaluation.retrieval import evaluate_response, evaluate_retrieval, price, summarize


class AssistantEvaluationReportTests(unittest.TestCase):
    def test_historical_report_still_describes_the_original_questions(self):
        directory = ROOT / "data/eval/assistant"
        questions = json.loads((directory / "questions.json").read_text(encoding="utf-8"))["questions"]
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["question_count"], len(questions))
        results = [{key: result[key] for key in questions[0]} for result in report["results"]]
        self.assertEqual(results, questions)

    def test_bilingual_suite_references_existing_records_and_sections(self):
        data = ROOT / "data"
        cases = json.loads((data / "eval/assistant/full_questions.json").read_text())["questions"]
        self.assertEqual(len({item["id"] for item in cases}), len(cases))
        self.assertEqual(Counter(item["locale"] for item in cases), {"ru": 25, "en": 25})
        records = {}
        for folder in ("articles", "terms"):
            for path in (data / "processed" / folder).glob("*.json"):
                item = json.loads(path.read_text())
                records[item["id"]] = {section["id"] for section in item["sections"]}
        for case in cases:
            groups = case.get("evidence", [])
            for fact in case.get("facts", []):
                for pattern in fact["patterns"]:
                    re.compile(pattern)
                groups = [*groups, *fact.get("evidence", [])]
            for group in groups:
                for selector in group:
                    identifier, _, section = selector.partition("#")
                    self.assertIn(identifier, records)
                    self.assertIn(section, records[identifier])

    def test_valid_existing_citation_to_the_wrong_section_fails_fact_coverage(self):
        case = {"expected_modes": ["explain"], "evidence": [["term:2#definition"]],
                "facts": [{"id": "distance", "patterns": [r"10 парсек"], "evidence": [["term:2#definition"]]}]}
        evidence = {"record_id": "term:2", "section_id": "other", "chunk_id": None, "url": "/api/v1/terms/2"}
        result = {"mode": "explain", "coverage": "complete", "grounding": "knowledge_base",
                  "answer": "10 парсек", "primary_url": evidence["url"], "images": [],
                  "sources": [evidence], "blocks": [{"text": "10 парсек", "source_indices": [0]}]}
        metrics = evaluate_response(case, result, [evidence])
        self.assertTrue(metrics["source_valid"])
        self.assertFalse(metrics["facts"]["distance"])
        self.assertFalse(metrics["passed"])

    def test_price_includes_failed_billable_answers_in_denominator(self):
        bill = price({"prompt_tokens": 1000, "completion_tokens": 200,
                      "prompt_tokens_details": {"cached_tokens": 200}}, 1, 2, .1)
        rows = [{"latency_ms": t, "passed": ok, "estimated_cost_usd": bill, "expected_modes": ["term"]}
                for t, ok in [(300, True), (900, False)]]
        summary = summarize(rows)
        self.assertEqual(summary["p95_ms"], 900)
        self.assertAlmostEqual(summary["estimated_usd_per_1000_valid"], 2.44)

    def test_retrieval_checks_fact_only_sources_and_uses_explicit_mode(self):
        case = {"id": "fact-only", "requested_mode": "article", "locale": "ru", "question": "Объясни",
                "facts": [{"evidence": [["article:stellar-magnitudes#absolute"]]}]}
        with patch("scripts.evaluation.retrieval.evidence_for", return_value={}) as retrieve:
            result = evaluate_retrieval(case)
        assert result is not None
        self.assertFalse(result["all_expected"])
        self.assertEqual(result["missing_groups"], [["article:stellar-magnitudes#absolute"]])
        retrieve.assert_called_once_with("Объясни", "ru", prefer_article=True)
