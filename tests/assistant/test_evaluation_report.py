import asyncio
import json
import os
import re
import subprocess
import sys
import unittest
from collections import Counter
from unittest.mock import patch

from app.catalog.corpus import corpus_digest
from app.infra.config import ROOT
from app.infra.files import checksum, write_json
from scripts.catalog_release import publish
from scripts.evaluation.retrieval import evaluate_response, evaluate_retrieval, main, passage_rank, price, summarize
from tests.catalog.fixtures import bind_vector_index, seed_vector_index
from tests.support.catalog import make_term, temporary_dir, write_term


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

        other_url = "/api/v1/terms/3"
        result["primary_url"] = other_url
        result["sources"] = [{**evidence, "url": other_url}]
        self.assertFalse(evaluate_response(case, result, [evidence])["source_valid"])

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
        retrieve.assert_called_once_with("Объясни", "ru", prefer_article=True, context_record_id=None, trace={})

    def test_required_hybrid_rejects_fallback_without_expected_evidence(self):
        directory = temporary_dir(self)
        suite = directory / "questions.json"
        output = directory / "report.json"
        write_json(suite, {"questions": [
            {"id": "scope", "question": "Sports", "locale": "en"},
            {"id": "general", "question": "Sports", "locale": "en", "requested_mode": "general"},
        ]})

        def lexical_fallback(*args, trace, **kwargs):
            trace.update(search_mode="lexical", vector_hits=0)
            return {}

        cli = ["evaluate", "--retrieval-only", "--check", "--require-search-mode", "hybrid",
               "--suite", str(suite), "--output", str(output)]
        with (
            patch.object(sys, "argv", cli),
            patch("builtins.print"),
            patch("scripts.evaluation.retrieval.load_env"),
            patch("scripts.evaluation.retrieval.evidence_for", side_effect=lexical_fallback) as retrieve,
        ):
            with self.assertRaisesRegex(SystemExit, "Required hybrid search"):
                asyncio.run(main())
        retrieve.assert_called_once()
        report = json.loads(output.read_bytes())
        self.assertEqual(report["metrics"]["search_modes"], {"lexical": 1})
        self.assertEqual(report["metrics"]["evidence_cases"], 0)
        self.assertEqual(report["metrics"]["with_all_evidence"], 0)
        (row,) = report["results"]
        self.assertEqual(row["id"], "scope")
        self.assertIsNone(row["all_expected"])

    def test_release_eval_reads_default_suite_and_writes_report_in_workspace(self):
        root = temporary_dir(self)
        source = root / "source"
        write_term(source, make_term(1, "Sun", "Солнце", "A star.", "Звезда."))
        seed_vector_index(source)
        bind_vector_index(source, corpus_digest(source))
        (source / "models/small.gguf").unlink()
        release = publish(source, root / "releases")
        workspace = root / "workspace"
        evaluation = workspace / "eval/assistant"
        write_json(evaluation / "full_questions.json", {"questions": [
            {"id": "sun", "question": "Sun", "locale": "en", "evidence": [["term:1#definition"]]},
        ]})

        result = subprocess.run(
            [sys.executable, "-m", "scripts.evaluate", "--retrieval-only", "--check", "--require-search-mode", "lexical"],
            cwd=ROOT, env={**os.environ, "DATA_DIR": str(workspace), "DATA_RELEASE": str(release)},
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((evaluation / "full_retrieval_report.json").read_bytes())
        self.assertEqual(report["index_sha256"], checksum(release / "index/index-manifest.json"))
        self.assertEqual(report["metrics"]["with_all_evidence"], 1)
        self.assertFalse((release / "eval").exists())

    def test_matching_article_without_the_passage_does_not_pass(self):
        source = {"record_id": "article:review", "section_id": "page-2"}
        passage = {"source": "article:review#page-2", "quote": "a specific passage"}
        self.assertIsNone(passage_rank(passage, [
            {**source, "text": "an unrelated paragraph"},
            {**source, "section_id": "page-3", "text": "a specific passage"},
        ]))
        self.assertEqual(passage_rank(passage, [{**source, "text": "a specific\n passage"}]), 1)


    def test_oae_suite_covers_both_languages_for_every_rechunked_article(self):
        data = ROOT / "data"
        cases = json.loads((data / "eval/assistant/oae_passage_questions.json").read_text())["questions"]
        self.assertEqual(len({item["id"] for item in cases}), 48)
        groups = Counter()
        for case in cases:
            (passage,) = case["passages"]
            record, _, section = passage["source"].partition("#")
            locale = case["locale"]
            source = data / "processed" / "articles" / (record.removeprefix("article:") + ".json")
            text = json.loads(source.read_text())["locales"][locale]["blocks"][section]["body"]
            self.assertIn(passage["quote"], " ".join(text.split()))
            groups[record, locale] += 1
        self.assertEqual(len({record for record, _ in groups}), 24)
        self.assertTrue(all(groups[record, "en"] == groups[record, "ru"] == 1 for record, _ in groups))
