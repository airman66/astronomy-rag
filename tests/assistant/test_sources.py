import json
from unittest.mock import AsyncMock, Mock, patch

from app.assistant import service
from app.assistant.client import Completion
from app.assistant.prompts import ANSWER_PROMPT, PROMPT_VERSION
from app.infra.config import ROOT
from tests.support.api import ANSWER_PATH, ApiCase
from tests.support.catalog import seed_star


class AssistantSourceTests(ApiCase):
    def setUp(self) -> None:
        super().setUp()
        seed_star(self.root)
        articles = self.root / "processed" / "articles"
        articles.mkdir(parents=True, exist_ok=True)
        for path in (ROOT / "data" / "processed" / "articles").glob("*.json"):
            (articles / path.name).write_bytes(path.read_bytes())

    def test_article_uses_exact_sections_and_single_completion(self):
        async def complete(system, payload):
            self.assertEqual(system, ANSWER_PROMPT)
            self.assertLessEqual(len(payload["evidence"]), 8)
            article = {item["heading"]: item["id"] for item in payload["evidence"]
                       if item["kind"] == "article" and item["title"] == "How astronomers detect exoplanets"}
            self.assertIn("Transit photometry", article)
            self.assertIn("Radial velocity", article)
            self.assertNotIn("url", json.dumps(payload))
            return Completion({"mode": "article", "coverage": "complete", "article_fallback": "default", "definition_id": None, "blocks": [
                {"text": "Transits cause dips in brightness.", "evidence_ids": [article["Transit photometry"]]},
                {"text": "Radial velocity uses spectral shifts.", "evidence_ids": [article["Radial velocity"]]},
            ]}, "test-model")

        call = AsyncMock(side_effect=complete)
        self.app.state.llm_client = Mock(complete=call)
        response = self.client.post(ANSWER_PATH, json={
            "question": "Find an exoplanet article comparing transit photometry and radial velocity.",
            "locale": "en", "context": {"unknown": "Private metadata must stay outside the prompt."},
        })
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["mode"], "article")
        self.assertEqual(body["diagnostics"]["model"], "test-model")
        self.assertEqual(body["diagnostics"]["prompt_version"], PROMPT_VERSION)
        self.assertEqual(body["primary_url"], "/api/v1/articles/article:exoplanet-detection")
        self.assertEqual([source["section_id"] for source in body["sources"]], ["transit", "radial-velocity"])
        self.assertEqual([row["source_indices"] for row in body["blocks"]], [[0], [1]])
        call.assert_awaited_once()
        self.assertNotIn("Private metadata", json.dumps(call.call_args.args[1]))

    def test_article_request_without_article_uses_only_supported_part(self):
        call = AsyncMock(return_value=Completion({
            "mode": "article", "coverage": "complete", "article_fallback": "default", "definition_id": None, "blocks": [
                {"text": "Звезда класса A имеет высокую температуру.", "evidence_ids": ["e1"]},
                {"text": "Этот тезис без ссылки.", "evidence_ids": []},
                {"text": "Ссылка неизвестна.", "evidence_ids": ["e999"]},
                {"text": "Смотрите https://invalid.example.", "evidence_ids": ["e1"]},
            ],
        }, "test-model"))
        self.app.state.llm_client = Mock(complete=call)
        evidence = service.evidence_for("Звезда", "ru")
        term = next(item for item in evidence.values() if item.record["id"] == "term:1")
        with patch.object(service, "evidence_for", return_value={"e1": term}):
            response = self.client.post(ANSWER_PATH, json={"question": "Объясни по статье", "mode": "article"})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["mode"], "explain")
        self.assertEqual(body["coverage"], "partial")
        self.assertEqual(len(body["blocks"]), 1)
        self.assertEqual(body["sources"][0]["record_id"], "term:1")
        self.assertNotIn("invalid.example", body["answer"])
        call.assert_awaited_once()

    def test_article_mode_reports_partial_when_only_terms_were_used(self):
        question = "Что такое звезда?"
        term = next(item for item in service.evidence_for(question, "ru").values() if item.record["id"] == "term:1")
        for model_mode in ("article", "explain"):
            with self.subTest(model_mode=model_mode):
                self.app.state.llm_client = Mock(complete=AsyncMock(return_value=Completion({
                    "mode": model_mode, "coverage": "complete", "article_fallback": "default", "definition_id": None,
                    "blocks": [{"text": "Горячая звезда.", "evidence_ids": ["e1"]}],
                }, "test-model")))
                with patch.object(service, "evidence_for", return_value={"e1": term}):
                    response = self.client.post(ANSWER_PATH, json={"question": question, "mode": "article"})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["coverage"], "partial")
                self.assertEqual(response.json()["mode"], "explain")

    def test_article_glossary_fallback_respects_source_policy(self):
        cases = (
            ("ru", "Если не найдётся отдельной статьи, объясни по справочнику.", "allow_glossary", "complete"),
            ("en", "If the article is unavailable, use the glossary.", "allow_glossary", "complete"),
            ("en", "Do not use the glossary unless the article is missing.", "allow_glossary", "complete"),
            ("en", "Use an article. Otherwise do not answer from the glossary.", "default", "insufficient"),
            ("ru", "Если статьи нет, не отвечай по справочнику.", "allow_glossary", "insufficient"),
            ("en", "Use the article as your sole source.", "article_only", "insufficient"),
        )
        for locale, question, fallback, coverage in cases:
            with self.subTest(locale=locale, coverage=coverage):
                term = next(
                    item for item in service.evidence_for("Star" if locale == "en" else "Звезда", locale).values()
                    if item.record["id"] == "term:1"
                )
                self.app.state.llm_client = Mock(complete=AsyncMock(return_value=Completion({
                    "mode": "explain", "coverage": "complete", "article_fallback": fallback, "definition_id": None,
                    "blocks": [{"text": term.text, "evidence_ids": ["e1"]}],
                }, "test-model")))
                with patch.object(service, "evidence_for", return_value={"e1": term}):
                    response = self.client.post(
                        ANSWER_PATH, json={"question": question, "mode": "article", "locale": locale},
                    )
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertEqual(body["mode"], "explain")
                self.assertEqual(body["coverage"], coverage)
                self.assertEqual(
                    [block["source_indices"] for block in body["blocks"]], [[0]] if coverage == "complete" else [],
                )
                self.assertEqual(
                    [source["record_id"] for source in body["sources"]], ["term:1"] if coverage == "complete" else [],
                )

    def test_glossary_fallback_drops_blocks_from_another_article(self):
        term = next(item for item in service.evidence_for("Star", "en").values() if item.record["id"] == "term:1")
        article = next(item for item in service.evidence_for("exoplanet transits", "en").values()
                       if item.kind == "article")
        cases = (
            ("article", "default", "Explain how a star shines, based on an article."),
            ("auto", "allow_glossary", "If the article on stars is unavailable, use the glossary."),
        )
        for mode, fallback, question in cases:
            with self.subTest(mode=mode):
                self.app.state.llm_client = Mock(complete=AsyncMock(return_value=Completion({
                    "mode": "explain", "coverage": "complete", "article_fallback": fallback, "definition_id": None,
                    "blocks": [
                        {"text": term.text, "evidence_ids": ["e1"]},
                        {"text": "An unrelated article claim.", "evidence_ids": ["e1", "e2"]},
                    ],
                }, "test-model")))
                with patch.object(service, "evidence_for", return_value={"e1": term, "e2": article}):
                    response = self.client.post(ANSWER_PATH, json={"question": question, "locale": "en", "mode": mode})
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertEqual(body["coverage"], "partial")
                self.assertEqual([source["record_id"] for source in body["sources"]], ["term:1"])
                self.assertEqual([block["text"] for block in body["blocks"]], [term.text])
