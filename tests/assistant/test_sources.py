import json
from unittest.mock import AsyncMock, Mock, patch

from app.assistant import service
from app.assistant.client import Completion
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
        async def complete(_system, payload):
            self.assertLessEqual(len(payload["evidence"]), 6)
            article = {item["heading"]: item["id"] for item in payload["evidence"]
                       if item["kind"] == "article" and item["title"] == "How astronomers detect exoplanets"}
            self.assertIn("Transit photometry", article)
            self.assertIn("Radial velocity", article)
            self.assertNotIn("url", json.dumps(payload))
            return Completion({"mode": "article", "coverage": "complete", "definition_id": None, "blocks": [
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
        self.assertEqual(body["primary_url"], "/api/v1/articles/article:exoplanet-detection")
        self.assertEqual([source["section_id"] for source in body["sources"]], ["transit", "radial-velocity"])
        self.assertEqual([row["source_indices"] for row in body["blocks"]], [[0], [1]])
        call.assert_awaited_once()
        self.assertNotIn("Private metadata", json.dumps(call.call_args.args[1]))

    def test_article_request_without_article_uses_only_supported_part(self):
        call = AsyncMock(return_value=Completion({
            "mode": "article", "coverage": "complete", "definition_id": None, "blocks": [
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
                    "mode": model_mode, "coverage": "complete", "definition_id": None,
                    "blocks": [{"text": "Горячая звезда.", "evidence_ids": ["e1"]}],
                }, "test-model")))
                with patch.object(service, "evidence_for", return_value={"e1": term}):
                    response = self.client.post(ANSWER_PATH, json={"question": question, "mode": "article"})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["coverage"], "partial")
                self.assertEqual(response.json()["mode"], "explain")
