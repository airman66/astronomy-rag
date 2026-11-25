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
            return Completion({"mode": "article", "coverage": "complete", "article_fallback": "default", "article_reference": None, "definition_id": None, "blocks": [
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
            "mode": "article", "coverage": "complete", "article_fallback": "default", "article_reference": None, "definition_id": None, "blocks": [
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
                    "mode": model_mode, "coverage": "complete", "article_fallback": "default", "article_reference": None, "definition_id": None,
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
            ("en", "If no article is available, answer using the glossary.", "default", "complete"),
            ("en", "Use an article. Otherwise do not answer from the glossary.", "default", "insufficient"),
            ("ru", "Если статьи нет, не отвечай по справочнику.", "allow_glossary", "insufficient"),
            ("en", "Use the article as your sole source.", "article_only", "insufficient"),
            ("en", "Use only the article “The spectra of stars”.", "allow_glossary", "insufficient"),
            ("en", 'Use only an article. The phrase "otherwise use the glossary" is an example.',
             "allow_glossary", "insufficient"),
        )
        for locale, question, fallback, coverage in cases:
            with self.subTest(question=question):
                term = next(
                    item for item in service.evidence_for("Star" if locale == "en" else "Звезда", locale).values()
                    if item.record["id"] == "term:1"
                )
                self.app.state.llm_client = Mock(complete=AsyncMock(return_value=Completion({
                    "mode": "explain", "coverage": "complete", "article_fallback": fallback, "article_reference": None, "definition_id": None,
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
                    "mode": "explain", "coverage": "complete", "article_fallback": fallback, "article_reference": None, "definition_id": None,
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

    def test_named_article_rejects_another_article_and_keeps_matching_passages(self):
        article = next(item for item in service.evidence_for("exoplanet transits", "en").values()
                       if item.record["id"] == "article:exoplanet-detection")
        other = next(item for item in service.evidence_for("dark matter and dark energy", "en").values()
                     if item.record["id"] == "article:dark-matter-and-energy")
        cases = (
            ("Use only the article “The spectra of stars”.", None, None, None),
            ("Use only the article “The spectra of stars”.", article.title, None, None),
            ("According to The spectra of stars, what does it say about absorption lines?",
             "The spectra of stars", None, None),
            (f"Use only the article “{article.title}”.", None, None, article.record["id"]),
            (f"According to {article.title}, what do transits measure?", article.title, None, article.record["id"]),
            (f"Answer using {article.record['id']}.", None, None, article.record["id"]),
            ("What do transits measure according to this article?", None, article.record["id"], article.record["id"]),
        )
        for question, reference, page, expected in cases:
            with self.subTest(question=question, reference=reference):
                self.app.state.llm_client = Mock(complete=AsyncMock(return_value=Completion({
                    "mode": "article", "coverage": "complete", "article_fallback": "default",
                    "article_reference": reference, "definition_id": None, "blocks": [
                        {"text": "Claim from another article.", "evidence_ids": ["e1"]},
                        {"text": "Claim from the exoplanet article.", "evidence_ids": ["e2"]},
                    ],
                }, "test-model")))
                with patch.object(service, "evidence_for", return_value={"e1": other, "e2": article}):
                    response = self.client.post(ANSWER_PATH, json={
                        "question": question, "locale": "en", "context": {"record_id": page},
                    })
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertEqual([item["record_id"] for item in body["sources"]], [expected] if expected else [])
                self.assertEqual(body["coverage"], "partial" if expected else "insufficient")
                self.assertEqual(body["mode"], "article" if expected else "explain")

    def test_explain_article_only_rejects_glossary_and_accepts_article(self):
        term = next(item for item in service.evidence_for("Star", "en").values() if item.record["id"] == "term:1")
        article = next(item for item in service.evidence_for("exoplanet transits", "en").values()
                       if item.record["id"] == "article:exoplanet-detection")
        for source, fallback, expected in ((term, "article_only", False), (term, "default", False),
                                             (article, "article_only", True)):
            with self.subTest(source=source.kind, fallback=fallback):
                self.app.state.llm_client = Mock(complete=AsyncMock(return_value=Completion({
                    "mode": "explain", "coverage": "complete", "article_fallback": fallback,
                    "article_reference": None, "definition_id": None,
                    "blocks": [{"text": source.text, "evidence_ids": ["e1"]}],
                }, "test-model")))
                with patch.object(service, "evidence_for", return_value={"e1": source}):
                    response = self.client.post(ANSWER_PATH, json={
                        "question": "Объясни только по статье. Не используй справочник.", "mode": "explain",
                    })
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertEqual(body["mode"], "explain")
                self.assertEqual(body["coverage"], "complete" if expected else "insufficient")
                self.assertEqual([item["record_id"] for item in body["sources"]],
                                 [article.record["id"]] if expected else [])

    def test_named_mini_review_excludes_the_topic_page(self):
        question = "По мини-обзору OAE «Дистанционные наблюдения» объясни, за что выдавали значки."
        pdf_id = "article:oae-document-remoteobserving_mini_review"
        topic_id = "article:oae-review-remote-observing"
        pdf, topic = [
            next(item for item in service.evidence_for("Дистанционные наблюдения", "ru", context_record_id=record_id).values()
                 if item.record["id"] == record_id)
            for record_id in (pdf_id, topic_id)
        ]
        self.app.state.llm_client = Mock(complete=AsyncMock(return_value=Completion({
            "mode": "article", "coverage": "complete", "article_fallback": "article_only",
            "article_reference": "Дистанционные наблюдения", "definition_id": None, "blocks": [
                {"text": "Тезис обзорной страницы.", "evidence_ids": ["e1"]},
                {"text": "Тезис мини-обзора.", "evidence_ids": ["e2"]},
            ],
        }, "test-model")))
        with patch.object(service, "evidence_for", return_value={"e1": topic, "e2": pdf}):
            response = self.client.post(ANSWER_PATH, json={"question": question})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["coverage"], "partial")
        self.assertEqual([item["record_id"] for item in body["sources"]], [pdf_id])

    def test_topic_or_invented_title_does_not_narrow_an_unnamed_article_request(self):
        article = next(item for item in service.evidence_for("stellar magnitude", "en").values()
                       if item.record["id"] == "article:stellar-magnitudes")
        for question, reference, fallback in (
            ("Using the article on stellar magnitudes, explain apparent magnitude.", "stellar magnitudes", "default"),
            ("Use the article on stellar magnitudes to explain apparent magnitude.",
             "the article on stellar magnitudes", "default"),
            ("Use an article to explain apparent magnitude.", article.title, "default"),
            ("Use the article to explain apparent magnitude.", "the article", "article_only"),
        ):
            with self.subTest(question=question):
                self.app.state.llm_client = Mock(complete=AsyncMock(return_value=Completion({
                    "mode": "article", "coverage": "complete", "article_fallback": fallback,
                    "article_reference": reference, "definition_id": None,
                    "blocks": [{"text": article.text, "evidence_ids": ["e1"]}],
                }, "test-model")))
                with patch.object(service, "evidence_for", return_value={"e1": article}):
                    response = self.client.post(ANSWER_PATH, json={"question": question, "locale": "en"})
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertEqual((body["mode"], body["coverage"]), ("article", "complete"))
                self.assertEqual([item["record_id"] for item in body["sources"]], [article.record["id"]])

    def test_explain_keeps_article_when_glossary_is_only_a_fallback(self):
        async def complete(_system, payload):
            section = next(item for item in payload["evidence"] if item["kind"] == "article"
                           and item["heading"] == "The apparent magnitude")
            return Completion({
                "mode": "explain", "coverage": "complete", "article_fallback": "allow_glossary",
                "article_reference": None, "definition_id": None,
                "blocks": [{"text": section["text"], "evidence_ids": [section["id"]]}],
            }, "test-model")

        self.app.state.llm_client = Mock(complete=AsyncMock(side_effect=complete))
        response = self.client.post(ANSWER_PATH, json={
            "question": "Explain apparent magnitude using the article on stellar magnitudes. "
                        "If the article is unavailable, use the glossary.",
            "locale": "en", "mode": "explain",
        })
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual((body["mode"], body["coverage"]), ("explain", "complete"))
        self.assertEqual([source["record_id"] for source in body["sources"]], ["article:stellar-magnitudes"])

    def test_explicit_sources_drive_search_and_comparison_coverage(self):
        mini_review = "article:oae-document-remoteobserving_mini_review"
        exoplanets = "article:exoplanet-detection"
        cases = (
            (f"According to {mini_review}, how were badges awarded?", (mini_review,), "explain", "complete"),
            (f"Compare the observing tasks in {mini_review} and {exoplanets}. How were badges awarded?",
             (mini_review, exoplanets), "explain", "complete"),
            (f"Compare the observing tasks in {mini_review} and {exoplanets}. How were badges awarded?",
             (mini_review, exoplanets), "explain", "partial"),
            ("Compare observations in the article “How astronomers detect exoplanets” and the article "
             "“Apparent and absolute stellar magnitude”.", (exoplanets, "article:stellar-magnitudes"), "article", "complete"),
        )
        for question, requested_ids, model_mode, coverage in cases:
            with self.subTest(question=question, coverage=coverage):
                used = requested_ids if coverage == "complete" else requested_ids[:1]

                async def complete(_system, payload):
                    groups = payload["requested_articles"]
                    self.assertEqual(tuple(groups), requested_ids)
                    self.assertTrue(all(groups.values()))
                    passages = {item["id"]: item["text"] for item in payload["evidence"]}
                    keys = [
                        next(key for key in groups[record_id] if "a new badge unlocked" in passages[key])
                        if record_id == mini_review else groups[record_id][0]
                        for record_id in used
                    ]
                    return Completion({
                        "mode": model_mode, "coverage": "complete", "article_fallback": "default",
                        "article_reference": None, "definition_id": None,
                        "blocks": [{"text": passages[key][:160], "evidence_ids": [key]} for key in keys],
                    }, "test-model")

                self.app.state.llm_client = Mock(complete=AsyncMock(side_effect=complete))
                with patch("app.catalog.search.vector.search", return_value=None):
                    response = self.client.post(ANSWER_PATH, json={"question": question, "locale": "en", "mode": "explain"})
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertEqual((body["mode"], body["coverage"]), ("explain", coverage))
                self.assertEqual(tuple(source["record_id"] for source in body["sources"]), used)
