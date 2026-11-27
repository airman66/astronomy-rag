import json
import unittest

import httpx

from experiments.gigachat4.grade import retrieval_row, validate_suite
from experiments.gigachat4.providers import DEEPSEEK, GIGACHAT, Candidate
from experiments.gigachat4.run import digest, generate


class AstronomyBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    def test_suite_and_exact_passage(self) -> None:
        questions, rules = validate_suite()
        self.assertEqual(questions.keys(), rules.keys())
        candidate_source = {
            "record_id": "article:sample",
            "section_id": "p1",
            "text": "Other passage in the same section",
        }
        snapshot = {
            "id": "test",
            "evidence": [candidate_source],
            "search_mode": "hybrid",
            "retrieval_ms": 1.0,
            "vector_hits": 1,
        }
        rule = {
            "category": "article",
            "facts": [
                {
                    "sources": [
                        {"source": "article:sample#p1", "quote": "required phrase"}
                    ]
                },
            ],
        }
        self.assertFalse(retrieval_row(snapshot, rule)["retrieved_all"])
        snapshot["evidence"].append(
            {**candidate_source, "text": "Contains the required phrase"}
        )
        self.assertEqual(retrieval_row(snapshot, rule)["expected_fact_ranks"], [2])

    async def test_candidates_get_the_same_production_messages(self) -> None:
        requests = []

        def completion(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            requests.append(body)
            reply = {
                "mode": "out_of_scope",
                "coverage": "complete",
                "article_fallback": "default",
                "article_reference": None,
                "definition_id": None,
                "blocks": [],
            }
            return httpx.Response(
                200,
                json={
                    "model": body["model"],
                    "usage": {"completion_tokens_details": {"reasoning_tokens": 7}},
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                (
                                    "reasoning"
                                    if body["model"].startswith("GigaChat/")
                                    else "reasoning_content"
                                ): "not part of the benchmark result",
                                "content": json.dumps(reply),
                            },
                        }
                    ],
                },
            )

        system = "Production prompt"
        payload = {
            "question": "Give me a recipe",
            "requested_mode": "auto",
            "locale": "en",
            "evidence": [],
            "page_evidence_ids": [],
        }
        snapshot = {
            "id": "test",
            "scope": None,
            "payload": payload,
            "evidence": [],
            "rubric": {"expected_mode": "out_of_scope", "private": "answer key"},
        }
        async with httpx.AsyncClient(transport=httpx.MockTransport(completion)) as http:
            for name in (DEEPSEEK, GIGACHAT):
                result = await generate(
                    system, snapshot, Candidate(name, 8192, "test"), http, 15
                )
                self.assertEqual(result["request_sha256"], digest([system, payload]))
                self.assertEqual(result["result"]["mode"], "out_of_scope")
                self.assertTrue(result["reasoning_observed"])
                self.assertNotIn("not part of the benchmark result", str(result))
        self.assertEqual(requests[0]["messages"], requests[1]["messages"])
        self.assertEqual(json.loads(requests[0]["messages"][1]["content"]), payload)
        self.assertEqual(requests[0]["max_tokens"], requests[1]["max_tokens"])
        self.assertEqual(requests[0]["reasoning_effort"], "max")
        self.assertEqual(requests[1]["reasoning_effort"], "high")
