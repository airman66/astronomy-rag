"""HTTP fixtures with an isolated database and local service doubles."""

import os
from unittest import TestCase
from unittest.mock import patch

import redis
from fastapi.testclient import TestClient

from app.assistant.client import Completion
from app.main import create_app
from tests.support.catalog import use_catalog

ANSWER_PATH = "/api/v1/assistant/answer"


class FakeLimiter:
    def __init__(self) -> None:
        self.retry = 0
        self.fail = False

    def check(self, identity, ip, category):
        if self.fail:
            raise redis.ConnectionError("redis is down")
        return self.retry

    def acquire_call(self, token):
        return True

    def release_call(self, token):
        pass


class FakeModel:
    async def complete(self, system, payload):
        evidence = payload["evidence"]
        requested = payload["requested_mode"]
        if "борщ" in payload["question"].lower():
            return Completion({"mode": "out_of_scope", "coverage": "complete", "article_fallback": "default", "definition_id": None, "blocks": []},
                              "test-model")
        if requested == "general":
            return Completion({"mode": "general", "coverage": "complete", "article_fallback": "default", "definition_id": None,
                               "blocks": [{"text": "Гравитация удерживает планету на орбите.", "evidence_ids": []}]},
                              "test-model")
        definition = next((item["id"] for item in evidence if item["definition"]), None)
        return Completion({"mode": "term", "coverage": "complete", "article_fallback": "default", "definition_id": definition, "blocks": []},
                          "test-model")


class ApiCase(TestCase):
    def setUp(self) -> None:
        self.root = use_catalog(self)
        self.enterContext(
            patch.dict(os.environ, {"ASSISTANT_PROVIDER": "deepseek", "DEEPSEEK_API_KEY": "unit-test"})
        )
        self.limiter = FakeLimiter()
        self.app = create_app(rate_limiter=self.limiter, llm_client=FakeModel())
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)
