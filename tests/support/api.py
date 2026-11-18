"""HTTP fixtures with an isolated database and local service doubles."""

import os
from unittest import TestCase
from unittest.mock import patch

import redis
from fastapi.testclient import TestClient

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
        if "classify" in system:
            scope = "out_of_scope" if "борщ" in payload["question"].lower() else "in_scope"
            return {"scope": scope, "mode": "term", "record_ids": ["term:1"], "reason": "definition"}
        return {"answer": "Звезда класса A — горячая звезда.", "used_record_ids": ["term:1"]}


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
