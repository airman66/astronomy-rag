import asyncio
import unittest

from app.middleware.body_limit import AssistantBodyLimitMiddleware
from tests.support.api import ANSWER_PATH, ApiCase
from tests.support.catalog import seed_star


class RequestLimitTests(ApiCase):
    def test_rate_limit_shape(self):
        self.limiter.retry = 7
        response = self.client.post(ANSWER_PATH, json={"question": "Что такое звезда?"})
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["Retry-After"], "7")
        self.assertEqual(response.json()["code"], "rate_limited")

    def test_reads_work_without_redis_but_assistant_is_closed(self):
        seed_star(self.root)
        self.limiter.fail = True
        listing = self.client.get("/api/v1/terms")
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.json()["total"], 1)
        response = self.client.post(ANSWER_PATH, json={"question": "Что такое звезда?"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["Retry-After"], "10")

    def test_assistant_body_limit_checks_header_and_actual_body(self):
        path = ANSWER_PATH
        invalid = self.client.post(path, content=b"{}", headers={"Content-Length": "invalid"})
        self.assertEqual((invalid.status_code, invalid.json()["code"]), (400, "invalid_content_length"))
        oversized_header = self.client.post(path, content=b"{}", headers={"Content-Length": "16385"})
        self.assertEqual(
            (oversized_header.status_code, oversized_header.json()["code"]), (413, "body_too_large")
        )
        understated = self.client.post(path, content=b"x" * 16385, headers={"Content-Length": "1"})
        self.assertEqual((understated.status_code, understated.json()["code"]), (413, "body_too_large"))
        schema = self.client.get("/openapi.json").json()
        responses = schema["paths"][path]["post"]["responses"]
        self.assertIn("400", responses)
        self.assertIn("413", responses)


class BodyLimitMiddlewareTests(unittest.TestCase):
    def test_assistant_body_without_length_is_bounded_and_replayed(self):
        scope = {"type": "http", "path": ANSWER_PATH, "headers": []}
        observed = []

        async def endpoint(_scope, receive, send):
            observed.append((await receive())["body"])
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        async def run(parts):
            messages = iter(
                {"type": "http.request", "body": part, "more_body": i < len(parts) - 1}
                for i, part in enumerate(parts)
            )
            sent = []

            async def receive():
                return next(messages)

            async def send(message):
                sent.append(message)

            await AssistantBodyLimitMiddleware(endpoint)(scope, receive, send)
            return sent

        accepted = asyncio.run(run([b"abc", b"def"]))
        self.assertEqual(observed, [b"abcdef"])
        self.assertEqual(accepted[0]["status"], 200)
        rejected = asyncio.run(run([b"a" * 16000, b"b" * 385]))
        self.assertEqual(rejected[0]["status"], 413)
        self.assertEqual(observed, [b"abcdef"])
