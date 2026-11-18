"""Regression tests for the backend review findings (search, storage, assistant, limits, illustrations)."""

import asyncio
import hashlib
import json
import os
import struct
import tempfile
import time
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

import redis
from fastapi.testclient import TestClient

from app.assistant import service as assistant
from app.assistant.client import AssistantNotConfigured
from app.catalog import storage, vector
from app.main import create_app
from app.middleware.body_limit import AssistantBodyLimitMiddleware


class Limiter:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def check(self, identity, ip, category):
        if self.fail:
            raise redis.ConnectionError("redis is down")
        return 0

    def acquire_call(self, token):
        return True

    def release_call(self, token):
        pass


def record(
    number: int, title_en: str, title_ru: str, body_en: str, body_ru: str, categories=("Stars",)
) -> dict:
    return {
        "id": f"term:{number}",
        "kind": "term_page",
        "source": {
            "type": "astro4edu_glossary",
            "term_number": number,
            "url": f"https://astro4edu.org/resources/glossary/term/{number}/",
        },
        "metadata": {
            "categories": list(categories),
            "categories_ru": ["Звёзды"] * len(categories),
            "related_term_numbers": [],
        },
        "sections": [
            {
                "id": "definition",
                "type": "definition",
                "heading": "",
                "body": body_en,
                "media_ids": [],
                "links": [],
            }
        ],
        "locales": {
            "en": {
                "title": title_en,
                "aliases": [],
                "blocks": {"definition": {"heading": "", "body": body_en}},
            },
            "ru": {
                "title": title_ru,
                "aliases": [],
                "blocks": {"definition": {"heading": "", "body": body_ru}},
            },
        },
        "media": [],
        "review": {"status": "draft", "reviewer": None},
    }


def tiny_png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + b"\x10\x20\x30" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


class HttpRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "processed/terms").mkdir(parents=True)
        self.write(
            record(
                1,
                "Black Hole",
                "Чёрная дыра",
                "A region of space from which nothing escapes.",
                "Область пространства, из которой ничто не может вырваться.",
            )
        )
        self.write(
            record(
                2,
                "Stars",
                "Звёзды",
                "Hot balls of gas.",
                "Горячие газовые шары, светящиеся за счёт термоядерных реакций.",
            )
        )
        self.write(
            record(
                3,
                "Comet",
                "Комета",
                "An icy body.",
                "Ледяное тело, у которого возле Солнца появляется хвост.",
                categories=("Solar System",),
            )
        )
        self.enterContext(patch.multiple(storage, DATA=self.root, DB=self.root / "glossary.db", _checked={}))
        self.enterContext(patch.multiple(vector, DATA=self.root, _loaded=None, _failure={}))
        storage.rebuild_database()
        self.app = create_app(rate_limiter=Limiter())
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)

    def write(self, value: dict) -> None:
        (self.root / f"processed/terms/{value['source']['term_number']}.json").write_text(
            json.dumps(value, ensure_ascii=False), encoding="utf-8"
        )

    # B3: embeddings/model/index failures degrade to lexical search instead of HTTP 500.
    def test_search_falls_back_to_lexical_when_embeddings_fail(self):
        with patch.object(vector, "search", side_effect=ConnectionError("embedding server down")):
            response = self.client.get("/api/v1/search", params={"q": "дыра", "locale": "ru"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["mode"], "lexical")
        self.assertEqual(response.json()["items"][0]["record_id"], "term:1")

    # B7: case, «ё» and word forms do not hide Russian matches.
    def test_russian_search_ignores_case_yo_and_word_forms(self):
        self.assertEqual(self.client.get("/api/v1/terms", params={"q": "звезды"}).json()["total"], 1)
        self.assertEqual(self.client.get("/api/v1/terms", params={"q": "ЧЕРНАЯ"}).json()["total"], 1)
        found = self.client.get("/api/v1/search", params={"q": "черные дыры", "locale": "ru"}).json()
        self.assertEqual(found["items"][0]["record_id"], "term:1")
        star = self.client.get("/api/v1/search", params={"q": "что такое звезда", "locale": "ru"}).json()
        self.assertEqual(star["items"][0]["record_id"], "term:2")
        self.assertLess(star["total"], 3, "служебные слова не должны находить весь корпус")

    def test_letter_filter_and_sorting_use_russian_titles(self):
        letter = self.client.get("/api/v1/terms", params={"letter": "з"}).json()
        self.assertEqual([item["title"] for item in letter["items"]], ["Звёзды"])
        titles = [item["title"] for item in self.client.get("/api/v1/terms").json()["items"]]
        self.assertEqual(titles, ["Звёзды", "Комета", "Чёрная дыра"])
        facets = self.client.get("/api/v1/terms/facets").json()
        self.assertEqual([item["letter"] for item in facets["letters"]], ["З", "К", "Ч"])
        self.assertEqual(facets["total"], 3)

    # B6: changing canonical JSON rebuilds the SQLite database.
    def test_database_rebuilds_after_canonical_change(self):
        changed = record(3, "Comet", "Комета (исправлено)", "An icy body.", "Ледяное тело.")
        time.sleep(0.01)
        self.write(changed)
        storage.ensure_database(force_check=True)
        self.assertEqual(
            self.client.get("/api/v1/terms/3").json()["locales"]["ru"]["title"], "Комета (исправлено)"
        )

    # Reading the glossary does not depend on Redis; the assistant does.
    def test_reads_work_without_redis_but_assistant_is_closed(self):
        self.app.state.rate_limiter = Limiter(fail=True)
        self.assertEqual(self.client.get("/api/v1/terms").status_code, 200)
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "unit-test"}):
            response = self.client.post("/api/v1/assistant/answer", json={"question": "Что такое звезда?"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["Retry-After"], "10")

    # B11: wrong JSON shapes from an LLM are a 502, not an unhandled 500.
    def test_assistant_rejects_malformed_model_json(self):
        class Broken:
            async def complete(self, system, payload):
                return ["not", "an", "object"]

        self.app.state.llm_client = Broken()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "unit-test"}):
            response = self.client.post(
                "/api/v1/assistant/answer", json={"question": "Что такое чёрная дыра?"}
            )
        self.assertEqual(response.status_code, 502, response.text)
        self.assertEqual(response.json()["code"], "assistant_error")

    def test_null_record_ids_fall_back_without_crash(self):
        class NullIds:
            async def complete(self, system, payload):
                if "classify" in system:
                    return {"scope": "in_scope", "mode": "term", "record_ids": None}
                return {"answer": "Общий ответ.", "used_record_ids": None}

        self.app.state.llm_client = NullIds()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "unit-test"}):
            response = self.client.post(
                "/api/v1/assistant/answer", json={"question": "Что такое чёрная дыра?"}
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["grounding"], "model")
        self.assertEqual(response.json()["sources"], [])

    def test_exhausted_deepseek_balance_is_reported_as_unavailable(self):
        class NoBalance:
            async def complete(self, system, payload):
                raise AssistantNotConfigured("На аккаунте DeepSeek закончился баланс", 402)

        self.app.state.llm_client = NoBalance()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "unit-test"}):
            response = self.client.post(
                "/api/v1/assistant/answer", json={"question": "Что такое чёрная дыра?"}
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], "assistant_unavailable")
        self.assertIn("баланс", response.json()["message"])

    def test_general_mode_skips_retrieval(self):
        class General:
            async def complete(self, system, payload):
                if "classify" in system:
                    return {"scope": "in_scope", "mode": "general", "record_ids": []}
                return {"answer": "Ответ.", "used_record_ids": []}

        self.app.state.llm_client = General()
        with (
            patch.object(assistant, "candidates_for", side_effect=AssertionError("retrieval must not run")),
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "unit-test"}),
        ):
            response = self.client.post(
                "/api/v1/assistant/answer", json={"question": "Почему звёзды светят?", "mode": "general"}
            )
        self.assertEqual(response.status_code, 200, response.text)

    # Team illustrations: served only when generated, with metadata in the record and list.
    def test_illustration_contract(self):
        self.assertEqual(
            self.client.get("/api/v1/terms/1/illustration").json()["code"], "illustration_not_found"
        )
        image = tiny_png(800, 600)
        (self.root / "media/illustrations").mkdir(parents=True)
        (self.root / "media/illustrations/1.png").write_bytes(image)
        (self.root / "illustrations").mkdir()
        (self.root / "illustrations/manifest.json").write_text(
            json.dumps(
                {
                    "items": [
                        {
                            "term_number": 1,
                            "status": "generated",
                            "path": "media/illustrations/1.png",
                            "width": 800,
                            "height": 600,
                            "sha256": hashlib.sha256(image).hexdigest(),
                            "alt_ru": "Схема чёрной дыры",
                            "kind": "conceptual",
                            "generator": "test",
                        },
                        {"term_number": 2, "status": "pending", "path": None},
                    ]
                }
            ),
            encoding="utf-8",
        )
        storage.ensure_database(force_check=True)
        detail = self.client.get("/api/v1/terms/1").json()
        self.assertEqual(detail["illustration"]["url"], "/api/v1/terms/1/illustration")
        self.assertEqual(detail["illustration"]["alt_ru"], "Схема чёрной дыры")
        self.assertIsNone(self.client.get("/api/v1/terms/2").json()["illustration"])
        file = self.client.get("/api/v1/terms/1/illustration")
        self.assertEqual(file.status_code, 200)
        self.assertEqual(file.content, image)
        listed = {item["id"]: item for item in self.client.get("/api/v1/terms").json()["items"]}
        self.assertEqual(listed["term:1"]["illustration_url"], "/api/v1/terms/1/illustration")
        self.assertIsNone(listed["term:2"]["illustration_url"])

    def test_openapi_describes_binary_files_and_limits(self):
        schema = self.client.get("/openapi.json").json()
        media = schema["paths"]["/api/v1/media/{media_id}/file"]["get"]["responses"]
        self.assertIn("image/png", media["200"]["content"])
        self.assertIn("429", schema["paths"]["/api/v1/search"]["get"]["responses"])
        self.assertIn("mode", schema["components"]["schemas"]["SearchResponse"]["properties"])

    def test_assistant_body_limit_checks_header_and_actual_body(self):
        path = "/api/v1/assistant/answer"
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

    def test_assistant_body_without_length_is_bounded_and_replayed(self):
        scope = {"type": "http", "path": "/api/v1/assistant/answer", "headers": []}
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


if __name__ == "__main__":
    unittest.main()
