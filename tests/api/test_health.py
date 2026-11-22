import os
import sqlite3
import time
from contextlib import closing
from threading import Event
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app.api.routes.health import SemanticReadiness
from app.catalog import storage, vector
from app.catalog.corpus import corpus_digest
from app.catalog.index import MODEL
from tests.catalog.fixtures import bind_vector_index, seed_vector_index
from tests.support.api import ANSWER_PATH, ApiCase
from tests.support.catalog import seed_star


class ReadinessTests(ApiCase):
    def test_wrong_embedding_model_fails_readiness_and_search_falls_back(self):
        seed_star(self.root)
        seed_vector_index(self.root)
        bind_vector_index(self.root, corpus_digest(self.root))
        probe = SemanticReadiness()
        self.app.state.semantic_readiness = probe

        for identity in ("other-model", None):
            stop = Mock(spec=Event)
            stop.is_set.side_effect = (False, True)
            embedding = Mock()
            payload: dict[str, object] = {"data": [{"embedding": [1.0, 0.0]}]}
            if identity is not None:
                payload["model"] = identity
            embedding.json.return_value = payload
            with (
                self.subTest(model=identity),
                patch("requests.post", return_value=embedding),
                patch.dict(os.environ, {"REQUIRE_HYBRID": "1"}),
            ):
                probe._check(stop)
                self.assertEqual(self.client.get("/ready").status_code, 503)
                response = self.client.get("/api/v1/search", params={"q": "звезда"})
                self.assertEqual(response.json()["mode"], "lexical")

    def test_catalogue_with_metadata_but_without_fts_is_unavailable(self):
        seed_star(self.root)
        self.assertEqual(self.client.get("/ready").status_code, 200)
        with closing(sqlite3.connect(storage.DB)) as db:
            db.execute("DROP TABLE search_fts")
        with patch.object(storage, "READ_ONLY", True):
            response = self.client.get("/ready")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["catalogue"], "unavailable")

    def test_slow_embeddings_do_not_block_readiness(self):
        seed_star(self.root)
        started = Event()
        unblock = Event()
        finished = Event()
        timeouts = []

        def slow_embedding(*args, timeout=vector.QUERY_TIMEOUT, **kwargs):
            timeouts.append(timeout)
            started.set()
            unblock.wait(5)
            finished.set()
            return []

        with (
            patch.object(vector, "search", side_effect=slow_embedding),
            patch.dict(os.environ, {"REQUIRE_HYBRID": "0"}),
            TestClient(self.app) as client,
        ):
            try:
                self.assertTrue(started.wait(2))
                self.assertEqual(timeouts, [vector.QUERY_TIMEOUT])
                degraded = client.get("/ready")
                self.assertEqual(degraded.status_code, 200)
                self.assertEqual(degraded.json()["status"], "degraded")
                with patch.dict(os.environ, {"REQUIRE_HYBRID": "1"}):
                    unavailable = client.get("/ready")
                    self.assertEqual(unavailable.status_code, 503)
                    self.assertEqual(unavailable.json()["semantic"], "unavailable")
                self.assertFalse(finished.is_set())
            finally:
                unblock.set()

            with patch.dict(os.environ, {"REQUIRE_HYBRID": "1"}):
                ready = client.get("/ready")
                deadline = time.monotonic() + 2
                while ready.status_code != 200 and time.monotonic() < deadline:
                    time.sleep(0.01)
                    ready = client.get("/ready")
            self.assertEqual(ready.status_code, 200)
            self.assertEqual(ready.json()["semantic"], "ready")
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertIn("astronomy_search_lexical_total", client.get("/metrics").text)
            with patch.dict(os.environ, {"REQUIRE_HYBRID": "1"}), patch.object(
                storage, "corpus_version", return_value="replacement",
            ):
                self.assertEqual(client.get("/ready").status_code, 503)

    def test_http_search_and_assistant_can_serve_while_index_load_is_blocked(self):
        seed_star(self.root)
        seed_vector_index(self.root)
        bind_vector_index(self.root, corpus_digest(self.root))
        started, unblock = Event(), Event()
        original = vector.load_index

        def slow_load(*args):
            started.set()
            if not unblock.wait(5):
                raise AssertionError("Test did not release index loader")
            return original(*args)

        embedding = Mock()
        embedding.json.return_value = {"model": MODEL, "data": [{"embedding": [1.0, 0.0]}]}
        with (
            patch.object(vector, "load_index", side_effect=slow_load),
            patch("requests.post", return_value=embedding),
            TestClient(self.app) as client,
        ):
            try:
                self.assertTrue(started.wait(2))
                response = client.get("/api/v1/search", params={"q": "звезда"})
                self.assertEqual(response.json()["mode"], "lexical")
                answer = client.post(ANSWER_PATH, json={"question": "Что такое звезда?", "locale": "ru"})
                self.assertEqual(answer.status_code, 200, answer.text)
                self.assertFalse(unblock.is_set())
                self.assertIsNone(vector.loaded_generation())
            finally:
                unblock.set()
            deadline = time.monotonic() + 2
            while vector.loaded_generation() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIsNotNone(vector.loaded_generation())
