import os
import time
from threading import Event
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.catalog import vector
from tests.support.api import ApiCase
from tests.support.catalog import seed_star


class ReadinessTests(ApiCase):
    def test_slow_embeddings_do_not_block_readiness(self):
        seed_star(self.root)
        started = Event()
        unblock = Event()
        finished = Event()

        def slow_embedding(*args, **kwargs):
            started.set()
            unblock.wait(5)
            finished.set()
            return []

        with (
            patch.object(vector, "search", side_effect=slow_embedding),
            patch.object(vector, "available", return_value=True),
            patch.dict(os.environ, {"REQUIRE_HYBRID": "0"}),
            TestClient(self.app) as client,
        ):
            try:
                self.assertTrue(started.wait(2))
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
