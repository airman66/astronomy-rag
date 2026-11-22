import os
from unittest.mock import patch

from app.api.routes import health
from tests.support.api import ApiCase
from tests.support.catalog import seed_star


class ReadinessTests(ApiCase):
    def test_degraded_and_required_hybrid_have_different_statuses(self):
        seed_star(self.root)
        with patch.object(health, "semantic_ready", return_value=False):
            with patch.dict(os.environ, {"REQUIRE_HYBRID": "0"}):
                degraded = self.client.get("/ready")
            with patch.dict(os.environ, {"REQUIRE_HYBRID": "1"}):
                unavailable = self.client.get("/ready")
        self.assertEqual(degraded.status_code, 200)
        self.assertEqual(degraded.json()["status"], "degraded")
        self.assertEqual(unavailable.status_code, 503)
        self.assertEqual(unavailable.json()["semantic"], "unavailable")
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertIn("astronomy_search_lexical_total", self.client.get("/metrics").text)
