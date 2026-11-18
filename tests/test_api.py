import unittest

from fastapi.testclient import TestClient

from app.main import app


class ApiSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "service": "glossary-rag"})

    def test_method_not_allowed_keeps_allow_header(self) -> None:
        response = self.client.post("/health")
        self.assertEqual(response.status_code, 405)
        self.assertEqual(response.headers["Allow"], "GET")

    def test_openapi_and_swagger(self) -> None:
        schema = self.client.get("/openapi.json")
        self.assertEqual(schema.status_code, 200)
        self.assertIn("/health", schema.json()["paths"])

        docs = self.client.get("/docs")
        self.assertEqual(docs.status_code, 200)
        self.assertIn("Swagger UI", docs.text)
