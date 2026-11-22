import hashlib
import json
import struct
import zlib

from app.catalog import storage
from tests.support.api import ApiCase
from tests.support.catalog import seed_search_terms, seed_star


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


class SourceMediaTests(ApiCase):
    def setUp(self) -> None:
        super().setUp()
        self.image = seed_star(self.root)

    def test_media_accepts_project_and_data_relative_paths(self):
        path = self.root / "processed/terms/1.json"
        record = json.loads(path.read_text())
        for filename in ("data/media/original/example.png", "media/original/example.png"):
            record["media"][0]["original_path"] = filename
            path.write_text(json.dumps(record))
            storage.rebuild_database()
            response = self.client.get("/api/v1/media/media:example/file")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, self.image.read_bytes())


    def test_hashed_media_urls_are_immutable_and_validate_legacy_urls(self):
        path = self.root / "processed/terms/1.json"
        record = json.loads(path.read_text())
        digest = hashlib.sha256(self.image.read_bytes()).hexdigest()
        record["media"][0]["sha256"] = digest
        path.write_text(json.dumps(record))
        storage.rebuild_database()
        detail = self.client.get("/api/v1/media/media:example").json()
        for locale in ("ru", "en"):
            response = self.client.get(detail[f"file_url_{locale}"])
            self.assertEqual(response.status_code, 200)
            self.assertIn("immutable", response.headers["Cache-Control"])
        url = "/api/v1/media/media:example/file"
        response = self.client.get(url, headers={"If-None-Match": f'"{digest}"'})
        self.assertEqual(response.status_code, 304)
        self.assertIn("must-revalidate", response.headers["Cache-Control"])
        self.assertEqual(self.client.get(url, params={"v": "stale"}).status_code, 404)


class IllustrationTests(ApiCase):
    def setUp(self) -> None:
        super().setUp()
        seed_search_terms(self.root)

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
