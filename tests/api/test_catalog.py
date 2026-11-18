from app.catalog import storage
from tests.support.api import ApiCase
from tests.support.catalog import make_term, seed_search_terms, seed_star, write_term


class CatalogReadingTests(ApiCase):
    def setUp(self) -> None:
        super().setUp()
        seed_star(self.root)

    def test_read_routes_and_media(self):
        self.assertEqual(self.client.get("/api/v1/terms").json()["total"], 1)
        term = self.client.get("/api/v1/terms/1")
        self.assertEqual(term.status_code, 200)
        self.assertEqual(term.json()["locales"]["ru"]["title"], "Звезда класса A")
        self.assertEqual(self.client.get("/api/v1/articles").json()["total"], 0)
        self.assertEqual(self.client.get("/api/v1/media/media:example").json()["caption_ru"], "Звезда.")
        self.assertEqual(self.client.get("/api/v1/media/media:example/file").content, b"not-a-real-png")
        self.assertEqual(self.client.get("/api/v1/terms/999").json()["code"], "term_not_found")

    def test_large_pages_return_empty_items(self):
        page = 10**18
        for route, total in (("terms", 1), ("articles", 0)):
            with self.subTest(route=route):
                response = self.client.get(f"/api/v1/{route}", params={"page": page})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"items": [], "total": total, "page": page, "page_size": 20})

    def test_term_paths_reject_numbers_outside_sqlite_range(self):
        for number in (-(2**63) - 1, 2**63):
            for suffix in ("", "/illustration"):
                with self.subTest(number=number, suffix=suffix):
                    response = self.client.get(f"/api/v1/terms/{number}{suffix}")
                    self.assertEqual(response.status_code, 422)
                    self.assertEqual(response.json()["code"], "validation_error")


class CatalogFilterTests(ApiCase):
    def setUp(self) -> None:
        super().setUp()
        seed_search_terms(self.root)

    def test_letter_filter_and_sorting_use_russian_titles(self):
        letter = self.client.get("/api/v1/terms", params={"letter": "з"}).json()
        self.assertEqual([item["title"] for item in letter["items"]], ["Звёзды"])
        titles = [item["title"] for item in self.client.get("/api/v1/terms").json()["items"]]
        self.assertEqual(titles, ["Звёзды", "Комета", "Чёрная дыра"])
        facets = self.client.get("/api/v1/terms/facets").json()
        self.assertEqual([item["letter"] for item in facets["letters"]], ["З", "К", "Ч"])
        self.assertEqual(facets["total"], 3)

    def test_database_rebuilds_after_canonical_change(self):
        changed = make_term(3, "Comet", "Комета (исправлено)", "An icy body.", "Ледяное тело.")
        write_term(self.root, changed)
        storage.ensure_database(force_check=True)
        self.assertEqual(
            self.client.get("/api/v1/terms/3").json()["locales"]["ru"]["title"], "Комета (исправлено)"
        )
