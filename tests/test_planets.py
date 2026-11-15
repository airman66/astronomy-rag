"""Ported scientific checks plus HTTP/cache/source isolation regressions."""

import asyncio
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from app.planets import routes
from app.planets.catalog import Catalog, build_snapshot
from app.planets.models import Candidate
from app.planets.scoring import boundaries, evaluate, rounded
from app.planets.sources import Enrichment, Hwc, parse_hwc
from app.planets.uncertainty import DotNetRandom
from fastapi import FastAPI
from fastapi.testclient import TestClient

EARTH = {
    "pl_name": "Earth test",
    "st_teff": 5780,
    "pl_insol": 1,
    "pl_rade": 1,
    "pl_bmasse": 1,
    "pl_orbeccen": 0,
    "pl_orbper": 365.25,
    "pl_orbsmax": 1,
    "st_rad": 1,
    "ra": 217.4,
    "dec": -62.7,
    "sy_dist": 1.3,
}


class ScoringTests(unittest.TestCase):
    def test_matches_original_dotnet_reference(self):
        fixture = json.loads(
            (Path(__file__).parent / "fixtures" / "planets-score-1.1.json").read_text(
                encoding="utf-8"
            )
        )
        actual = build_snapshot(fixture["planets"], "reference")["candidates"]
        self.assertEqual(actual, fixture["expected"])

    def test_dotnet_midpoint_rounding(self):
        self.assertEqual(rounded(2.55, 1), 2.6)
        self.assertEqual(rounded(2.65, 1), 2.6)

    def test_earth_and_coordinates(self):
        c = evaluate(EARTH)[0]
        self.assertEqual(c["score"], 100)
        self.assertEqual(
            c["score"], c["habitableZonePoints"] + c["sizePoints"] + c["orbitPoints"]
        )
        self.assertEqual(
            (c["conservativeInnerFlux"], c["conservativeOuterFlux"]), (1.107, 0.356)
        )
        self.assertEqual(c["orbitalPeriodDays"], 365.25)
        self.assertEqual(c["rightAscensionDegrees"], 217.4)
        self.assertEqual(c["declinationDegrees"], -62.7)
        Candidate.model_validate(c)

    def test_unknown_orbit_and_contradictory_mass(self):
        c = evaluate(dict(EARTH, pl_orbeccen=None))[0]
        self.assertEqual(c["score"], 92.5)
        self.assertIn("eccentricityUnknownOrLimited", c["dataFlags"])
        self.assertEqual(evaluate(dict(EARTH, pl_bmasse=15))[1], "massOutOfRange")

    def test_derived_flux_and_limits(self):
        p = dict(EARTH, pl_insol=None, st_lum=-1, pl_orbsmax=math.sqrt(0.1))
        self.assertEqual(evaluate(p)[0]["zone"], "conservative")
        for column, reason in (
            ("st_tefflim", "starTemperatureUnknown"),
            ("st_lumlim", "fluxUnknown"),
            ("pl_orbsmaxlim", "fluxUnknown"),
        ):
            with self.subTest(column=column):
                self.assertEqual(evaluate(dict(p, **{column: 1}))[1], reason)

    def test_funnel(self):
        for changes, reason, zone in [
            ({"pl_insol": 2}, "outsideHabitableZone", "outside"),
            ({"pl_rade": 4}, "radiusOutOfRange", "conservative"),
            ({"pl_rade": None, "pl_bmasse": None}, "sizeUnknown", "conservative"),
            ({"st_teff": 2000}, "starTemperatureOutOfRange", None),
            ({"pl_controv_flag": 1, "cb_flag": 1}, "controversial", None),
            ({"cb_flag": 1}, "circumbinary", None),
        ]:
            with self.subTest(reason=reason):
                e = evaluate(dict(EARTH, **changes))
                self.assertEqual((e[1], e[4]), (reason, zone))

    def test_boundaries_ordered(self):
        for temperature in range(2600, 7201, 50):
            b = boundaries(temperature)
            self.assertTrue(
                b["recentVenus"]
                > b["runawayGreenhouse"]
                > b["maximumGreenhouse"]
                > b["earlyMars"]
                > 0
            )
        self.assertLess(
            boundaries(3300)["runawayGreenhouse"], boundaries(5780)["runawayGreenhouse"]
        )

    def test_seeded_random_matches_system_random(self):
        rng = DotNetRandom(1)
        # Known System.Random(1).NextDouble sequence.
        self.assertAlmostEqual(rng.next_double(), 0.24866858415709278, places=15)
        self.assertAlmostEqual(rng.next_double(), 0.11074397718102856, places=15)
        self.assertAlmostEqual(rng.next_double(), 0.46701067987224587, places=15)

    def test_stability_coverage_reproducible_and_catalog_wide(self):
        p = dict(
            EARTH,
            pl_rade=1.2,
            pl_radeerr1=0.15,
            pl_radeerr2=-0.15,
            pl_insolerr1=0.2,
            pl_insolerr2=-0.2,
            st_tefferr1=100,
            st_tefferr2=-100,
        )
        a = build_snapshot([p], "test")
        self.assertEqual(a, build_snapshot([p], "test"))
        s = a["candidates"][0]["stability"]
        self.assertEqual(
            (s["samples"], s["inputsWithErrors"], s["scoredInputs"]), (160, 3, 5)
        )
        self.assertLess(s["scoreP10"], s["scoreP90"])
        self.assertTrue(0 < s["eligiblePercent"] <= 100)
        # A planet excluded at baseline can enter and displace a baseline candidate.
        borderline = dict(
            EARTH,
            pl_name="New entrant",
            pl_insol=1.78,
            pl_insolerr1=0.3,
            pl_insolerr2=-0.8,
        )
        baseline = dict(EARTH, pl_name="Baseline", pl_rade=2)
        sampled = build_snapshot([baseline, borderline], "test")["candidates"][0][
            "stability"
        ]
        self.assertGreater(sampled["rankP90"], 1)

    def test_hwc_quotes_and_bad_values(self):
        worlds = parse_hwc(
            '\ufeffP_NAME,P_HABITABLE,P_ESI,COMMENT\n"TRAPPIST-1 e",1,0.85,"quoted, field"\nExample b,2,nan,"line one\nline two"\nHot b,0,0.1,no\n'
        )
        self.assertEqual(
            worlds,
            [
                {"name": "TRAPPIST-1 e", "category": 1, "esi": 0.85},
                {"name": "Example b", "category": 2, "esi": None},
            ],
        )


class CatalogTests(unittest.TestCase):
    def test_refresh_backoff_with_and_without_snapshot(self):
        async def run(ready, success, attempts):
            catalog = Catalog()
            catalog.current = {"fetchedAt": "test"} if ready else None
            delays = []

            async def sleep(seconds):
                delays.append(seconds)
                if len(delays) == attempts:
                    raise asyncio.CancelledError

            with (
                patch.object(catalog, "load_cache"),
                patch.object(catalog, "refresh", return_value=success),
                patch("app.planets.catalog.asyncio.sleep", side_effect=sleep),
                self.assertRaises(asyncio.CancelledError),
            ):
                await catalog.run()
            self.assertIsNotNone(catalog.next_attempt)
            return delays

        self.assertEqual(
            asyncio.run(run(False, False, 6)), [60, 120, 240, 480, 960, 1800]
        )
        self.assertEqual(asyncio.run(run(True, False, 1)), [1800])
        self.assertEqual(asyncio.run(run(False, True, 1)), [21600])

    def test_legacy_snapshot_and_refresh_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nasa-planets.json"
            path.write_text(
                json.dumps({"FetchedAt": "2026-11-14T00:00:00Z", "Planets": [EARTH]})
            )
            catalog = Catalog(directory)
            catalog.load_cache()
            self.assertTrue(catalog.status()["fromCache"])
            snapshot = catalog.current
            with patch(
                "app.planets.catalog.httpx.Client.get",
                side_effect=httpx.ReadTimeout("simulated"),
            ):
                self.assertFalse(catalog.refresh())
            self.assertIs(catalog.current, snapshot)
            self.assertEqual(catalog.last_error, "timeout")
            response = httpx.Response(
                503, request=httpx.Request("GET", "https://nasa.test")
            )
            with patch("app.planets.catalog.httpx.Client.get", return_value=response):
                self.assertFalse(catalog.refresh())
            self.assertEqual(catalog.last_error, "http 503")
            self.assertEqual(json.loads(path.read_text())["Planets"], [EARTH])

    def test_success_saves_compatible_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            # Many excluded rows keep this test quick while meeting the NASA size guard.
            data = [EARTH] + [
                {"pl_name": f"Hot {i}", "st_teff": 9000} for i in range(999)
            ]
            response = httpx.Response(
                200, json=data, request=httpx.Request("GET", "https://nasa.test")
            )
            c = Catalog(directory)
            with patch("app.planets.catalog.httpx.Client.get", return_value=response):
                self.assertTrue(c.refresh())
            self.assertFalse(c.status()["fromCache"])
            self.assertEqual(len(json.loads(c.path.read_text())["Planets"]), 1000)
            replacement = Catalog(directory)
            replacement.load_cache()
            self.assertEqual(replacement.current["candidates"], c.current["candidates"])

    def test_hwc_legacy_cache_survives_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hwc.json"
            path.write_text(
                json.dumps(
                    {
                        "FetchedAt": "old",
                        "Worlds": [{"Name": "Cached b", "Category": 1, "Esi": 0.8}],
                    }
                )
            )
            service = Hwc(directory)
            with patch(
                "app.planets.sources.httpx.AsyncClient.get",
                side_effect=httpx.ReadTimeout("timeout"),
            ):
                saved = asyncio.run(service.get())
            self.assertEqual(saved["worlds"][0]["name"], "Cached b")


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.previous = routes.catalog.current
        routes.catalog.current = None
        app = FastAPI()
        app.include_router(routes.router)
        self.client = TestClient(app)

    def tearDown(self):
        routes.catalog.current = self.previous

    def test_cold_start_and_validation(self):
        self.assertFalse(self.client.get("/api/status").json()["ready"])
        for url in (
            "/api/candidates",
            "/api/candidates/Earth",
            "/api/candidates/Earth/sources",
            "/api/planets",
        ):
            r = self.client.get(url)
            self.assertEqual(r.status_code, 503)
            self.assertEqual(r.json()["detail"], "Catalog has not loaded yet")
            self.assertIn("application/problem+json", r.headers["content-type"])
        for query in ("limit=0", "limit=1001", "limit=bad", "zone=bad"):
            self.assertEqual(
                self.client.get("/api/candidates?" + query).status_code, 400
            )

    def test_loaded_contracts(self):
        routes.catalog.current = build_snapshot(
            [EARTH, dict(EARTH, pl_name="Hot", pl_insol=2)], "date"
        )
        self.assertEqual(self.client.get("/api/candidates").json()["total"], 1)
        c = self.client.get("/api/candidates/earth%20TEST").json()
        self.assertEqual(c["score"], 100)
        self.assertIn("dataFlags", c)
        self.assertEqual(c["rank"], 1)
        self.assertEqual(c["stability"]["samples"], 160)
        self.assertEqual(self.client.get("/api/candidates/unknown").status_code, 404)
        self.assertEqual(
            self.client.get("/api/candidates?zone=optimistic").json()["items"], []
        )
        all_planets = self.client.get("/api/planets").json()
        self.assertEqual(
            all_planets["exclusions"], {"eligible": 1, "outsideHabitableZone": 1}
        )
        self.assertEqual(len(all_planets["items"]), 2)
        method = self.client.get("/api/methodology").json()
        self.assertEqual(len(method["boundaries"]), 47)
        self.assertEqual(method["scoreVersion"], "1.1")
        self.assertIn(
            "/api/candidates/{name}/sources",
            self.client.get("/openapi.json").json()["paths"],
        )


class SourceTests(unittest.TestCase):
    def test_isolated_failure_and_cache(self):
        calls = []

        def handler(request):
            calls.append(request)
            if "gaia" in request.url.host:
                return httpx.Response(503)
            if "mast" in request.url.host:
                return httpx.Response(
                    200,
                    json={
                        "status": "COMPLETE",
                        "paging": {"rowsFiltered": 3},
                        "data": [
                            {"sequence_number": 5},
                            {"sequence_number": 2},
                            {"sequence_number": 5},
                        ],
                    },
                )
            return httpx.Response(
                200,
                json=[
                    {
                        "spec_type": "Transmission",
                        "facility": "JWST",
                        "instrument": "NIRSpec",
                        "bibcode": "paper",
                    }
                ],
            )

        async def run():
            service = Enrichment()
            # Patch only the constructor; preserve real httpx request/response parsing.
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            with patch("app.planets.sources.httpx.AsyncClient", return_value=client):
                p = {
                    "name": "O'Brien b",
                    "gaiaDr3Id": "Gaia DR3 123",
                    "ticId": "TIC 456",
                }
                a = await service.get(p)
                b = await service.get(p)
            self.assertEqual(a, b)
            self.assertEqual(len(calls), 3)
            self.assertEqual(a["gaia"]["status"], "unavailable")
            self.assertEqual(
                a["tess"]["value"], {"observationCount": 3, "sectors": [2, 5]}
            )
            self.assertEqual(a["spectra"]["status"], "available")
            self.assertIn("O''Brien b", calls[2].url.params["query"])

        asyncio.run(run())

    def test_gaia_positional_tap_json(self):
        async def run():
            def handler(request):
                return httpx.Response(
                    200,
                    json={
                        "metadata": [{"name": "parallax"}, {"name": "teff_gspphot"}],
                        "data": [[12.3, 4000]],
                    },
                )

            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                service = Enrichment()
                self.assertEqual(
                    (await service.gaia(client, None))["status"], "missing_id"
                )
                value = (await service.gaia(client, "Gaia DR3 123"))["value"]
                self.assertEqual(value["parallaxMas"], 12.3)
                self.assertEqual(value["temperatureK"], 4000)

        asyncio.run(run())
