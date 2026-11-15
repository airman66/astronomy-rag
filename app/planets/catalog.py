"""Atomic catalog publication and compatible on-disk snapshots, off the ASGI loop."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from .scoring import (
    FIELDS,
    SCORE_VERSION,
    calculated,
    evaluate,
    measured,
    minimum_mass,
    rounded,
)
from .uncertainty import calculate

log = logging.getLogger(__name__)
NASA_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
COLUMNS = "pl_name,hostname,pl_rade,pl_radeerr1,pl_radeerr2,pl_bmasse,pl_bmasseerr1,pl_bmasseerr2,pl_bmassprov,pl_insol,pl_insolerr1,pl_insolerr2,st_teff,st_tefferr1,st_tefferr2,st_tefflim,st_lum,st_lumerr1,st_lumerr2,st_lumlim,pl_orbsmax,pl_orbsmaxerr1,pl_orbsmaxerr2,pl_orbsmaxlim,pl_orbeccen,pl_orbeccenerr1,pl_orbeccenerr2,pl_orbper,pl_orbper_reflink,pl_orbincl,pl_dens,pl_eqt,st_rad,pl_rade_reflink,pl_bmasse_reflink,pl_insol_reflink,pl_radelim,pl_bmasselim,pl_insollim,pl_orbeccenlim,pl_controv_flag,cb_flag,sy_dist,ra,dec,st_spectype,gaia_dr3_id,tic_id,st_age,st_rotp,pl_tsm,pl_esm,disc_year,discoverymethod,sy_pnum"


def now():
    return datetime.now(UTC)


def timestamp(value=None):
    return (value or now()).isoformat().replace("+00:00", "Z")


def read_saved(path):
    # .NET serialized the wrapper in PascalCase; Python also writes that format.
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def build_snapshot(planets, fetched_at, from_cache=False):
    evaluations = [(p, evaluate(p)) for p in planets]
    candidates = sorted(
        (e[0] for _, e in evaluations if e[0] is not None),
        key=lambda c: (-c["score"], c["name"]),
    )
    for rank, c in enumerate(candidates, 1):
        c["rank"] = rank
    calculate(planets, candidates)
    points = []
    for p, (c, exclusion, flux, provenance, zone) in evaluations:
        if not (p.get("pl_name") or "").strip():
            continue
        point = {
            k: p.get(FIELDS[k])
            for k in (
                "name",
                "hostName",
                "distanceParsec",
                "rightAscensionDegrees",
                "declinationDegrees",
                "discoveryYear",
                "discoveryMethod",
            )
        }
        point.update(
            radiusEarth=measured(p, "pl_rade"),
            radiusCalculated=calculated(p.get("pl_rade_reflink")),
            massEarth=measured(p, "pl_bmasse"),
            massMayBeMinimum=minimum_mass(p),
            fluxEarth=rounded(flux, 4) if flux is not None else None,
            fluxProvenance=provenance,
            stellarTemperatureK=measured(p, "st_teff"),
            habitableZone=zone,
            eligible=c is not None,
            exclusion=exclusion,
            score=c["score"] if c else None,
        )
        points.append(point)
    return {
        "fetchedAt": fetched_at,
        "totalPlanets": len(planets),
        "candidates": candidates,
        "points": points,
        "fromCache": from_cache,
    }


class Catalog:
    def __init__(self, data_dir=None):
        self.path = (
            Path(data_dir or os.getenv("CATALOG_DATA_DIR", "data/catalog"))
            / "nasa-planets.json"
        )
        self.current = None
        self.last_attempt = self.last_error = self.next_attempt = None

    def load_cache(self):
        try:
            saved = read_saved(self.path)
            if saved["Planets"]:
                self.current = build_snapshot(
                    saved["Planets"], saved["FetchedAt"], True
                )
        except FileNotFoundError:
            pass
        except Exception:
            log.exception("Could not load NASA cache")

    def refresh(self):
        self.last_attempt = timestamp()
        try:
            with httpx.Client(timeout=90, follow_redirects=True) as client:
                response = client.get(
                    NASA_URL,
                    params={
                        "query": f"select {COLUMNS} from pscomppars",
                        "format": "json",
                    },
                )
                response.raise_for_status()
                planets = response.json()
            if not isinstance(planets, list) or len(planets) < 1000:
                raise ValueError("NASA returned an unexpectedly small catalog")
            fetched = timestamp()
            snapshot = build_snapshot(planets, fetched)
            self.current = snapshot
            self.last_error = None
            try:
                save(self.path, {"FetchedAt": fetched, "Planets": planets})
            except OSError:
                log.exception("Fresh catalog loaded, but cache could not be saved")
            log.info(
                "Catalog ready: %s planets, %s candidates",
                len(planets),
                len(snapshot["candidates"]),
            )
            return True
        except Exception as exc:
            self.last_error = (
                "timeout"
                if isinstance(exc, httpx.TimeoutException)
                else f"http {exc.response.status_code}"
                if isinstance(exc, httpx.HTTPStatusError)
                else type(exc).__name__
            )
            log.warning(
                "NASA refresh failed (%s); keeping snapshot",
                self.last_error,
                exc_info=True,
            )
            return False

    async def run(self):
        await asyncio.to_thread(self.load_cache)
        failures = 0
        while True:
            success = await asyncio.to_thread(self.refresh)
            failures = 0 if success else failures + 1
            seconds = (
                21600
                if success
                else 1800
                if self.current
                else min(1800, 60 * 2 ** min(failures - 1, 5))
            )
            self.next_attempt = timestamp(now() + timedelta(seconds=seconds))
            await asyncio.sleep(seconds)

    def status(self):
        s = self.current
        return {
            "ready": s is not None,
            "fetchedAt": s["fetchedAt"] if s else None,
            "totalPlanets": s["totalPlanets"] if s else 0,
            "candidateCount": len(s["candidates"]) if s else 0,
            "source": "NASA Exoplanet Archive / pscomppars",
            "scoreVersion": SCORE_VERSION,
            "fromCache": s["fromCache"] if s else False,
            "lastAttemptAt": self.last_attempt,
            "refreshError": self.last_error,
            "nextAttemptAt": self.next_attempt,
            "refreshIntervalHours": 6,
        }
