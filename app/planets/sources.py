"""Independent HWC, Gaia, TESS and spectrum lookups; never change the ranking."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import os
import time
from pathlib import Path

import httpx

from .catalog import NASA_URL, read_saved, save, timestamp
from .scoring import finite

log = logging.getLogger(__name__)


def parse_hwc(text):
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    reader.fieldnames = [name.strip().upper() for name in (reader.fieldnames or [])]
    if not {"P_NAME", "P_HABITABLE"}.issubset(reader.fieldnames):
        raise ValueError("HWC CSV lacks P_NAME or P_HABITABLE")
    worlds = []
    for row in reader:
        try:
            name, category = (
                (row.get("P_NAME") or "").strip(),
                int(row.get("P_HABITABLE", "")),
            )
        except (ValueError, TypeError):
            continue
        if not name or category not in (1, 2):
            continue
        try:
            esi = float(row.get("P_ESI", ""))
        except (ValueError, TypeError):
            esi = None
        worlds.append(
            {"name": name, "category": category, "esi": esi if finite(esi) else None}
        )
    return worlds


class Hwc:
    def __init__(self, data_dir=None):
        self.path = (
            Path(data_dir or os.getenv("CATALOG_DATA_DIR", "data/catalog")) / "hwc.json"
        )
        self.current = None
        self.next_attempt = 0
        self.lock = asyncio.Lock()

    async def get(self):
        async with self.lock:
            if time.monotonic() < self.next_attempt:
                return self.current
            if self.current is None:
                try:
                    saved = await asyncio.to_thread(read_saved, self.path)
                    self.current = {
                        "fetchedAt": saved["FetchedAt"],
                        "worlds": [
                            {
                                "name": w["Name"],
                                "category": w["Category"],
                                "esi": w.get("Esi"),
                            }
                            for w in saved["Worlds"]
                        ],
                    }
                except FileNotFoundError:
                    pass
                except Exception:
                    log.exception("Could not load HWC cache")
            try:
                async with httpx.AsyncClient(
                    timeout=25, follow_redirects=True
                ) as client:
                    response = await client.get(
                        "https://www.hpcf.upr.edu/~abel/phl/hwc/data/hwc.csv"
                    )
                    response.raise_for_status()
                worlds = parse_hwc(response.text)
                if len(worlds) < 20:
                    raise ValueError("HWC returned too few habitable worlds")
                fetched = timestamp()
                self.current = {"fetchedAt": fetched, "worlds": worlds}
                self.next_attempt = time.monotonic() + 86400
                await asyncio.to_thread(
                    save,
                    self.path,
                    {
                        "FetchedAt": fetched,
                        "Worlds": [
                            {
                                "Name": w["name"],
                                "Category": w["category"],
                                "Esi": w["esi"],
                            }
                            for w in worlds
                        ],
                    },
                )
            except Exception:
                log.warning("HWC refresh failed; keeping snapshot", exc_info=True)
                self.next_attempt = time.monotonic() + 900
            return self.current


def result(status, value=None):
    return {"status": status, "value": value}


class Enrichment:
    def __init__(self):
        self.cache = {}

    async def get(self, candidate):
        name = candidate["name"]
        cached = self.cache.get(name)
        if cached and cached[0] > time.monotonic():
            return cached[1]
        async with httpx.AsyncClient(timeout=18, follow_redirects=True) as client:
            values = await asyncio.gather(
                self.gaia(client, candidate.get("gaiaDr3Id")),
                self.tess(client, candidate.get("ticId")),
                self.spectra(client, name),
            )
        payload = dict(zip(("gaia", "tess", "spectra"), values))
        self.cache[name] = (time.monotonic() + 900, payload)
        return payload

    @staticmethod
    async def read(client, url, params):
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    async def gaia(self, client, identifier):
        if (
            not identifier
            or not identifier.startswith("Gaia DR3 ")
            or not identifier[9:].isdigit()
            or int(identifier[9:]) <= 0
        ):
            return result("missing_id")
        source_id = int(identifier[9:])
        columns = "g.source_id,g.parallax,g.parallax_error,a.teff_gspphot,a.lum_flame,a.lum_flame_lower,a.lum_flame_upper,a.radius_flame"
        query = f"SELECT TOP 1 {columns} FROM gaiadr3.gaia_source AS g LEFT OUTER JOIN gaiadr3.astrophysical_parameters AS a ON g.source_id=a.source_id WHERE g.source_id={source_id}"
        try:
            data = await self.read(
                client,
                "https://gaia.ari.uni-heidelberg.de/tap/sync",
                {
                    "REQUEST": "doQuery",
                    "LANG": "ADQL",
                    "FORMAT": "json",
                    "QUERY": query,
                },
            )
            if not data["data"]:
                return result("not_found")
            row = data["data"][0]
            # TAP JSON commonly supplies positional rows with a metadata column list.
            if isinstance(row, list):
                row = dict(zip((c["name"] for c in data["metadata"]), row))
            mapping = {
                "parallaxMas": "parallax",
                "parallaxErrorMas": "parallax_error",
                "temperatureK": "teff_gspphot",
                "luminositySolar": "lum_flame",
                "luminosityLowerSolar": "lum_flame_lower",
                "luminosityUpperSolar": "lum_flame_upper",
                "radiusSolar": "radius_flame",
            }
            return result(
                "available",
                dict(
                    sourceId=source_id,
                    **{
                        k: row.get(v) if finite(row.get(v)) else None
                        for k, v in mapping.items()
                    },
                ),
            )
        except Exception:
            log.warning("Gaia lookup failed for %s", source_id, exc_info=True)
            return result("unavailable")

    async def tess(self, client, identifier):
        if (
            not identifier
            or not identifier.startswith("TIC ")
            or not identifier[4:].isdigit()
            or int(identifier[4:]) <= 0
        ):
            return result("missing_id")
        request = {
            "service": "Mast.Caom.Filtered",
            "params": {
                "columns": "obs_id,sequence_number,dataproduct_type",
                "filters": [
                    {"paramName": "obs_collection", "values": ["TESS"]},
                    {"paramName": "target_name", "values": [str(int(identifier[4:]))]},
                    {"paramName": "dataproduct_type", "values": ["timeseries"]},
                ],
            },
            "format": "json",
            "pagesize": 100,
        }
        try:
            data = await self.read(
                client,
                "https://mast.stsci.edu/api/v0/invoke",
                {"request": json.dumps(request)},
            )
            if data["status"] != "COMPLETE":
                return result("unavailable")
            if not data["data"]:
                return result("not_found")
            sectors = sorted(
                {
                    int(r["sequence_number"])
                    for r in data["data"]
                    if finite(r.get("sequence_number")) and r["sequence_number"] > 0
                }
            )
            return result(
                "available",
                {
                    "observationCount": data["paging"]["rowsFiltered"],
                    "sectors": sectors,
                },
            )
        except Exception:
            log.warning("MAST lookup failed for %s", identifier, exc_info=True)
            return result("unavailable")

    async def spectra(self, client, name):
        query = (
            "select distinct top 8 spec_type,facility,instrument,bibcode from spectra where pl_name='"
            + name.replace("'", "''")
            + "'"
        )
        try:
            rows = await self.read(client, NASA_URL, {"query": query, "format": "json"})
            return (
                result(
                    "available",
                    [
                        {
                            "type": r.get("spec_type") or "Unknown",
                            "facility": r.get("facility"),
                            "instrument": r.get("instrument"),
                            "bibcode": r.get("bibcode"),
                        }
                        for r in rows
                    ],
                )
                if rows
                else result("not_found")
            )
        except Exception:
            log.warning("NASA spectra lookup failed for %s", name, exc_info=True)
            return result("unavailable")
