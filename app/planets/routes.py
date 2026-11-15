"""Existing /api/* contracts, included in the shared FastAPI application."""

import json
from collections import Counter
from pathlib import Path
from typing import Generic, Literal, TypeVar

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .catalog import Catalog
from .models import Candidate, CatalogPoint
from .scoring import SCORE_VERSION, boundaries, rounded
from .sources import Enrichment, Hwc

router = APIRouter(tags=["Planets"])
catalog = Catalog()
hwc = Hwc()
enrichment = Enrichment()


class Problem(BaseModel):
    type: str = "about:blank"
    title: str
    status: int
    detail: str


class Status(BaseModel):
    ready: bool
    fetchedAt: str | None
    totalPlanets: int
    candidateCount: int
    source: str
    scoreVersion: str
    fromCache: bool
    lastAttemptAt: str | None
    refreshError: str | None
    nextAttemptAt: str | None
    refreshIntervalHours: float


class CandidateList(BaseModel):
    fetchedAt: str
    scoreVersion: str
    total: int
    items: list[Candidate]


class PlanetList(BaseModel):
    fetchedAt: str
    scoreVersion: str
    total: int
    exclusions: dict[str, int]
    items: list[CatalogPoint]


class HwcWorld(BaseModel):
    name: str
    category: int
    esi: float | None


class HwcResponse(BaseModel):
    available: bool
    fetchedAt: str | None
    catalogUpdatedAt: str
    source: str
    worlds: list[HwcWorld]


class GaiaInfo(BaseModel):
    sourceId: int
    parallaxMas: float | None
    parallaxErrorMas: float | None
    temperatureK: float | None
    luminositySolar: float | None
    luminosityLowerSolar: float | None
    luminosityUpperSolar: float | None
    radiusSolar: float | None


class TessInfo(BaseModel):
    observationCount: int
    sectors: list[int]


class SpectrumInfo(BaseModel):
    type: str
    facility: str | None
    instrument: str | None
    bibcode: str | None


T = TypeVar("T")


class SourceResult(BaseModel, Generic[T]):
    status: Literal["available", "not_found", "missing_id", "unavailable"]
    value: T | None


class Sources(BaseModel):
    gaia: SourceResult[GaiaInfo]
    tess: SourceResult[TessInfo]
    spectra: SourceResult[list[SpectrumInfo]]


def problem(status, detail):
    title = "Service Unavailable" if status == 503 else "Not Found"
    return JSONResponse(
        status_code=status,
        media_type="application/problem+json",
        content=Problem(title=title, status=status, detail=detail).model_dump(),
    )


ERRORS = {503: {"model": Problem}, 404: {"model": Problem}}


@router.get("/api/status", response_model=Status)
def status():
    return catalog.status()


@router.get(
    "/api/candidates",
    response_model=CandidateList,
    responses={503: {"model": Problem}, 400: {"description": "Invalid limit or zone"}},
)
def candidates(limit: str = "50", zone: str = "all"):
    try:
        count = int(limit)
    except ValueError:
        count = 0
    if not 1 <= count <= 1000:
        return JSONResponse(
            status_code=400, content={"error": "limit must be between 1 and 1000"}
        )
    if zone not in ("all", "conservative", "optimistic"):
        return JSONResponse(
            status_code=400,
            content={"error": "zone must be all, conservative, or optimistic"},
        )
    s = catalog.current
    if s is None:
        return problem(503, "Catalog has not loaded yet")
    return {
        "fetchedAt": s["fetchedAt"],
        "scoreVersion": SCORE_VERSION,
        "total": len(s["candidates"]),
        "items": [c for c in s["candidates"] if zone == "all" or c["zone"] == zone][
            :count
        ],
    }


def find(name):
    s = catalog.current
    if s is None:
        return problem(503, "Catalog has not loaded yet")
    return next(
        (c for c in s["candidates"] if c["name"].lower() == name.lower()), None
    ) or problem(404, "Candidate not found")


@router.get("/api/candidates/{name}", response_model=Candidate, responses=ERRORS)
def candidate(name: str):
    return find(name)


@router.get("/api/candidates/{name}/sources", response_model=Sources, responses=ERRORS)
async def sources(name: str):
    c = find(name)
    return c if isinstance(c, JSONResponse) else await enrichment.get(c)


@router.get(
    "/api/planets", response_model=PlanetList, responses={503: {"model": Problem}}
)
def planets():
    s = catalog.current
    if s is None:
        return problem(503, "Catalog has not loaded yet")
    return {
        "fetchedAt": s["fetchedAt"],
        "scoreVersion": SCORE_VERSION,
        "total": len(s["points"]),
        "exclusions": dict(Counter(p["exclusion"] or "eligible" for p in s["points"])),
        "items": s["points"],
    }


@router.get("/api/hwc", response_model=HwcResponse)
async def habitable_worlds():
    s = await hwc.get()
    return {
        "available": s is not None,
        "fetchedAt": s["fetchedAt"] if s else None,
        "catalogUpdatedAt": "2024-03-21",
        "source": "https://phl.upr.edu/hwc/data",
        "worlds": s["worlds"] if s else [],
    }


METHODOLOGY = json.loads(
    Path(__file__).with_name("methodology.json").read_text(encoding="utf-8")
)
METHODOLOGY["boundaries"] = [
    dict(teffK=t, **{k: rounded(v, 4) for k, v in boundaries(t).items()})
    for t in range(2600, 7201, 100)
]


@router.get("/api/methodology")
def methodology():
    return METHODOLOGY
