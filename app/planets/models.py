"""Public Planets response contracts (camelCase retained for existing clients)."""

from pydantic import BaseModel


class RankStability(BaseModel):
    samples: int
    inputsWithErrors: int
    scoredInputs: int
    scoreP10: float
    scoreP90: float
    rankP10: int
    rankP90: int
    top50Percent: float
    eligiblePercent: float


class Candidate(BaseModel):
    name: str
    hostName: str | None
    score: float
    habitableZonePoints: float
    sizePoints: float
    orbitPoints: float
    zone: str
    radiusEarth: float | None
    massEarth: float | None
    fluxEarth: float
    stellarTemperatureK: float
    eccentricity: float | None
    conservativeOuterFlux: float
    conservativeInnerFlux: float
    optimisticOuterFlux: float
    optimisticInnerFlux: float
    distanceParsec: float | None
    spectralType: str | None
    gaiaDr3Id: str | None
    ticId: str | None
    starAgeGyr: float | None
    starRotationDays: float | None
    transmissionSpectroscopyMetric: float | None
    emissionSpectroscopyMetric: float | None
    dataFlags: list[str]
    semiMajorAxisAu: float | None
    orbitalPeriodDays: float | None
    inclinationDegrees: float | None
    densityGramsCm3: float | None
    equilibriumTemperatureK: float | None
    stellarRadiusSolar: float | None
    rightAscensionDegrees: float | None
    declinationDegrees: float | None
    orbitalPeriodProvenance: str | None
    fluxProvenance: str | None
    discoveryYear: int | None
    discoveryMethod: str | None
    systemPlanetCount: int | None
    rank: int
    stability: RankStability | None


class CatalogPoint(BaseModel):
    name: str
    hostName: str | None
    radiusEarth: float | None
    radiusCalculated: bool
    massEarth: float | None
    massMayBeMinimum: bool
    fluxEarth: float | None
    fluxProvenance: str | None
    stellarTemperatureK: float | None
    distanceParsec: float | None
    rightAscensionDegrees: float | None
    declinationDegrees: float | None
    discoveryYear: int | None
    discoveryMethod: str | None
    habitableZone: str | None
    eligible: bool
    exclusion: str | None
    score: float | None
