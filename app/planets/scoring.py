"""Port of Planets.Api score 1.1; bounds are never treated as measurements."""

from __future__ import annotations

import math

SCORE_VERSION = "1.1"


def rounded(value, digits):
    """Math.Round(double, digits): scale first, then round ties to even."""
    scale = 10**digits
    return round(value * scale) / scale if abs(value) < 1e16 else value


COEFFICIENTS = {
    "recentVenus": (1.776, 2.136e-4, 2.533e-8, -1.332e-11, -3.097e-15),
    "runawayGreenhouse": (1.107, 1.332e-4, 1.58e-8, -8.308e-12, -1.931e-15),
    "maximumGreenhouse": (0.356, 6.171e-5, 1.698e-9, -3.198e-12, -5.575e-16),
    "earlyMars": (0.320, 5.547e-5, 1.526e-9, -2.874e-12, -5.011e-16),
}
# Names used by the public API, mapped to the unchanged NASA/cache columns.
FIELDS = {
    "name": "pl_name",
    "hostName": "hostname",
    "radiusEarth": "pl_rade",
    "massEarth": "pl_bmasse",
    "fluxEarth": "pl_insol",
    "stellarTemperatureK": "st_teff",
    "eccentricity": "pl_orbeccen",
    "distanceParsec": "sy_dist",
    "spectralType": "st_spectype",
    "gaiaDr3Id": "gaia_dr3_id",
    "ticId": "tic_id",
    "starAgeGyr": "st_age",
    "starRotationDays": "st_rotp",
    "transmissionSpectroscopyMetric": "pl_tsm",
    "emissionSpectroscopyMetric": "pl_esm",
    "semiMajorAxisAu": "pl_orbsmax",
    "orbitalPeriodDays": "pl_orbper",
    "inclinationDegrees": "pl_orbincl",
    "densityGramsCm3": "pl_dens",
    "equilibriumTemperatureK": "pl_eqt",
    "stellarRadiusSolar": "st_rad",
    "rightAscensionDegrees": "ra",
    "declinationDegrees": "dec",
    "discoveryYear": "disc_year",
    "discoveryMethod": "discoverymethod",
    "systemPlanetCount": "sy_pnum",
}


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def measured(p, column, positive=False):
    value = p.get(column)
    return (
        value
        if p.get(column + "lim") in (None, 0)
        and finite(value)
        and (not positive or value > 0)
        else None
    )


def calculated(reference):
    return "calculated_value" in (reference or "").lower()


def minimum_mass(p):
    return "sin" in (p.get("pl_bmassprov") or "").lower()


def boundaries(teff):
    t = teff - 5780
    return {
        key: c[0] + c[1] * t + c[2] * t * t + c[3] * t * t * t + c[4] * t * t * t * t
        for key, c in COEFFICIENTS.items()
    }


def evaluate(p):
    """Return (candidate, first exclusion, flux, flux provenance, zone)."""
    if not (p.get("pl_name") or "").strip():
        return None, "missingName", None, None, None
    for flag, reason in (
        ("pl_controv_flag", "controversial"),
        ("cb_flag", "circumbinary"),
    ):
        if p.get(flag) == 1:
            return None, reason, None, None, None
    teff = measured(p, "st_teff")
    if teff is None:
        return None, "starTemperatureUnknown", None, None, None
    if not 2600 <= teff <= 7200:
        return None, "starTemperatureOutOfRange", None, None, None
    flags = []
    flux = measured(p, "pl_insol", True)
    lum, axis = measured(p, "st_lum"), measured(p, "pl_orbsmax", True)
    if flux is None and lum is not None and axis is not None:
        try:
            flux = 10**lum / axis**2
        except (OverflowError, ZeroDivisionError):
            flux = None
        flags.append("fluxEstimatedFromStarLuminosityAndOrbit")
    if not finite(flux) or flux <= 0:
        return None, "fluxUnknown", None, None, None
    provenance = (
        "calculated" if flags or calculated(p.get("pl_insol_reflink")) else "published"
    )
    b = boundaries(teff)
    conservative = b["maximumGreenhouse"] <= flux <= b["runawayGreenhouse"]
    optimistic = b["earlyMars"] <= flux <= b["recentVenus"]
    zone = "conservative" if conservative else "optimistic" if optimistic else "outside"

    def excluded(reason):
        return None, reason, flux, provenance, zone

    if not optimistic:
        return excluded("outsideHabitableZone")
    radius, mass = measured(p, "pl_rade", True), measured(p, "pl_bmasse", True)
    if radius is None and mass is None:
        return excluded("sizeUnknown")
    if radius is not None and not 0.5 <= radius <= 2.5:
        return excluded("radiusOutOfRange")
    if mass is not None and not 0.1 <= mass <= 10:
        return excluded("massOutOfRange")
    hz = 50 if conservative else 25
    size = rounded(
        35
        * max(
            0,
            1
            - abs(math.log(radius if radius is not None else mass))
            / math.log(2.5 if radius is not None else 10),
        ),
        1,
    )
    eccentricity = measured(p, "pl_orbeccen")
    if eccentricity is not None and 0 <= eccentricity < 1:
        orbit = rounded(15 * (1 - min(eccentricity / 0.5, 1)), 1)
    else:
        orbit = 7.5
        flags.append("eccentricityUnknownOrLimited")
    if radius is None:
        flags.append("radiusMissingOrLimited")
    if mass is None:
        flags.append("massMissingOrLimited")
    for column, flag in (
        ("pl_rade", "radius"),
        ("pl_bmasse", "mass"),
        ("pl_insol", "flux"),
    ):
        if calculated(p.get(column + "_reflink")):
            flags.append(flag + "CalculatedByArchive")
    if minimum_mass(p):
        flags.append("massMayBeMinimum")
    c = {key: p.get(column) for key, column in FIELDS.items()}
    c.update(
        score=rounded(hz + size + orbit, 1),
        habitableZonePoints=hz,
        sizePoints=size,
        orbitPoints=orbit,
        zone=zone,
        radiusEarth=radius,
        massEarth=mass,
        fluxEarth=rounded(flux, 4),
        stellarTemperatureK=teff,
        eccentricity=eccentricity,
        conservativeOuterFlux=rounded(b["maximumGreenhouse"], 4),
        conservativeInnerFlux=rounded(b["runawayGreenhouse"], 4),
        optimisticOuterFlux=rounded(b["earlyMars"], 4),
        optimisticInnerFlux=rounded(b["recentVenus"], 4),
        dataFlags=flags,
        orbitalPeriodProvenance="calculated"
        if calculated(p.get("pl_orbper_reflink"))
        else "published",
        fluxProvenance=provenance,
        rank=0,
        stability=None,
    )
    return c, None, flux, provenance, zone
