"""160 deterministic asymmetric-error samples, including all catalog planets.

The seeded subtractive generator matches System.Random(int) used by the .NET implementation.
"""

import math

from .scoring import evaluate, finite, rounded

SAMPLES = 160
PERTURBED = (
    "pl_rade",
    "pl_bmasse",
    "pl_insol",
    "st_teff",
    "st_lum",
    "pl_orbsmax",
    "pl_orbeccen",
)


class DotNetRandom:
    def __init__(self, seed):
        self.values = [0] * 56
        mj = 161803398 - abs(seed)
        self.values[55] = mj
        mk = 1
        for i in range(1, 55):
            ii = 21 * i % 55
            self.values[ii] = mk
            mk = mj - mk
            if mk < 0:
                mk += 2147483647
            mj = self.values[ii]
        for _ in range(4):
            for i in range(1, 56):
                self.values[i] -= self.values[1 + (i + 30) % 55]
                if self.values[i] < 0:
                    self.values[i] += 2147483647
        self.a, self.b = 0, 21

    def next_double(self):
        self.a = self.a + 1 if self.a < 55 else 1
        self.b = self.b + 1 if self.b < 55 else 1
        value = self.values[self.a] - self.values[self.b]
        if value == 2147483647:
            value -= 1
        if value < 0:
            value += 2147483647
        self.values[self.a] = value
        return value * (1.0 / 2147483647)


def has_error(high, low):
    return (finite(high) and high > 0) or (finite(low) and low < 0)


def percentile(values, fraction, fallback):
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)] if values else fallback


def perturb(p, rng):
    result = p.copy()
    for column in PERTURBED:
        value, high, low = p.get(column), p.get(column + "err1"), p.get(column + "err2")
        if value is None:
            continue
        if has_error(high, low):
            z = math.sqrt(-2 * math.log(1 - rng.next_double())) * math.cos(
                2 * math.pi * rng.next_double()
            )
            sigma = (
                (high if high is not None and high > 0 else abs(low))
                if z >= 0
                else (abs(low) if low is not None and low < 0 else high)
            )
            value += z * sigma
        if column in ("pl_rade", "pl_bmasse", "pl_insol", "pl_orbsmax"):
            value = max(value, 0.000001)
        if column == "pl_orbeccen":
            value = max(0, min(0.999, value))
        result[column] = value
    return result


def calculate(planets, candidates):
    rng = DotNetRandom(20261114)
    results = {c["name"]: {"scores": [], "ranks": [], "top": 0} for c in candidates}
    for _ in range(SAMPLES):
        ranked = [c for p in planets if (c := evaluate(perturb(p, rng))[0]) is not None]
        ranked.sort(key=lambda c: (-c["score"], c["name"]))
        for rank, c in enumerate(ranked, 1):
            if c["name"] in results:
                r = results[c["name"]]
                r["scores"].append(c["score"])
                r["ranks"].append(rank)
                r["top"] += rank <= 50
    source = {p["pl_name"]: p for p in reversed(planets) if p.get("pl_name")}
    for c in candidates:
        r, p = results[c["name"]], source[c["name"]]
        fields = ["pl_rade", "pl_bmasse", "st_teff", "pl_orbeccen"]
        fields += (
            ["pl_insol"]
            if (p.get("pl_insol") or 0) > 0 and p.get("pl_insollim") in (None, 0)
            else ["st_lum", "pl_orbsmax"]
        )
        present = [f for f in fields if p.get(f) is not None]

        c["stability"] = {
            "samples": SAMPLES,
            "inputsWithErrors": sum(
                has_error(p.get(f + "err1"), p.get(f + "err2")) for f in present
            ),
            "scoredInputs": len(present),
            "scoreP10": rounded(percentile(r["scores"], 0.1, c["score"]), 1),
            "scoreP90": rounded(percentile(r["scores"], 0.9, c["score"]), 1),
            "rankP10": round(percentile(r["ranks"], 0.1, c["rank"])),
            "rankP90": round(percentile(r["ranks"], 0.9, c["rank"])),
            "top50Percent": rounded(100 * r["top"] / SAMPLES, 1),
            "eligiblePercent": rounded(100 * len(r["ranks"]) / SAMPLES, 1),
        }
    return candidates
