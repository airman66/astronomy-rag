"""Apply checked QA corrections to the cache and matching draft record fields."""

from __future__ import annotations

import json
import re

from app.infra.files import write_json
from app.pipeline.translation_fields import fields as translation_fields
from app.pipeline.translation_fields import set_path
from scripts.translation_review.common import (
    DATA,
    TERMS,
    digest,
    legacy_translation_input,
)
from scripts.translation_review.legacy.common import CACHE, DECISIONS, RESOLUTIONS

NUMBERS = re.compile(r"(?<!\d)\d+(?:[.,]\d+)*(?!\d)")
EAST_ASIAN = re.compile(r"[一-龯ぁ-んァ-ン가-힣]")


def numeric_sequence(value: str) -> list[str]:
    return [match.replace(",", ".") for match in NUMBERS.findall(value)]


def main() -> None:
    cache = json.loads(CACHE.read_text())
    resolutions = json.loads(legacy_translation_input(DATA, RESOLUTIONS.name).read_text())["results"]
    selected: dict[str, tuple[str, str]] = {}
    rejected: list[dict] = []
    for source, current in cache.items():
        identifier = digest(source, current)
        result = resolutions.get(identifier)
        if not result or not result["valid"] or result["corrected_ru"] == current:
            continue
        proposal = result["corrected_ru"]
        reason = None
        if not proposal.strip():
            reason = "empty"
        elif len(proposal) < 0.5 * len(current):
            reason = "truncated"
        elif len(proposal) > 2.0 * len(current) and not EAST_ASIAN.search(source):
            reason = "inflated"
        elif numeric_sequence(source) != numeric_sequence(proposal):
            reason = "numbers_changed"
        elif len(source) > 800 and abs(len(proposal) - len(current)) > max(400, 0.35 * len(current)):
            reason = "large_rewrite"
        if reason:
            rejected.append(
                {
                    "id": identifier,
                    "reason": reason,
                    "source": source,
                    "current_ru": current,
                    "proposed_ru": proposal,
                }
            )
        else:
            selected[source] = (current, proposal)
    touched = 0
    fields = 0
    for path in sorted(TERMS.glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("review", {}).get("status") == "reviewed":
            continue
        changed = False
        for field in translation_fields(record):
            if field.get("media_reviewed"):
                continue
            correction = selected.get(field["en"])
            if correction and field["ru"] == correction[0]:
                set_path(record, field["ru_path"], correction[1])
                fields += 1
                changed = True
        if changed:
            write_json(path, record)
            touched += 1
    for source, (_, proposal) in selected.items():
        cache[source] = proposal
    write_json(CACHE, cache)
    write_json(
        DECISIONS,
        {
            "applied_unique": len(selected),
            "applied_fields": fields,
            "touched_records": touched,
            "rejected": rejected,
        },
    )
    print(
        f"applied_unique={len(selected)} applied_fields={fields} touched_records={touched} rejected={len(rejected)}"
    )
