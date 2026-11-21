"""Collect deterministic review units and look up term context."""

from typing import Any

from app.pipeline.translation_fields import fields
from scripts.translation_review.common import UNITS, digest, now, terms_by_number, write_json


def definition(record: dict[str, Any], locale: str) -> str:
    blocks = record["locales"][locale].get("blocks", {})
    for section in record["sections"]:
        if section["type"] == "definition":
            return (blocks.get(section["id"], {}).get("body") or "")[:1200]
    return ""


def collect(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_number = terms_by_number(records)
    units: dict[str, dict[str, Any]] = {}
    occurrences = 0
    for record_id, record in records.items():
        for field in fields(record):
            occurrences += 1
            unit_id = digest(field["en"], field["ru"])
            unit = units.get(unit_id)
            if unit is None:
                number = field.get("target_term")
                target = by_number.get(number) if isinstance(number, int) else None
                unit = units[unit_id] = {
                    "id": unit_id,
                    "en": field["en"],
                    "ru": field["ru"],
                    "kinds": [],
                    "context_record": record_id,
                    "target_ru_title": target["locales"]["ru"]["title"] if target else None,
                    "target_en_title": target["locales"]["en"]["title"] if target else None,
                    "occurrences": [],
                }
            if field["kind"] not in unit["kinds"]:
                unit["kinds"].append(field["kind"])
            unit["occurrences"].append([record_id, field["ru_path"]])
    payload = {"generated_at": now(), "records": len(records), "occurrences": occurrences, "units": units}
    write_json(UNITS, payload)
    return payload
