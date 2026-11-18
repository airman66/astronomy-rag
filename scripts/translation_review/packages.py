"""Export and import batches for review outside the CLI."""

import json
from pathlib import Path
from typing import Any

from scripts.translation_review.common import (
    CODEX_REVIEW_VERSION,
    CODEX_VERIFY_VERSION,
    NL,
    REVIEW,
    REVIEW_VERSIONS,
    VERIFY,
    VERIFY_VERSIONS,
    append_jsonl,
    latest,
    now,
    read_jsonl,
)
from scripts.translation_review.prompts import REVIEW_SYSTEM, VERIFY_SYSTEM
from scripts.translation_review.review import (
    review_batches,
    review_payload,
    validate_review,
)


def export_verify(
    units: dict[str, dict[str, Any]], records: dict[str, dict[str, Any]], out: Path, chunk: int
) -> list[Path]:
    """Pending proposals as self-contained JSON packages for an independent verifier model."""
    reviews = latest(read_jsonl(REVIEW), REVIEW_VERSIONS)
    done = {row["unit"] for row in read_jsonl(VERIFY) if row.get("prompt_version") in VERIFY_VERSIONS}
    pending = [reviews[u] for u in reviews if reviews[u]["verdict"] == "fix" and u not in done and u in units]
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(0, len(pending), chunk):
        items = []
        for row in pending[index : index + chunk]:
            unit = units[row["unit"]]
            record = records[unit["context_record"]]
            items.append(
                {
                    "id": row["unit"][:16],
                    "kind": "/".join(unit["kinds"]),
                    "term_en": record["locales"]["en"]["title"],
                    "en": unit["en"],
                    "ru_current": unit["ru"],
                    "ru_proposed": row["corrected_ru"],
                    "reviewer_issues": row["issues"],
                    "target_ru_title": unit.get("target_ru_title"),
                }
            )
        path = out / f"verify-{index // chunk + 1:02d}.json"
        path.write_text(
            json.dumps({"instructions": VERIFY_SYSTEM, "items": items}, ensure_ascii=False, indent=1),
            encoding="utf-8",
            newline=NL,
        )
        paths.append(path)
    print(f"export-verify: proposals={len(pending)} packages={len(paths)} dir={out}", flush=True)
    return paths


def import_verify(units: dict[str, dict[str, Any]], files: list[Path], model: str) -> int:
    reviews = latest(read_jsonl(REVIEW), REVIEW_VERSIONS)
    by_prefix = {unit_id[:16]: unit_id for unit_id in reviews if unit_id in units}
    rows = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("results", []):
            unit_id = by_prefix.get(item.get("id"))
            if unit_id and item.get("decision") in {"accept", "reject", "uncertain"}:
                rows.append(
                    {
                        "unit": unit_id,
                        "decision": item["decision"],
                        "reason": str(item.get("reason") or "")[:800],
                        "model": model,
                        "prompt_version": CODEX_VERIFY_VERSION,
                        "verified_at": now(),
                    }
                )
    append_jsonl(VERIFY, rows)
    print(f"import-verify: {len(rows)} decisions from {len(files)} files", flush=True)
    return len(rows)


def export_review(
    units: dict[str, dict[str, Any]], records: dict[str, dict[str, Any]], out: Path, max_chars: int
) -> list[Path]:
    """Units without a review as self-contained packages in the reviewer's input format."""
    done = {row["unit"] for row in read_jsonl(REVIEW) if row.get("prompt_version") in REVIEW_VERSIONS}
    batches = review_batches(units, records, done, max_chars)
    out.mkdir(parents=True, exist_ok=True)
    packages: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    size = 0
    for batch in batches:
        length = sum(len(unit["en"]) + len(unit["ru"]) + 200 for unit in batch)
        if current and size + length > 10 * max_chars:
            packages.append(current)
            current, size = [], 0
        current.extend(batch)
        size += length
    if current:
        packages.append(current)
    paths = []
    for index, package in enumerate(packages, 1):
        path = out / f"review-{index:02d}.json"
        path.write_text(
            json.dumps(
                {"instructions": REVIEW_SYSTEM, **review_payload(package, records)},
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
            newline=NL,
        )
        paths.append(path)
    print(f"export-review: units={sum(len(p) for p in packages)} packages={len(paths)} dir={out}", flush=True)
    return paths


def import_review(units: dict[str, dict[str, Any]], files: list[Path], model: str) -> int:
    by_prefix = {unit_id[:16]: unit for unit_id, unit in units.items()}
    rows = []
    skipped = 0
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        stamp = now()
        for item in payload.get("items", []):
            unit = by_prefix.get(item.get("id")) if isinstance(item, dict) else None
            if unit is None:
                skipped += 1
                continue
            try:  # each item is validated on its own: one malformed answer does not discard the package
                accepted = validate_review({"items": [item]}, [unit])
            except ValueError:
                skipped += 1
                continue
            rows.extend(
                {**row, "model": model, "prompt_version": CODEX_REVIEW_VERSION, "reviewed_at": stamp}
                for row in accepted
            )
    append_jsonl(REVIEW, rows)
    print(f"import-review: {len(rows)} reviews from {len(files)} files, skipped {skipped}", flush=True)
    return len(rows)
