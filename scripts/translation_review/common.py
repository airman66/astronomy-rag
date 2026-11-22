"""Canonical inputs, review checkpoints, and report paths."""

import hashlib
import json
import os
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.infra.config import ROOT, data_directory
from app.infra.files import existing_path
from app.infra.files import write_json as write_json_atomic
from app.pipeline.translation_fields import fields

DATA = data_directory(ROOT / "data")
TERMS = DATA / "processed/terms"
ARTICLES = DATA / "processed/articles"
REVIEW_CORPUS = os.getenv("AUDIT_CORPUS") == "reviews"
OUT = DATA / ("eval/review_translation_audit" if REVIEW_CORPUS else "eval/translation_audit")
UNITS = OUT / "units.json"
REVIEW = OUT / "review.jsonl"
VERIFY = OUT / "verify.jsonl"
APPLIED = OUT / "applied.json"
REPORT = OUT / "report.json"
SUMMARY = OUT / "summary.md"
EDITORIAL = OUT / "editorial_sync.json"
MANUAL_TERMS = OUT / "manual_terms.json"
MANUAL_MEDIA = DATA / "eval/review_translation_audit/manual_media.json"
MANUAL_PAGES = DATA / "eval/review_translation_audit/manual_pages.json"
REVIEW_MODEL = os.getenv("AUDIT_REVIEW_MODEL", "deepseek-v4-pro")
VERIFY_MODEL = os.getenv("AUDIT_VERIFY_MODEL", "deepseek-flash")
REVIEW_PROMPT_VERSION = "audit-review-2026-11-14.1"
# Preserve the Codex review version used when DeepSeek ran out of credit.
CODEX_REVIEW_VERSION = "audit-review-codex-2026-11-14.1"
REVIEW_VERSIONS = {REVIEW_PROMPT_VERSION, CODEX_REVIEW_VERSION}
VERIFY_PROMPT_VERSION = "audit-verify-2026-11-14.1"
# Preserve the Codex verification version used when DeepSeek was unavailable.
CODEX_VERIFY_VERSION = "audit-verify-codex-2026-11-14.1"
VERIFY_VERSIONS = {VERIFY_PROMPT_VERSION, CODEX_VERIFY_VERSION}
ISSUE_TYPES = {
    "calque",
    "grammar",
    "terminology",
    "meaning",
    "omission",
    "addition",
    "number",
    "unit",
    "formula",
    "untranslated",
    "link",
    "consistency",
    "punctuation",
    "other",
}
VERDICTS = {"ok", "fix", "source_error", "uncertain"}
LOCK = threading.Lock()
NL = "\n"


def legacy_translation_path(data: Path, filename: str) -> Path:
    return data / "eval" / "translation_legacy" / filename


def legacy_translation_input(data: Path, filename: str) -> Path:
    return existing_path(legacy_translation_path(data, filename), data / "eval" / filename)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(source: str, translated: str) -> str:
    return hashlib.sha256((source + "\0" + translated).encode()).hexdigest()


def load_terms() -> dict[str, dict[str, Any]]:
    records = {}
    for path in sorted(TERMS.glob("*.json"), key=lambda p: int(p.stem)):
        record = json.loads(path.read_text(encoding="utf-8"))
        records[record["id"]] = record
    return records


def load_records() -> dict[str, dict[str, Any]]:
    if not REVIEW_CORPUS:
        return load_terms()
    records = {}
    for path in sorted(ARTICLES.glob("oae-*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        records[record["id"]] = record
    return records


def manual_definition_edits(
    records: dict[str, dict[str, Any]],
) -> Iterator[tuple[str, str, str, dict[str, Any]]]:
    if not MANUAL_TERMS.exists():
        return
    ledger = json.loads(MANUAL_TERMS.read_text(encoding="utf-8"))["records"]
    for record_id, evidence in ledger.items():
        definition = evidence.get("definition", {})
        record = records.get(record_id)
        if not record or "previous_ru" not in definition:
            continue
        for field in fields(record):
            if (
                field["kind"] == "definition"
                and hashlib.sha256(field["en"].encode()).hexdigest() == definition["en_sha256"]
                and hashlib.sha256(field["ru"].encode()).hexdigest() == definition["ru_sha256"]
            ):
                yield record_id, field["ru_path"], field["ru"], definition


def manual_media_edits(
    records: dict[str, dict[str, Any]],
) -> Iterator[tuple[str, str, str]]:
    if not MANUAL_MEDIA.exists():
        return
    ledger = json.loads(MANUAL_MEDIA.read_text(encoding="utf-8"))["records"]
    for record_id, evidence in ledger.items():
        record = records.get(record_id)
        if record is None:
            continue
        for index, media in enumerate(record.get("media", [])):
            audited = evidence.get(media["id"], {}).get("fields", {})
            for field, hashes in audited.items():
                en = media.get(f"{field}_en", "")
                ru = media.get(f"{field}_ru", "")
                if (
                    hashlib.sha256(en.encode()).hexdigest() == hashes["en_sha256"]
                    and hashlib.sha256(ru.encode()).hexdigest() == hashes["ru_sha256"]
                ):
                    yield record_id, f"/media/{index}/{field}_ru", ru


def manual_page_edits(
    records: dict[str, dict[str, Any]],
) -> Iterator[tuple[str, str, str]]:
    """Match the reviewed pair in each section and its mirrored locale block."""
    if not MANUAL_PAGES.exists():
        return
    payload = json.loads(MANUAL_PAGES.read_text(encoding="utf-8"))
    for record_id, review in payload.get("titles", {}).items():
        record = records.get(record_id)
        if record is not None:
            title_en = record["locales"]["en"]["title"]
            title_ru = record["locales"]["ru"]["title"]
            if (
                title_en.strip() and title_ru.strip()
                and hashlib.sha256(title_en.encode()).hexdigest() == review["en_sha256"]
                and hashlib.sha256(title_ru.encode()).hexdigest() == review["ru_sha256"]
            ):
                yield record_id, "/locales/ru/title", title_ru
    ledger = payload["records"]
    for record_id, evidence in ledger.items():
        record = records.get(record_id)
        if record is None:
            continue
        blocks_en = record["locales"]["en"].get("blocks", {})
        blocks_ru = record["locales"]["ru"].get("blocks", {})
        for index, section in enumerate(record["sections"]):
            section_id = section["id"]
            review = evidence.get(section_id, {})
            body_en = section.get("body") or ""
            body_ru = section.get("body_ru") or ""
            if (
                body_en.strip() and body_ru.strip()
                and hashlib.sha256(body_en.encode()).hexdigest() == review.get("en_sha256")
                and hashlib.sha256(body_ru.encode()).hexdigest() == review.get("ru_sha256")
            ):
                yield record_id, f"/sections/{index}/body_ru", body_ru
                if (
                    blocks_en.get(section_id, {}).get("body") == body_en
                    and blocks_ru.get(section_id, {}).get("body") == body_ru
                ):
                    yield record_id, f"/locales/ru/blocks/{section_id}/body", body_ru

            if not review.get("heading_links_en_sha256"):
                continue
            heading_en = section.get("heading") or ""
            heading_ru = section.get("heading_ru") or ""
            links = section.get("links", [])
            headings_and_links = (
                ("en", heading_en, [(link["url"], link.get("text", "")) for link in links]),
                ("ru", heading_ru, [(link["url"], link.get("text_ru", "")) for link in links]),
            )
            if any(
                hashlib.sha256(
                    json.dumps([heading, labels], ensure_ascii=False, separators=(",", ":")).encode()
                ).hexdigest() != review.get(f"heading_links_{lang}_sha256")
                for lang, heading, labels in headings_and_links
            ):
                continue
            if heading_en.strip() and heading_ru.strip():
                yield record_id, f"/sections/{index}/heading_ru", heading_ru
                if (
                    blocks_en.get(section_id, {}).get("heading") == heading_en
                    and blocks_ru.get(section_id, {}).get("heading") == heading_ru
                ):
                    yield record_id, f"/locales/ru/blocks/{section_id}/heading", heading_ru
            for link_index, link in enumerate(links):
                if link.get("text", "").strip() and link.get("text_ru", "").strip():
                    yield record_id, f"/sections/{index}/links/{link_index}/text_ru", link["text_ru"]


def record_path(record: dict[str, Any], terms: Path = TERMS) -> Path:
    if (number := record["source"].get("term_number")) is not None:
        return terms / f"{number}.json"
    return ARTICLES / f"{record['id'].removeprefix('article:')}.json"


def terms_by_number(records: dict[str, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    terms = load_terms() if REVIEW_CORPUS else records
    return {
        number: record for record in terms.values()
        if (number := record["source"].get("term_number")) is not None
    }


def write_json(path: Path, value: Any) -> None:
    write_json_atomic(path, value, indent=1)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn last line after a crash
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
            target.flush()
            os.fsync(target.fileno())


def latest(rows: list[dict[str, Any]], version: str | set[str]) -> dict[str, dict[str, Any]]:
    versions = {version} if isinstance(version, str) else version
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("prompt_version") in versions:
            result[row["unit"]] = row
    return result
