"""Canonical inputs, review checkpoints, and report paths."""

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import ROOT, data_directory
from app.fileio import write_json as write_json_atomic

DATA = data_directory(ROOT / "data")
TERMS = DATA / "processed/terms"
OUT = DATA / "eval/translation_audit"
UNITS = OUT / "units.json"
REVIEW = OUT / "review.jsonl"
VERIFY = OUT / "verify.jsonl"
APPLIED = OUT / "applied.json"
REPORT = OUT / "report.json"
SUMMARY = OUT / "summary.md"
EDITORIAL = OUT / "editorial_sync.json"
REVIEW_MODEL = os.getenv("AUDIT_REVIEW_MODEL", "deepseek-v4-pro")
VERIFY_MODEL = os.getenv("AUDIT_VERIFY_MODEL", "deepseek-flash")
REVIEW_PROMPT_VERSION = "audit-review-2026-11-14.1"
# Та же задача рецензента, выполненная GPT-6 Astra через Codex CLI, когда у DeepSeek закончился баланс.
CODEX_REVIEW_VERSION = "audit-review-codex-2026-11-14.1"
REVIEW_VERSIONS = {REVIEW_PROMPT_VERSION, CODEX_REVIEW_VERSION}
VERIFY_PROMPT_VERSION = "audit-verify-2026-11-14.1"
# Та же проверка, выполненная GPT-6 Astra через Codex CLI, когда DeepSeek недоступен.
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


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(source: str, translated: str) -> str:
    return hashlib.sha256((source + "\0" + translated).encode()).hexdigest()


def load_records() -> dict[str, dict[str, Any]]:
    records = {}
    for path in sorted(TERMS.glob("*.json"), key=lambda p: int(p.stem)):
        record = json.loads(path.read_text(encoding="utf-8"))
        records[record["id"]] = record
    return records


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
