"""Register localized diagrams without overwriting their source images."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pytesseract
from PIL import Image

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.infra.config import ROOT, data_directory
from app.infra.files import checksum, write_json

DATA = data_directory(ROOT / "data").resolve()


def recorded_path(path: Path) -> str:
    if path.is_relative_to(DATA):
        return (Path("data") / path.relative_to(DATA)).as_posix()
    return path.as_posix()


def labels(path: Path, language: str) -> list[str]:
    with Image.open(path) as image:
        text = pytesseract.image_to_string(image.convert("RGB"), lang=language, config="--psm 11")
    return [line.strip() for line in text.splitlines() if line.strip()]


def main() -> None:
    journal_path = DATA / "image_edits.json"
    journal = json.loads(journal_path.read_text())
    existing = {item["media_id"]: item for item in journal["items"]}
    processed = sorted((DATA / "processed" / "terms").glob("*.json"))
    records = {path: json.loads(path.read_text()) for path in processed}
    media_by_id = defaultdict(list)
    for path, record in records.items():
        for item in record.get("media", []):
            media_by_id[item["id"]].append((path, record["id"], item))

    changed: set[Path] = set()
    for ru_path in sorted((DATA / "media" / "ru").glob("*")):
        if ru_path.name.startswith(".") or not ru_path.is_file():
            continue
        original_matches = list((DATA / "media" / "original").glob(ru_path.stem + ".*"))
        if len(original_matches) != 1:
            raise FileNotFoundError(f"Expected one original for {ru_path.stem}: {original_matches}")
        original_path = original_matches[0]
        media_id = ru_path.stem.replace("_", ":", 1)
        matches = media_by_id.get(media_id)
        if not matches:
            raise ValueError(f"No record refers to {media_id}")
        digest = checksum(ru_path)
        filename = recorded_path(ru_path)
        references = sorted({record_id for _, record_id, _ in matches})
        source_url = matches[-1][2]["source_url"]
        for path, _, item in matches:
            if item.get("ru_path") != filename or item.get("ru_sha256") != digest:
                item["ru_path"] = filename
                item["ru_sha256"] = digest
                changed.add(path)
        if media_id in existing:
            existing[media_id]["record_ids"] = references
            existing[media_id]["ru_sha256"] = digest
            continue
        existing[media_id] = {
            "media_id": media_id,
            "record_ids": references,
            "source_url": source_url,
            "original_path": recorded_path(original_path),
            "ru_path": filename,
            "ru_sha256": digest,
            "source_labels": labels(original_path, "eng"),
            "ru_labels": labels(ru_path, "rus+eng"),
            "status": "edited",
            "edited_at": datetime.now(timezone.utc).isoformat(),
            "reviewer_notes": "ChatGPT image editing; OCR inventory is provisional. Verify every label and scientific geometry before publication.",
        }
    for path in sorted(changed):
        write_json(path, records[path])
    journal["items"] = sorted(existing.values(), key=lambda item: item["media_id"])
    write_json(journal_path, journal)
    print(f"registered {len(journal['items'])} localized images")


if __name__ == "__main__":
    main()
