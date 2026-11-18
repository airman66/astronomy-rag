"""Refresh the cover manifest and verify cover files and attribution.

python -m scripts.illustrations manifest          # refresh from canonical records
python -m scripts.illustrations verify --complete # check that every term has a cover
"""

from __future__ import annotations

import argparse
import json
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.infra.config import ROOT, data_directory
from app.infra.files import checksum
from app.infra.files import write_json as write_json_atomic

DATA = data_directory(ROOT / "data")
TERMS = DATA / "processed" / "terms"
MANIFEST = DATA / "illustrations" / "manifest.json"
FILES = DATA / "media" / "illustrations"
SCHEMA_VERSION = 1


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, value: Any) -> None:
    write_json_atomic(path, value, indent=1)


def load_manifest(*, allow_missing: bool = False) -> dict[str, Any]:
    if allow_missing and not MANIFEST.exists():
        return {"schema_version": SCHEMA_VERSION, "items": []}
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def image_size(path: Path) -> tuple[str, int, int]:
    """Format and dimensions from the file header (PNG, JPEG, WebP) without extra libraries."""
    with path.open("rb") as source:
        data = source.read(65536)
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        width, height = struct.unpack(">II", data[16:24])
        return "png", width, height
    if data[:2] == b"\xff\xd8":
        i = 2
        while i < len(data) - 9:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                height, width = struct.unpack(">HH", data[i + 5 : i + 9])
                return "jpeg", width, height
            length = struct.unpack(">H", data[i + 2 : i + 4])[0]
            i += 2 + length
        raise ValueError("JPEG without frame header")
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        chunk = data[12:16]
        if chunk == b"VP8X":
            width = 1 + int.from_bytes(data[24:27], "little")
            height = 1 + int.from_bytes(data[27:30], "little")
            return "webp", width, height
        if chunk == b"VP8 ":
            width, height = struct.unpack("<HH", data[26:30])
            return "webp", width & 0x3FFF, height & 0x3FFF
        if chunk == b"VP8L":
            bits = int.from_bytes(data[21:25], "little")
            return "webp", (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    raise ValueError("unsupported image format")


def build_manifest() -> dict[str, Any]:
    manifest = load_manifest(allow_missing=True)
    existing = {int(item["term_number"]): item for item in manifest["items"]}
    items = []
    for path in sorted(TERMS.glob("*.json"), key=lambda p: int(p.stem)):
        record = json.loads(path.read_text(encoding="utf-8"))
        number = int(record["source"]["term_number"])
        base = {
            "term_id": record["id"],
            "term_number": number,
            "title_en": record["locales"]["en"]["title"],
            "title_ru": record["locales"]["ru"]["title"],
            "categories": record["metadata"].get("categories", []),
        }
        item = existing.get(number, {})
        merged = {
            **base,
            "kind": item.get("kind", "conceptual"),
            "alt_ru": item.get("alt_ru", ""),
            "alt_en": item.get("alt_en", ""),
            "caption_ru": item.get("caption_ru", ""),
            "status": item.get("status", "pending"),
            "path": item.get("path"),
            "width": item.get("width"),
            "height": item.get("height"),
            "bytes": item.get("bytes"),
            "sha256": item.get("sha256"),
            "generator": item.get("generator", ""),
            "source_type": item.get("source_type"),
            "source": item.get("source"),
            "generated_at": item.get("generated_at"),
            "last_error": item.get("last_error"),
            "review": item.get("review", {"status": "unreviewed", "reviewer": None, "notes": None}),
        }
        items.append(merged)
    manifest = {"schema_version": SCHEMA_VERSION, "updated_at": now(), "items": items}
    write_json(MANIFEST, manifest)
    print(f"manifest: {len(items)} terms", flush=True)
    return manifest


def verify(complete: bool = False) -> int:
    manifest = load_manifest(allow_missing=True)
    records = {int(p.stem) for p in TERMS.glob("*.json")}
    problems: list[str] = []
    numbers = [int(item["term_number"]) for item in manifest["items"]]
    if sorted(numbers) != sorted(records):
        missing = sorted(records - set(numbers))
        extra = sorted(set(numbers) - records)
        problems.append(f"manifest/record mismatch: missing {missing[:10]}, unknown {extra[:10]}")
    if len(numbers) != len(set(numbers)):
        problems.append("duplicate term numbers in manifest")
    counts: dict[str, int] = {}
    referenced = set()
    for item in manifest["items"]:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
        if item["status"] in ("generated", "approved"):
            path = DATA / (item.get("path") or "")
            referenced.add(path.resolve())
            if not path.is_file():
                problems.append(f"term {item['term_number']}: file missing {item.get('path')}")
                continue
            if checksum(path) != item.get("sha256"):
                problems.append(f"term {item['term_number']}: checksum mismatch")
            try:
                _, width, height = image_size(path)
                if (width, height) != (item.get("width"), item.get("height")):
                    problems.append(f"term {item['term_number']}: size differs from manifest")
            except ValueError as exc:
                problems.append(f"term {item['term_number']}: {exc}")
            if not item.get("alt_ru"):
                problems.append(f"term {item['term_number']}: missing alt_ru")
            if item.get("source_type") in ("wikimedia", "astro4edu", "related"):
                source = item.get("source") or {}
                license_name = source.get("license", "").casefold()
                public_domain = (
                    "public domain" in license_name or license_name.startswith("pd-") or license_name == "pd"
                )
                if not all(source.get(field) for field in ("file_page", "license")) or (
                    not public_domain and not source.get("license_url")
                ):
                    problems.append(f"term {item['term_number']}: missing source attribution")
        elif complete:
            problems.append(f"term {item['term_number']}: no cover")
    orphans = (
        [p.name for p in FILES.glob("*") if p.is_file() and p.resolve() not in referenced]
        if FILES.exists()
        else []
    )
    if orphans:
        problems.append(f"files without manifest entry: {orphans[:10]}")
    print(
        json.dumps(
            {"terms": len(numbers), "records": len(records), "statuses": counts, "problems": len(problems)},
            ensure_ascii=False,
        ),
        flush=True,
    )
    for problem in problems[:40]:
        print(" -", problem)
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["manifest", "verify"])
    parser.add_argument("--complete", action="store_true", help="require a ready cover for every term")
    args = parser.parse_args()
    if args.command == "verify":
        return verify(args.complete)
    build_manifest()
    return 0
