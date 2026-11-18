"""Shared illustration manifest and file operations."""

import json
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.infra.config import ROOT, data_directory
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


def load_manifest() -> dict[str, Any]:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"schema_version": SCHEMA_VERSION, "items": []}


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
