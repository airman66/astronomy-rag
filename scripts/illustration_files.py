"""Shared illustration manifest and file operations."""

import json
import shutil
import struct
import threading
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
NL = "\n"
LOCK = threading.Lock()
MIN_SIDE = 768
SCHEMA_VERSION = 1

STYLE = (
    "Единый стиль портала «Алкиона»: строгая научно-популярная иллюстрация в духе современного астрономического атласа. "
    "Глубокий тёмно-синий фон #18233A, светлые тонкие линии #FBFBF9, приглушённые акценты: зелёный #2C7A4B, синий #2A78D6, "
    "оранжевый #EB6834, бирюзовый #1BAF7A; цвета звёзд — по их температуре. Матовая плоская заливка с мягкими градиентами, "
    "без свечения ради эффекта, без неонов и фиолетовых градиентов. Горизонтальный формат 4:3. Никакого текста, букв, цифр, "
    "подписей, логотипов и водяных знаков на изображении. Не изображать фотореалистичный снимок телескопа, если это не концепт."
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, value: Any) -> None:
    write_json_atomic(path, value, indent=1)


def load_manifest() -> dict[str, Any]:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"schema_version": SCHEMA_VERSION, "style": STYLE, "items": []}


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


def accept_file(item: dict[str, Any], source: Path, generator: str) -> None:
    kind, width, height = image_size(source)
    if min(width, height) < MIN_SIDE:
        raise ValueError(f"image too small: {width}×{height}, need at least {MIN_SIDE} px on the short side")
    ratio = width / height
    if not 0.5 <= ratio <= 2.0:
        raise ValueError(f"unexpected aspect ratio {ratio:.2f}")
    extension = {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}[kind]
    FILES.mkdir(parents=True, exist_ok=True)
    target = FILES / f"{item['term_number']}{extension}"
    for old in FILES.glob(f"{item['term_number']}.*"):
        if old != target:
            old.unlink()
    shutil.copyfile(source, target)
    digest = checksum(target)
    item.update(
        {
            "status": "generated",
            "path": target.relative_to(DATA).as_posix(),
            "width": width,
            "height": height,
            "bytes": target.stat().st_size,
            "sha256": digest,
            "generator": generator,
            "generated_at": now(),
            "last_error": None,
            "review": {
                "status": "unreviewed",
                "reviewer": None,
                "notes": "Автоматически проверены формат, размер и хэш; содержание не проверено человеком.",
            },
        }
    )
    if not item.get("alt_ru"):
        item["alt_ru"] = f"Иллюстрация к термину «{item['title_ru']}»"
    if not item.get("alt_en"):
        item["alt_en"] = f"Illustration for the term “{item['title_en']}”"


def save_item(item: dict[str, Any]) -> None:
    with LOCK:
        manifest = load_manifest()
        for index, current in enumerate(manifest["items"]):
            if int(current["term_number"]) == int(item["term_number"]):
                manifest["items"][index] = item
        manifest["updated_at"] = now()
        write_json(MANIFEST, manifest)
