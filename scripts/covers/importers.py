"""Import and verify covers from Commons, Astro4Edu and related terms."""

import json
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from app.catalog.media import local_file
from app.infra.files import checksum
from scripts.covers.sources import (
    CACHE,
    download,
    freely_licensed,
    image_metadata,
    normalized,
)
from scripts.illustration_files import DATA, FILES, MANIFEST, image_size, write_json

RELATED = DATA / "illustrations" / "related_fallbacks.json"
CURATED = DATA / "illustrations" / "curated_wikimedia.json"
ASTRO_CURATED = DATA / "illustrations" / "curated_astro.json"
REJECTED_MATCHES = {
    13,
    14,
    24,
    35,
    45,
    47,
    59,
    60,
    68,
    78,
    80,
    81,
    83,
    85,
    96,
    107,
    109,
    139,
    146,
    158,
    198,
    204,
    226,
    228,
    229,
    241,
    247,
    249,
    260,
    302,
    304,
    316,
    320,
    353,
    354,
    355,
    371,
    389,
    396,
    399,
    409,
    438,
    446,
    448,
    456,
    465,
    475,
    484,
    508,
    510,
    522,
    523,
    524,
    528,
}


def ocr_latin(path: Path) -> str:
    result = subprocess.run(
        ["tesseract", str(path), "stdout", "-l", "eng+rus", "tsv"],
        capture_output=True,
        text=True,
        timeout=25,
        check=False,
    )
    words = []
    for line in result.stdout.splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) >= 12:
            try:
                confidence = float(fields[10])
            except ValueError:
                continue
            word = fields[11].strip()
            if confidence >= 60 and re.search(r"[A-Za-z]{4,}", word):
                words.append(word)
    return " ".join(words[:12])


def import_one(item: dict, candidate: dict, *, curated: bool = False) -> tuple[int, str]:
    number = int(item["term_number"])
    if number in REJECTED_MATCHES and not curated:
        return number, "semantic_mismatch"
    metadata = candidate.get("metadata")
    if not metadata:
        return number, "missing"
    target = FILES / f"{number}.webp"
    if (
        not curated
        and item.get("source_type") == "wikimedia"
        and item.get("status") in ("generated", "approved")
        and target.exists()
    ):
        return number, "existing"
    try:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            download(metadata["download_url"], raw)
            converted = Path(directory) / "cover.webp"
            subprocess.run(
                [
                    "magick",
                    str(raw),
                    "-auto-orient",
                    "-resize",
                    "960x960>",
                    "-strip",
                    "-quality",
                    "84",
                    str(converted),
                ],
                check=True,
                capture_output=True,
                timeout=40,
            )
            _, width, height = image_size(converted)
            if min(width, height) < 240 or not 0.35 <= width / height <= 3:
                return number, f"unsuitable_dimensions:{width}x{height}"
            latin = ocr_latin(converted)
            if latin:
                return number, f"latin_text:{latin}"
            FILES.mkdir(parents=True, exist_ok=True)
            converted.replace(target)
            for old in FILES.glob(f"{number}.*"):
                if old != target:
                    old.unlink()
        item.update(
            {
                "status": "generated",
                "path": target.relative_to(DATA).as_posix(),
                "width": width,
                "height": height,
                "bytes": target.stat().st_size,
                "sha256": checksum(target),
                "generator": "Wikimedia Commons",
                "source_type": "wikimedia",
                "source": {
                    **metadata,
                    "article_url": candidate.get("article_url"),
                    "imported_at": datetime.now(timezone.utc).isoformat(),
                    "modified": "resized and converted to WebP",
                },
                "alt_ru": item["title_ru"],
                "alt_en": item["title_en"],
                "caption_ru": f"{item['title_ru']}. Источник: Wikimedia Commons.",
                "review": {
                    "status": "unreviewed",
                    "reviewer": None,
                    "notes": "Сверены лицензия, размер и OCR; смысл и вид требуют редакторской проверки.",
                },
                "last_error": None,
            }
        )
        return number, "imported"
    except Exception as exc:
        return number, str(exc)[:160]


def import_all(workers: int) -> None:
    manifest = json.loads(MANIFEST.read_text())
    candidates = json.loads(CACHE.read_text())["items"]
    pending = [
        item for item in manifest["items"] if candidates.get(str(item["term_number"]), {}).get("metadata")
    ]
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(import_one, item, candidates[str(item["term_number"])]) for item in pending]
        for future in as_completed(futures):
            results.append(future.result())
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(MANIFEST, manifest)
    report = {
        "imported": sum(s in ("imported", "existing") for _, s in results),
        "unresolved": [
            {"term_number": n, "reason": s} for n, s in sorted(results) if s not in ("imported", "existing")
        ],
        "no_match": [int(k) for k, v in candidates.items() if not v.get("metadata")],
    }
    write_json(DATA / "illustrations" / "wikimedia_report.json", report)
    print(json.dumps(report, ensure_ascii=False))


def import_astro(*, curated: bool = False) -> None:
    """Use already licensed, locally stored source media where Wikimedia has no usable cover."""
    manifest = json.loads(MANIFEST.read_text())
    overrides = json.loads(ASTRO_CURATED.read_text()) if curated else {}
    results = []
    for item in manifest["items"]:
        number = int(item["term_number"])
        if curated and str(number) not in overrides:
            continue
        if not curated and item.get("source_type") == "wikimedia":
            continue
        record = json.loads((DATA / "processed" / "terms" / f"{number}.json").read_text())
        selected = None
        term_words = set(normalized(item["title_en"]))

        def relevance(media: dict) -> int:
            description = " ".join(
                str(media.get(field) or "") for field in ("title_en", "short_caption_en", "alt_en")
            )
            words = set(normalized(description))
            score = 5 * len(term_words & words)
            if item["title_en"].casefold() in description.casefold():
                score += 10
            if media.get("ru_path"):
                score += 2
            if str(media.get("original_path") or "").casefold().endswith(".gif"):
                score -= 2
            return score

        media_options = sorted(record.get("media", []), key=relevance, reverse=True)
        if curated:
            media_options = [media for media in media_options if media.get("id") == overrides[str(number)]]
        for media in media_options:
            license_name = str(media.get("license") or "")
            if not freely_licensed(license_name):
                continue
            for field in ("ru_path", "original_path"):
                path = local_file(DATA, media.get(field))
                if path is None or not path.is_relative_to((DATA / "media").resolve()):
                    continue
                try:
                    with tempfile.TemporaryDirectory() as directory:
                        converted = Path(directory) / "cover.webp"
                        subprocess.run(
                            [
                                "magick",
                                f"{path}[0]",
                                "-auto-orient",
                                "-resize",
                                "960x960>",
                                "-strip",
                                "-quality",
                                "84",
                                str(converted),
                            ],
                            check=True,
                            capture_output=True,
                            timeout=40,
                        )
                        _, width, height = image_size(converted)
                        if min(width, height) < 240 or not 0.35 <= width / height <= 3:
                            continue
                        if ocr_latin(converted):
                            continue
                        FILES.mkdir(parents=True, exist_ok=True)
                        target = FILES / f"{number}.webp"
                        converted.replace(target)
                        for old in FILES.glob(f"{number}.*"):
                            if old != target:
                                old.unlink()
                    selected = (media, field, target, width, height)
                    break
                except (subprocess.CalledProcessError, TimeoutError, ValueError):
                    continue
            if selected:
                break
        if not selected:
            results.append({"term_number": number, "reason": "no_suitable_source_media"})
            continue
        media, field, target, width, height = selected
        item.update(
            {
                "status": "generated",
                "path": target.relative_to(DATA).as_posix(),
                "width": width,
                "height": height,
                "bytes": target.stat().st_size,
                "sha256": checksum(target),
                "generator": "Astro4Edu source media",
                "source_type": "astro4edu",
                "source": {
                    "file_page": media.get("source_url"),
                    "article_url": record["source"].get("url"),
                    "license": media.get("license"),
                    "license_url": media.get("license_url"),
                    "author": media.get("credit"),
                    "media_id": media.get("id"),
                    "imported_at": datetime.now(timezone.utc).isoformat(),
                    "modified": "resized and converted to WebP",
                    "from_locale": field,
                },
                "alt_ru": media.get("alt_ru") or media.get("title_ru") or item["title_ru"],
                "alt_en": media.get("alt_en") or media.get("title_en") or item["title_en"],
                "caption_ru": media.get("short_caption_ru") or media.get("title_ru") or item["title_ru"],
                "review": {
                    "status": "unreviewed",
                    "reviewer": None,
                    "notes": "Лицензия и OCR проверены; выбор изображения требует редакторской проверки.",
                },
                "last_error": None,
            }
        )
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(MANIFEST, manifest)
    write_json(
        DATA / "illustrations" / ("astro_curated_report.json" if curated else "astro_fallback_report.json"),
        {
            "imported": sum(i.get("source_type") == "astro4edu" for i in manifest["items"]),
            "unresolved": results,
        },
    )
    print(
        json.dumps(
            {
                "imported": sum(i.get("source_type") == "astro4edu" for i in manifest["items"]),
                "unresolved": len(results),
            },
            ensure_ascii=False,
        )
    )


def import_related() -> None:
    """Apply explicit, reviewable links to an image of a related glossary concept."""
    mapping = json.loads(RELATED.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text())
    by_number = {int(item["term_number"]): item for item in manifest["items"]}
    imported = []
    skipped = []
    for raw_number, raw_donor in mapping.items():
        number = int(raw_number)
        donor_number = int(raw_donor["donor"] if isinstance(raw_donor, dict) else raw_donor)
        force = bool(raw_donor.get("force")) if isinstance(raw_donor, dict) else False
        item, donor = by_number[number], by_number[donor_number]
        if item.get("source_type") in ("wikimedia", "astro4edu") and not force:
            continue
        source = donor.get("source") or {}
        donor_path = DATA / str(donor.get("path") or "")
        if not donor_path.is_file() or not source.get("file_page") or not source.get("license"):
            skipped.append(number)
            continue
        FILES.mkdir(parents=True, exist_ok=True)
        target = FILES / f"{number}.webp"
        shutil.copyfile(donor_path, target)
        for old in FILES.glob(f"{number}.*"):
            if old != target:
                old.unlink()
        _, width, height = image_size(target)
        item.update(
            {
                "status": "generated",
                "path": target.relative_to(DATA).as_posix(),
                "width": width,
                "height": height,
                "bytes": target.stat().st_size,
                "sha256": checksum(target),
                "generator": "Licensed related image",
                "source_type": "related",
                "source": {
                    **source,
                    "related_term_number": donor_number,
                    "related_title_ru": donor["title_ru"],
                },
                "alt_ru": donor.get("alt_ru") or donor["title_ru"],
                "alt_en": donor.get("alt_en") or donor["title_en"],
                "caption_ru": f"Связанное изображение: «{donor['title_ru']}».",
                "review": {
                    "status": "unreviewed",
                    "reviewer": None,
                    "notes": "Связанный термин, а не прямое изображение понятия; требуется редакторская проверка.",
                },
                "last_error": None,
            }
        )
        imported.append(number)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(MANIFEST, manifest)
    print(json.dumps({"imported": imported, "skipped": skipped}, ensure_ascii=False))


def import_curated() -> None:
    """Import manually selected Commons files for ambiguous or missing matches."""
    mapping = json.loads(CURATED.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text())
    by_number = {int(item["term_number"]): item for item in manifest["items"]}
    results = []
    for raw_number, file_title in mapping.items():
        number = int(raw_number)
        current = by_number[number]
        if (
            current.get("status") in ("generated", "approved")
            and (current.get("source") or {}).get("file_title") == file_title
        ):
            results.append((number, "existing"))
            continue
        metadata = image_metadata(file_title)
        if not metadata:
            results.append((number, "no_free_file_metadata"))
            continue
        results.append(
            import_one(
                by_number[number],
                {"metadata": metadata, "article_url": None},
                curated=True,
            )
        )
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(MANIFEST, manifest)
    print(json.dumps({"results": sorted(results)}, ensure_ascii=False))
