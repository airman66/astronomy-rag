"""Find and import individually licensed Wikimedia covers for glossary terms.

Each request addresses one article or file. No paid image generation or batch API is used.
The candidate cache makes discovery and downloads resumable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from scripts.illustrations import DATA, FILES, MANIFEST, ROOT, image_size, write_json

CACHE = DATA / "illustrations" / "wikimedia_candidates.json"
RELATED = DATA / "illustrations" / "related_fallbacks.json"
CURATED = DATA / "illustrations" / "curated_wikimedia.json"
ASTRO_CURATED = DATA / "illustrations" / "curated_astro.json"
HEADERS = {
    "User-Agent": "AstronomyGlossaryRAG/1.0 (educational glossary; https://github.com/airman66/astro)"
}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".svg")
LICENSES = ("cc by", "cc-by", "cc0", "public domain", "pd-")
RATE_LOCK = Lock()
LAST_REQUEST = 0.0
TITLE_OVERRIDES = {
    "Apollo": "Apollo program",
    "Andromeda": "Andromeda (constellation)",
    "Altitude": "Altitude (astronomy)",
    "Axis": "Axis of rotation",
    "Corona": "Stellar corona",
    "Cluster": "Star cluster",
    "Color": "Color of stars",
    "Disk": "Accretion disk",
    "Dust": "Cosmic dust",
    "Energy": "Energy",
    "Mercury": "Mercury (planet)",
    "Moon": "Moon",
    "Nucleus": "Atomic nucleus",
    "Ring": "Planetary ring",
    "Space": "Outer space",
    "Spring": "Season",
    "Autumn": "Autumn",
    "Summer": "Summer",
    "Winter": "Winter",
    "Taurus": "Taurus (constellation)",
    "Leo": "Leo (constellation)",
    "Aries": "Aries (constellation)",
    "Libra": "Libra (constellation)",
    "Aquarius": "Aquarius (constellation)",
    "Cancer": "Cancer (constellation)",
    "Gemini": "Gemini (constellation)",
    "Pisces": "Pisces (constellation)",
    "Sagittarius": "Sagittarius (constellation)",
    "Scorpius": "Scorpius",
    "Virgo": "Virgo (constellation)",
    "Orion": "Orion (constellation)",
    "Capricornus": "Capricornus",
    "Io": "Io (moon)",
    "Titan": "Titan (moon)",
    "Triton": "Triton (moon)",
    "Venus": "Venus",
    "Mars": "Mars",
}
# The Wikipedia lead image can illustrate a homonym or a tangential example.
# These concrete matches were checked against the glossary definitions and rejected.
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


def api(host: str, **params: str | int) -> dict:
    global LAST_REQUEST
    query = urllib.parse.urlencode(
        {"action": "query", "format": "json", "formatversion": 2, **params}
    )
    request = urllib.request.Request(
        f"https://{host}/w/api.php?{query}", headers=HEADERS
    )
    for attempt in range(4):
        try:
            with RATE_LOCK:
                delay = 0.55 - (time.monotonic() - LAST_REQUEST)
                if delay > 0:
                    time.sleep(delay)
                LAST_REQUEST = time.monotonic()
            with urllib.request.urlopen(request, timeout=25) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 3:
                raise
            time.sleep(
                max(15, int(exc.headers.get("Retry-After") or 15)) * (attempt + 1)
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    return {}


def plain(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", value)).strip()


def freely_licensed(value: str) -> bool:
    name = value.casefold().strip()
    return name == "pd" or any(token in name for token in LICENSES)


def normalized(value: str) -> list[str]:
    return re.findall(r"[a-zа-яё0-9]+", value.casefold().replace("–", "-"))


def file_ok(title: str) -> bool:
    return title.casefold().endswith(IMAGE_SUFFIXES) and not any(
        bad in title.casefold()
        for bad in ("flag of", "logo", "icon", "seal of", "coat of arms", "wordmark")
    )


def pages(host: str, title: str, search: bool = False) -> list[dict]:
    params = {
        "prop": "pageimages|pageprops",
        "piprop": "name",
        "pilicense": "free",
        "pithumbsize": 960,
        "pilangcode": "ru",
    }
    if search:
        params.update(
            {
                "generator": "search",
                "gsrsearch": f'intitle:"{title}"',
                "gsrlimit": 5,
                "gsrnamespace": 0,
            }
        )
    else:
        params.update({"titles": title, "redirects": 1})
    return api(host, **params).get("query", {}).get("pages", [])


def files(title: str) -> list[dict]:
    results = []
    for query in (f'intitle:"{title}"',):
        data = api(
            "commons.wikimedia.org",
            generator="search",
            gsrsearch=query,
            gsrnamespace=6,
            gsrlimit=20,
            prop="imageinfo",
            iiprop="size",
            redirects=1,
        )
        results.extend(data.get("query", {}).get("pages", []))
    return results


def image_metadata(title: str, host: str = "commons.wikimedia.org") -> dict | None:
    params = {
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|size|extmetadata",
        "iiurlwidth": 960,
    }
    if title.casefold().endswith(".svg"):
        params["iiurlparam"] = "lang=ru"
    data = api(host, **params)
    for page in data.get("query", {}).get("pages", []):
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        license_name = plain(meta.get("LicenseShortName", {}).get("value", ""))
        if not freely_licensed(license_name):
            continue
        url = info.get("thumburl") or info.get("url")
        if not url:
            continue
        return {
            "file_title": page["title"],
            "file_page": info.get("descriptionurl"),
            "download_url": url,
            "original_url": info.get("url"),
            "license": license_name,
            "license_url": plain(meta.get("LicenseUrl", {}).get("value", "")),
            "author": plain(meta.get("Artist", {}).get("value", "")),
            "credit": plain(meta.get("Credit", {}).get("value", "")),
            "source_wiki": host,
            "original_width": info.get("width"),
            "original_height": info.get("height"),
        }
    return None


def candidate_titles(item: dict) -> list[tuple[str, str, int]]:
    title_en, title_ru = (
        TITLE_OVERRIDES.get(item["title_en"], item["title_en"]),
        item["title_ru"],
    )
    clean_en = re.sub(r"\s*\([^)]*\)", "", title_en).strip()
    clean_ru = re.sub(r"\s*\([^)]*\)", "", title_ru).strip()
    found: list[tuple[str, str, int]] = []
    for host, title in (
        ("en.wikipedia.org", title_en),
        ("ru.wikipedia.org", title_ru),
        ("en.wikipedia.org", clean_en),
        ("ru.wikipedia.org", clean_ru),
    ):
        if any(host == h and title == t for h, t, _ in found):
            continue
        for page in pages(host, title):
            if (
                "missing" in page
                or "disambiguation" in page.get("pageprops", {})
                or not page.get("pageimage")
            ):
                continue
            if file_ok(page["pageimage"]):
                found.append(
                    (
                        "File:" + page["pageimage"],
                        f"https://{host}/wiki/{urllib.parse.quote(page['title'].replace(' ', '_'))}",
                        100 if host.startswith("en") else 95,
                    )
                )
        if found:
            break
    if not found:
        for page in files(clean_en):
            if file_ok(page.get("title", "")):
                words = set(normalized(clean_en))
                name = set(normalized(page["title"].removeprefix("File:")))
                score = round(100 * len(words & name) / max(1, len(words)))
                if score >= 60:
                    found.append((page["title"], "", score))
        found.sort(key=lambda row: row[2], reverse=True)
    return found[:5]


def discover(item: dict) -> dict:
    number = item["term_number"]
    try:
        for file_title, article, score in candidate_titles(item):
            metadata = image_metadata(file_title)
            if metadata:
                return {
                    "term_number": number,
                    "article_url": article or None,
                    "match_score": score,
                    "metadata": metadata,
                    "error": None,
                }
        return {
            "term_number": number,
            "article_url": None,
            "match_score": 0,
            "metadata": None,
            "error": "no_matching_free_wikimedia_file",
        }
    except Exception as exc:
        return {
            "term_number": number,
            "article_url": None,
            "match_score": 0,
            "metadata": None,
            "error": str(exc)[:250],
        }


def discover_all(workers: int) -> None:
    manifest = json.loads(MANIFEST.read_text())
    existing = json.loads(CACHE.read_text()) if CACHE.exists() else {"items": {}}
    items = existing["items"]
    queue = [
        item
        for item in manifest["items"]
        if str(item["term_number"]) not in items
        or items[str(item["term_number"])].get("error")
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(discover, item) for item in queue]):
            result = future.result()
            items[str(result["term_number"])] = result
            if len(items) % 20 == 0:
                write_json(CACHE, existing)
    existing["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(CACHE, existing)
    print(
        json.dumps(
            {
                "terms": len(items),
                "matched": sum(bool(x["metadata"]) for x in items.values()),
                "missing": [int(k) for k, x in items.items() if not x["metadata"]],
            },
            ensure_ascii=False,
        )
    )


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                target.write_bytes(response.read(15_000_000))
            return
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)


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


def import_one(
    item: dict, candidate: dict, *, curated: bool = False
) -> tuple[int, str]:
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
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
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
        item
        for item in manifest["items"]
        if candidates.get(str(item["term_number"]), {}).get("metadata")
    ]
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(import_one, item, candidates[str(item["term_number"])])
            for item in pending
        ]
        for future in as_completed(futures):
            results.append(future.result())
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(MANIFEST, manifest)
    report = {
        "imported": sum(s in ("imported", "existing") for _, s in results),
        "unresolved": [
            {"term_number": n, "reason": s}
            for n, s in sorted(results)
            if s not in ("imported", "existing")
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
        record = json.loads(
            (DATA / "processed" / "terms" / f"{number}.json").read_text()
        )
        selected = None
        term_words = set(normalized(item["title_en"]))

        def relevance(media: dict) -> int:
            description = " ".join(
                str(media.get(field) or "")
                for field in ("title_en", "short_caption_en", "alt_en")
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
            media_options = [
                media for media in media_options
                if media.get("id") == overrides[str(number)]
            ]
        for media in media_options:
            license_name = str(media.get("license") or "")
            if not freely_licensed(license_name):
                continue
            for field in ("ru_path", "original_path"):
                raw_path = str(media.get(field) or "")
                path = (
                    ROOT / raw_path if raw_path.startswith("data/") else DATA / raw_path
                )
                # Stored paths are data-relative; never permit a path outside the media tree.
                if not path.is_file() or not path.resolve().is_relative_to(
                    (DATA / "media").resolve()
                ):
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
            results.append(
                {"term_number": number, "reason": "no_suitable_source_media"}
            )
            continue
        media, field, target, width, height = selected
        item.update(
            {
                "status": "generated",
                "path": target.relative_to(DATA).as_posix(),
                "width": width,
                "height": height,
                "bytes": target.stat().st_size,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
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
                "alt_ru": media.get("alt_ru")
                or media.get("title_ru")
                or item["title_ru"],
                "alt_en": media.get("alt_en")
                or media.get("title_en")
                or item["title_en"],
                "caption_ru": media.get("short_caption_ru")
                or media.get("title_ru")
                or item["title_ru"],
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
            "imported": sum(
                i.get("source_type") == "astro4edu" for i in manifest["items"]
            ),
            "unresolved": results,
        },
    )
    print(
        json.dumps(
            {
                "imported": sum(
                    i.get("source_type") == "astro4edu" for i in manifest["items"]
                ),
                "unresolved": len(results),
            },
            ensure_ascii=False,
        )
    )


def write_review() -> None:
    manifest = json.loads(MANIFEST.read_text())
    items = sorted(manifest["items"], key=lambda item: int(item["term_number"]))
    directory = DATA / "illustrations"
    with (directory / "coverage.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "term_number",
                "title_ru",
                "status",
                "source_type",
                "file_page",
                "author",
                "license",
                "license_url",
                "review_status",
            )
        )
        for item in items:
            source = item.get("source") or {}
            writer.writerow(
                (
                    item["term_number"],
                    item["title_ru"],
                    item["status"],
                    item.get("source_type", ""),
                    source.get("file_page", ""),
                    source.get("author", ""),
                    source.get("license", ""),
                    source.get("license_url", ""),
                    item.get("review", {}).get("status", ""),
                )
            )
    cards = []
    for item in items:
        source = item.get("source") or {}
        if item.get("path") and item["status"] in ("generated", "approved"):
            picture = f'<img src="../{html.escape(item["path"], quote=True)}" alt="" loading="lazy">'
        else:
            picture = '<div class="empty">Нет обложки</div>'
        file_page = source.get("file_page") or ""
        link = (
            f'<a href="{html.escape(file_page, quote=True)}" rel="noopener">Источник</a>'
            if file_page
            else ""
        )
        cards.append(
            f'<article><div class="image">{picture}</div><h2>{item["term_number"]}. {html.escape(item["title_ru"])}</h2>'
            f"<p>{html.escape(item.get('source_type') or item['status'])} · {html.escape(source.get('license') or '')}</p>{link}</article>"
        )
    page = (
        """<!doctype html><html lang="ru"><meta charset="utf-8"><title>Обложки глоссария — 501 ID</title>
<style>body{font:15px/1.4 system-ui,sans-serif;margin:24px;background:#f7f7f4;color:#20221f}h1{font-size:28px}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:16px}article{background:white;padding:10px;border:1px solid #ddd}
.image{height:160px;display:grid;place-items:center;background:#e9e9e5}.image img{max-width:100%;max-height:160px;object-fit:contain}
h2{font-size:15px;margin:8px 0 3px}p{color:#555;margin:0 0 6px}a{color:#174b90}.empty{color:#777}</style>
<h1>Обложки глоссария — 501 ID</h1><main>"""
        + "\n".join(cards)
        + "</main></html>\n"
    )
    (directory / "review.html").write_text(page, encoding="utf-8")
    print(
        json.dumps(
            {
                "items": len(items),
                "ready": sum(i["status"] in ("generated", "approved") for i in items),
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
        donor_number = int(
            raw_donor["donor"] if isinstance(raw_donor, dict) else raw_donor
        )
        force = bool(raw_donor.get("force")) if isinstance(raw_donor, dict) else False
        item, donor = by_number[number], by_number[donor_number]
        if item.get("source_type") in ("wikimedia", "astro4edu") and not force:
            continue
        source = donor.get("source") or {}
        donor_path = DATA / str(donor.get("path") or "")
        if (
            not donor_path.is_file()
            or not source.get("file_page")
            or not source.get("license")
        ):
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
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "discover",
            "import",
            "astro-fallback",
            "astro-curated",
            "curated",
            "related",
            "report",
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "discover":
        discover_all(args.workers)
    elif args.command == "astro-fallback":
        import_astro()
    elif args.command == "astro-curated":
        import_astro(curated=True)
    elif args.command == "report":
        write_review()
    elif args.command == "related":
        import_related()
    elif args.command == "curated":
        import_curated()
    else:
        import_all(args.workers)
