"""Wikimedia API and term-to-file candidate selection."""

import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from scripts.covers.manifest import DATA, load_manifest, write_json

CACHE = DATA / "illustrations" / "wikimedia_candidates.json"
HEADERS = {"User-Agent": "AstronomyGlossaryRAG/1.0 (educational glossary; https://github.com/airman66)"}
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


def api(host: str, **params: str | int) -> dict:
    global LAST_REQUEST
    query = urllib.parse.urlencode({"action": "query", "format": "json", "formatversion": 2, **params})
    request = urllib.request.Request(f"https://{host}/w/api.php?{query}", headers=HEADERS)
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
            time.sleep(max(15, int(exc.headers.get("Retry-After") or 15)) * (attempt + 1))
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
        bad in title.casefold() for bad in ("flag of", "logo", "icon", "seal of", "coat of arms", "wordmark")
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
    manifest = load_manifest()
    existing = json.loads(CACHE.read_text()) if CACHE.exists() else {"items": {}}
    items = existing["items"]
    queue = [
        item
        for item in manifest["items"]
        if str(item["term_number"]) not in items or items[str(item["term_number"])].get("error")
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
