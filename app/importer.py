"""Resumable mirror of the Astro4Edu English glossary and linked media."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE = "https://www.astro4edu.org"
API = f"{BASE}/oae-api"
DATA = Path("data")
HEADERS = {"User-Agent": "glossary-rag-import/1.0 (+https://astro4edu.org/)"}
REFRESH = False


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def fetch(url: str, *, binary: bool = False) -> bytes:
    url = url.replace("http://www.astro4edu.org", BASE).replace(
        "http://astro4edu.org", BASE
    )
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            response = requests.get(url, headers=HEADERS, timeout=(10, 45))
            response.raise_for_status()
            if not binary:
                response.json() if "/oae-api/" in url else None
            return response.content
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if "/oae-api/diagrams/?glossary-term=" in url and "IncompleteRead" in str(exc):
                break
            time.sleep(min(0.5 * 2**attempt, 8))
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def raw_json(path: Path, url: str) -> dict:
    if path.exists() and not REFRESH:
        return json.loads(path.read_text())
    content = fetch(url)
    write_bytes(path, content)
    return json.loads(content)


def media_id(item: dict, kind: str) -> str:
    if kind == "diagram":
        return f"diagram:{item['diagram_slug']}"
    slug = urlparse(item["record_url"]).path.rstrip("/").split("/")[-1]
    return f"media:{slug}"


def select_caption(item: dict) -> dict:
    return next(
        (caption for caption in item.get("captions", []) if caption.get("language_code") == "en"),
        {},
    )


def normalize_media(item: dict, kind: str) -> dict:
    caption = select_caption(item) if kind == "media" else item
    source_url = item.get("file_url") if kind == "media" else item.get("multimedia_file")
    identifier = media_id(item, kind)
    title = caption.get("file_title") if kind == "media" else item.get("diagram_title")
    return {
        "id": identifier,
        "kind": kind,
        "file_type": item.get("file_type") if kind == "media" else "img",
        "source_url": source_url,
        "record_url": item.get("record_url") or item.get("diagram_url"),
        "original_path": None,
        "ru_path": None,
        "sha256": None,
        "size": None,
        "title_en": title or "",
        "title_ru": "",
        "short_caption_en": caption.get("short_caption", "") if kind == "media" else "",
        "short_caption_ru": "",
        "caption_en": caption.get("caption", "") or "",
        "caption_ru": "",
        "alt_en": caption.get("alt_text", "") or "",
        "alt_ru": "",
        "credit": caption.get("credit") if kind == "media" else item.get("credit_text"),
        "credit_url": item.get("credit_url"),
        "license": item.get("license"),
        "license_url": item.get("license_url"),
        "term_numbers": item.get("glossary_terms", []),
        "source_language": item.get("language_code") or "en",
        "review": {"status": "draft", "reviewer": None},
    }


def page_sections(html: str, media: list[dict]) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one("div.main")
    if main is None:
        raise ValueError("Missing main page content")
    media_by_url = {urlparse(m["record_url"]).path.rstrip("/"): m["id"] for m in media}
    result = []
    for section in main.find_all("section"):
        heading_tag = section.select_one("h2.module-title")
        if heading_tag is None or heading_tag.get_text(" ", strip=True) == "You may also be interested in":
            continue
        heading = heading_tag.get_text(" ", strip=True)
        name = heading.lower()
        if name.startswith("glossary term"):
            section_id, kind = "definition", "definition"
            description = next(
                (p for p in section.select("p") if p.get_text(" ", strip=True).startswith("Description:")),
                None,
            )
            body = description.get_text(" ", strip=True).removeprefix("Description:").strip() if description else ""
        elif "other languages" in name:
            section_id, kind = "other_languages", "language_links"
            body = ""
        elif "related media" in name:
            section_id, kind = "related_media", "media_collection"
            body = ""
        elif "related diagram" in name:
            section_id, kind = "related_diagrams", "media_collection"
            body = ""
        else:
            section_id = f"section_{len(result) + 1}"
            kind = "source_block"
            body = section.get_text(" ", strip=True)
        links = [
            {"text": a.get_text(" ", strip=True), "url": urljoin(BASE, a.get("href", ""))}
            for a in section.select("a[href]")
            if a.get("href", "").startswith(("/resources/", "https://"))
        ]
        media_ids = [
            media_by_url[urlparse(link["url"]).path.rstrip("/")]
            for link in links
            if urlparse(link["url"]).path.rstrip("/") in media_by_url
        ]
        result.append(
            {
                "id": section_id,
                "type": kind,
                "heading": heading,
                "body": body,
                "links": links,
                "media_ids": list(dict.fromkeys(media_ids)),
                "source_html": str(section),
            }
        )
    return result


def import_term(number: int) -> tuple[int, str]:
    raw = DATA / "raw" / "terms" / f"{number}.json"
    envelope = raw_json(raw, f"{API}/glossary-terms/?language=en&term-number={number}")
    rows = envelope.get("results", [])
    if not rows:
        return number, "missing"
    if len(rows) != 1 or rows[0].get("term_number") != number:
        raise ValueError(f"Unexpected term response for {number}")
    term = rows[0]
    page_url = term.get("url") or f"{BASE}/resources/glossary/term/{number}/"
    page_path = DATA / "raw" / "pages" / f"{number}.html"
    if REFRESH or not page_path.exists():
        write_bytes(page_path, fetch(page_url, binary=True))
    media_rows = []
    for endpoint, kind in (("multimedia", "media"), ("diagrams", "diagram")):
        media_path = DATA / "raw" / endpoint / f"{number}.json"
        try:
            media_envelope = raw_json(
                media_path,
                f"{API}/{endpoint}/?glossary-term={number}&language=en",
            )
        except RuntimeError:
            # Some filtered responses exceed an upstream transfer limit. The
            # page gives the complete ordered set of linked record slugs.
            soup = BeautifulSoup(page_path.read_text(), "html.parser")
            page_kind = "diagram" if kind == "diagram" else "media"
            slugs = list(dict.fromkeys(
                a["href"].rstrip("/").split("/")[-1]
                for a in soup.select(f'a[href^="/resources/{page_kind}/"]')
            ))
            rows = []
            for slug in slugs:
                single = raw_json(
                    DATA / "raw" / endpoint / "by_slug" / f"{slug}.json",
                    f"{API}/{endpoint}/?slug={slug}&language=en",
                )
                rows.extend(single.get("results", []))
            media_envelope = {"count": len(rows), "next": None, "results": rows, "source": "page_slugs"}
            write_json(media_path, media_envelope)
        if media_envelope.get("next"):
            raise ValueError(f"Media pagination needed for term {number}: {endpoint}")
        media_rows.extend(normalize_media(row, kind) for row in media_envelope.get("results", []))
    media_rows = list({row["id"]: row for row in media_rows}.values())
    sections = page_sections(page_path.read_text(), media_rows)
    source_hash = digest(raw.read_bytes() + page_path.read_bytes())
    processed_path = DATA / "processed" / "terms" / f"{number}.json"
    prior = json.loads(processed_path.read_text()) if processed_path.exists() else {}
    old_sections = {section["id"]: section for section in prior.get("sections", [])}
    source_changed = False
    # A changed English source invalidates its translation, but an editor's Russian text is
    # never discarded: it is kept here for the next translation/review pass.
    stale = list(prior.get("stale_translations", []))

    def keep_stale(path: str, previous_en: str | None, previous_ru: str | None) -> None:
        if previous_ru and previous_ru.strip():
            stale.append({"path": path, "previous_en": previous_en or "", "previous_ru": previous_ru, "detected_at": now()})

    for index, section in enumerate(sections):
        previous = old_sections.get(section["id"], {})
        if prior and not previous:
            source_changed = True  # a new section has no translation or review yet
        for field in ("heading", "body"):
            if previous.get(field) == section.get(field):
                if field + "_ru" in previous:
                    section[field + "_ru"] = previous[field + "_ru"]
            elif previous and section.get(field):
                source_changed = True
                keep_stale(f"/sections/{index}/{field}_ru", previous.get(field), previous.get(field + "_ru"))
        old_links = {(link.get("url"), link.get("text")): link for link in previous.get("links", [])}
        if previous and {(link.get("url"), link.get("text")) for link in previous.get("links", [])} != {(link.get("url"), link.get("text")) for link in section["links"]}:
            source_changed = True
        for link in section["links"]:
            old_link = old_links.get((link["url"], link["text"]), {})
            if "text_ru" in old_link:
                link["text_ru"] = old_link["text_ru"]
    old_media = {m["id"]: m for m in prior.get("media", [])}
    for media_index, row in enumerate(media_rows):
        previous = old_media.get(row["id"], {})
        media_changed = False
        for field in ("title", "short_caption", "caption", "alt"):
            if row.get(field + "_en") == previous.get(field + "_en"):
                row[field + "_ru"] = previous.get(field + "_ru", row[field + "_ru"])
            elif previous and row.get(field + "_en"):
                source_changed = media_changed = True
                keep_stale(f"/media/{media_index}/{field}_ru", previous.get(field + "_en"), previous.get(field + "_ru"))
        for field in ("ru_path", "ru_sha256"):
            if field in previous:
                row[field] = previous[field]
        # A reviewed caption whose source changed must return to the translation queue.
        row["review"] = row["review"] if media_changed else previous.get("review", row["review"])
        for field in ("original_path", "sha256", "size"):
            row[field] = previous.get(field, row[field])
    old_ru = prior.get("locales", {}).get("ru", {})
    blocks_en = {s["id"]: {"heading": s["heading"], "body": s["body"]} for s in sections}
    old_en_blocks = prior.get("locales", {}).get("en", {}).get("blocks", {})
    blocks_ru = {}
    for section in sections:
        sid = section["id"]
        blocks_ru[sid] = {}
        for field in ("heading", "body"):
            if old_en_blocks.get(sid, {}).get(field) == blocks_en[sid][field]:
                blocks_ru[sid][field] = old_ru.get("blocks", {}).get(sid, {}).get(field, "")
            else:
                blocks_ru[sid][field] = ""
                if old_en_blocks.get(sid, {}).get(field):
                    source_changed = True
                    keep_stale(f"/locales/ru/blocks/{sid}/{field}", old_en_blocks[sid][field], old_ru.get("blocks", {}).get(sid, {}).get(field))
    old_en = prior.get("locales", {}).get("en", {})
    old_metadata = prior.get("metadata", {})
    if old_en.get("title") != (term.get("term_name") or "") and old_en:
        source_changed = True
    if old_en.get("aliases") != (term.get("alternate_terms") or []) and old_en:
        source_changed = True
    if old_metadata.get("categories") != term.get("categories", []) and old_metadata:
        source_changed = True
    record = {
        "id": f"term:{number}",
        "kind": "term_page",
        "source": {
            "type": "astro4edu_glossary",
            "term_number": number,
            "url": page_url,
            "api_url": f"{API}/glossary-terms/?language=en&term-number={number}",
            "language": "en",
            "fetched_at": prior.get("source", {}).get("fetched_at") or now(),
            "sha256": source_hash,
        },
        "metadata": {
            "approval_level": term.get("term_approval_level"),
            "categories": term.get("categories", []),
            "category_ids": term.get("category_ids", []),
            "categories_ru": old_metadata.get("categories_ru", []) if old_metadata.get("categories") == term.get("categories", []) else [],
            "related_term_numbers": term.get("linked_terms", []),
            "redirect_term_number": term.get("override_term_number"),
            "credit": "IAU OAE",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/deed.en",
        },
        "sections": sections,
        "locales": {
            "en": {
                "title": term.get("term_name") or "",
                "aliases": term.get("alternate_terms") or [],
                "blocks": blocks_en,
            },
            "ru": {
                "title": old_ru.get("title", "") if old_en.get("title") == (term.get("term_name") or "") else "",
                "aliases": old_ru.get("aliases", []) if old_en.get("aliases") == (term.get("alternate_terms") or []) else [],
                "blocks": blocks_ru,
            },
        },
        "media": media_rows,
        "review": {"status": "draft", "reviewer": None, "notes": "Source changed; translation requires review"} if source_changed else prior.get("review", {"status": "draft", "reviewer": None}),
    }
    if old_en.get("title") and old_en.get("title") != record["locales"]["en"]["title"]:
        keep_stale("/locales/ru/title", old_en.get("title"), old_ru.get("title"))
    if stale:
        record["stale_translations"] = stale
    write_json(processed_path, record)
    return number, "imported"


def download_media(workers: int = 8) -> tuple[int, int]:
    files = sorted((DATA / "processed" / "terms").glob("*.json"))
    entries: dict[str, dict] = {}
    for path in files:
        record = json.loads(path.read_text())
        for item in record["media"]:
            entries[item["id"]] = item
    downloaded = 0
    failed = 0
    def one(entry: tuple[str, dict]) -> tuple[str, bool, str | None]:
        identifier, item = entry
        url = item.get("source_url")
        if not url:
            return identifier, False, None
        matches = re.findall(r"\.(png|jpe?g|webp|gif|svg|tiff?)(?:/|$)", urlparse(url).path, re.I)
        # Animated GIFs are typed "vid" by the source API but are still image files to mirror.
        if item.get("file_type") not in (None, "img") and not matches:
            return identifier, False, None
        extension = "." + matches[-1].lower() if matches else ".bin"
        path = DATA / "media" / "original" / f"{identifier.replace(':', '_')}{extension}"
        # The recorded checksum is authoritative: a restored file must match it, otherwise the
        # source changed since import and the file is reported instead of silently accepted.
        expected = item.get("sha256")
        try:
            created = False
            if not path.exists():
                content = fetch(url, binary=True)
                actual = digest(content)
                if expected and actual != expected:
                    return identifier, False, f"checksum mismatch: source changed since import (recorded {expected[:12]}, downloaded {actual[:12]}); file not stored"
                write_bytes(path, content)
                created = True
            else:
                actual = digest(path.read_bytes())
                if expected and actual != expected:
                    return identifier, False, f"local file differs from recorded checksum {expected[:12]}"
            item["original_path"] = path.as_posix()
            item["sha256"] = expected or actual
            item["size"] = path.stat().st_size
            return identifier, created, None
        except Exception as exc:
            return identifier, False, str(exc)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one, entry) for entry in entries.items()]):
            identifier, created, problem = future.result()
            downloaded += int(created)
            failed += int(problem is not None)
            if problem:
                print(f"media {identifier}: {problem}", flush=True)
            if (downloaded + failed) % 25 == 0 and (downloaded or failed):
                print(f"media downloaded={downloaded} failed={failed}", flush=True)
    for path in files:
        record = json.loads(path.read_text())
        changed = False
        for item in record["media"]:
            source = entries[item["id"]]
            for field in ("original_path", "sha256", "size"):
                if item.get(field) != source.get(field):
                    item[field] = source.get(field)
                    changed = True
        if changed:
            write_json(path, record)
    return downloaded, failed


def discover_max_id() -> tuple[int, int]:
    """Read current API pagination and use its count as a coverage gate."""
    first = raw_json(DATA / "raw" / "glossary_pages" / "1.json", f"{API}/glossary-terms/?language=en&page=1")
    count = int(first["count"])
    size = len(first["results"])
    if not size:
        raise ValueError("Glossary API returned an empty first page")
    max_id = max(int(row["term_number"]) for row in first["results"])
    pages = (count + size - 1) // size
    for page in range(2, pages + 1):
        try:
            envelope = raw_json(DATA / "raw" / "glossary_pages" / f"{page}.json", f"{API}/glossary-terms/?language=en&page={page}")
            max_id = max(max_id, *(int(row["term_number"]) for row in envelope["results"]))
        except RuntimeError:
            # Per-ID fetching below is resilient to large truncated pages.
            pass
    prior = [int(path.stem) for path in (DATA / "processed" / "terms").glob("*.json") if path.stem.isdigit()]
    return max(max_id, *(prior or [0]), count + 100), count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-id", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--download-media", action="store_true")
    parser.add_argument("--media-only", action="store_true")
    parser.add_argument("--ids", nargs="*", type=int)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    global REFRESH
    REFRESH = args.refresh
    if args.media_only:
        print("media summary", download_media(args.workers), flush=True)
        return
    expected = None
    if args.ids is None:
        maximum, expected = discover_max_id()
        maximum = args.max_id or maximum
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        numbers = args.ids if args.ids is not None else range(1, maximum + 1)
        futures = {pool.submit(import_term, number): number for number in numbers}
        counts = {"imported": 0, "missing": 0, "failed": 0}
        for future in as_completed(futures):
            number = futures[future]
            try:
                _, status = future.result()
                counts[status] += 1
            except Exception as exc:
                counts["failed"] += 1
                print(f"term {number}: {exc}", flush=True)
            if sum(counts.values()) % 50 == 0:
                print(counts, flush=True)
    print("import summary", counts, flush=True)
    if expected is not None and counts["imported"] != expected:
        print(f"API count mismatch: imported {counts['imported']} of {expected}", flush=True)
        raise SystemExit(1)
    if args.download_media:
        print("media summary", download_media(args.workers), flush=True)
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
