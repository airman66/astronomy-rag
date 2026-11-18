"""Resumable mirror of the Astro4Edu English glossary and linked media."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from app.astro4edu_records import (
    API,
    BASE,
    TermSources,
    merge_term,
    normalize_media,
    page_sections,
    snapshot_hash,
)
from app.config import data_directory
from app.fileio import write_bytes, write_json

DATA = data_directory()
HEADERS = {"User-Agent": "glossary-rag-import/1.0 (+https://astro4edu.org/)"}
REFRESH = False


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


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


def current_snapshot(path: Path) -> Path:
    pointer = path.with_suffix(path.suffix + ".current")
    if not pointer.exists():
        return path
    name = pointer.read_text(encoding="utf-8").strip()
    if not name or Path(name).name != name or name == ".." or not (name == path.name or (name.startswith(f"{path.stem}.") and name.endswith(path.suffix))):
        raise ValueError(f"Invalid raw snapshot pointer: {pointer}")
    return path.with_name(name)


def write_immutable(target: Path, content: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != content:
            raise ValueError(f"Immutable file differs: {target}")
    else:
        temporary = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_bytes(content)
            try:
                os.link(temporary, target)
            except FileExistsError:
                if target.read_bytes() != content:
                    raise ValueError(f"Immutable file differs: {target}")
        finally:
            temporary.unlink(missing_ok=True)


def save_snapshot(path: Path, content: bytes) -> Path:
    """Keep the first snapshot and all changed versions immutable."""
    current = current_snapshot(path)
    if current.exists() and current.read_bytes() == content:
        return current
    target = path if not path.exists() or path.read_bytes() == content else path.with_name(f"{path.stem}.{digest(content)}{path.suffix}")
    write_immutable(target, content)
    pointer = path.with_suffix(path.suffix + ".current")
    if target != path or pointer.exists():
        write_bytes(pointer, f"{target.name}\n".encode())
    return target


def raw_bytes(path: Path, url: str) -> bytes:
    current = current_snapshot(path)
    if current.exists() and not REFRESH:
        return current.read_bytes()
    content = fetch(url, binary=True)
    save_snapshot(path, content)
    return content


def raw_json(path: Path, url: str) -> dict:
    current = current_snapshot(path)
    if current.exists() and not REFRESH:
        return json.loads(current.read_bytes())
    content = fetch(url)
    value = json.loads(content)
    save_snapshot(path, content)
    return value


def load_term_sources(number: int) -> TermSources | None:
    raw = DATA / "raw" / "terms" / f"{number}.json"
    envelope = raw_json(raw, f"{API}/glossary-terms/?language=en&term-number={number}")
    rows = envelope.get("results", [])
    if not rows:
        return None
    if len(rows) != 1 or rows[0].get("term_number") != number:
        raise ValueError(f"Unexpected term response for {number}")
    term = rows[0]
    page_url = term.get("url") or f"{BASE}/resources/glossary/term/{number}/"
    page_path = DATA / "raw" / "pages" / f"{number}.html"
    raw_bytes(page_path, page_url)
    page_path = current_snapshot(page_path)
    media_rows = []
    media_paths = []
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
            save_snapshot(media_path, (json.dumps(media_envelope, ensure_ascii=False, indent=2) + "\n").encode())
        if media_envelope.get("next"):
            raise ValueError(f"Media pagination needed for term {number}: {endpoint}")
        media_paths.append(current_snapshot(media_path))
        media_rows.extend(normalize_media(row, kind) for row in media_envelope.get("results", []))
    media_rows = list({row["id"]: row for row in media_rows}.values())
    sections = page_sections(page_path.read_text(), media_rows)
    raw_paths = [current_snapshot(raw), page_path, *media_paths]
    return TermSources(term, page_url, sections, media_rows, raw_paths, snapshot_hash(raw_paths))


def import_term(number: int) -> tuple[int, str]:
    sources = load_term_sources(number)
    if sources is None:
        return number, "missing"
    path = DATA / "processed" / "terms" / f"{number}.json"
    prior = json.loads(path.read_text()) if path.exists() else {}
    write_json(path, merge_term(number, sources, prior, refresh=REFRESH))
    return number, "imported"


def download_media(workers: int = 8, refresh: bool = False, preferred_ids: set[int] | None = None) -> tuple[int, int]:
    files = sorted((DATA / "processed" / "terms").glob("*.json"))
    entries: dict[str, dict] = {}
    for path in files:
        record = json.loads(path.read_text())
        for item in record["media"]:
            if item["id"] not in entries or preferred_ids is None or record.get("source", {}).get("term_number") in preferred_ids:
                entries[item["id"]] = item
    downloaded = 0
    failed = 0
    changed_media: set[str] = set()

    def one(entry: tuple[str, dict]) -> tuple[str, bool, str | None, bool]:
        identifier, item = entry
        url = item.get("source_url")
        if not url:
            return identifier, False, None, False
        matches = re.findall(r"\.(png|jpe?g|webp|gif|svg|tiff?)(?:/|$)", urlparse(url).path, re.I)
        # Animated GIFs are typed "vid" by the source API but are still image files to mirror.
        if item.get("file_type") not in (None, "img") and not matches:
            return identifier, False, None, False
        extension = "." + matches[-1].lower() if matches else ".bin"
        base_path = DATA / "media" / "original" / f"{identifier.replace(':', '_')}{extension}"
        recorded_path = Path(item["original_path"]) if item.get("original_path") else None
        # The recorded checksum is authoritative: a restored file must match it, otherwise the
        # source changed since import and the file is reported instead of silently accepted.
        expected = item.get("sha256")
        try:
            if recorded_path and recorded_path.exists() and expected and digest(recorded_path.read_bytes()) != expected:
                return identifier, False, f"local file differs from recorded checksum {expected[:12]}", False
            if recorded_path and recorded_path.exists() and not refresh:
                item["size"] = recorded_path.stat().st_size
                return identifier, False, None, False
            content = fetch(url, binary=True)
            actual = digest(content)
            if expected and actual != expected and not refresh:
                return identifier, False, f"checksum mismatch: source changed since import (recorded {expected[:12]}, downloaded {actual[:12]}); file not stored", False
            if recorded_path and recorded_path.exists() and digest(recorded_path.read_bytes()) == actual:
                path = recorded_path
            elif base_path.exists() and digest(base_path.read_bytes()) == actual:
                path = base_path
            else:
                path = base_path if not base_path.exists() else base_path.with_name(f"{base_path.stem}.{actual}{extension}")
                write_immutable(path, content)
            content_changed = bool(expected and actual != expected)
            item["original_path"] = path.as_posix()
            item["sha256"] = actual
            item["size"] = path.stat().st_size
            if content_changed:
                item["ru_path"] = None
                item["ru_sha256"] = None
                item["review"] = {"status": "draft", "reviewer": None, "notes": "Source image changed; localization requires review"}
            return identifier, path != recorded_path, None, content_changed
        except Exception as exc:
            return identifier, False, str(exc), False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one, entry) for entry in entries.items()]):
            identifier, created, problem, content_changed = future.result()
            downloaded += int(created)
            failed += int(problem is not None)
            if content_changed:
                changed_media.add(identifier)
            if problem:
                print(f"media {identifier}: {problem}", flush=True)
            if (downloaded + failed) % 25 == 0 and (downloaded or failed):
                print(f"media downloaded={downloaded} failed={failed}", flush=True)
    for path in files:
        record = json.loads(path.read_text())
        changed = False
        for item in record["media"]:
            source = entries[item["id"]]
            if item.get("source_url") != source.get("source_url"):
                item["source_url"] = source.get("source_url")
                item["review"] = {"status": "draft", "reviewer": None, "notes": "Source image changed; localization requires review"}
                record["review"] = {"status": "draft", "reviewer": None, "notes": "Source image changed; localization requires review"}
                changed = True
            for field in ("original_path", "sha256", "size", "ru_path", "ru_sha256", "review"):
                if item.get(field) != source.get(field):
                    item[field] = source.get(field)
                    changed = True
        if any(item["id"] in changed_media for item in record["media"]):
            record["review"] = {"status": "draft", "reviewer": None, "notes": "Source image changed; localization requires review"}
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
        media_summary = download_media(args.workers, refresh=args.refresh, preferred_ids=set(args.ids) if args.ids is not None else None)
        print("media summary", media_summary, flush=True)
        if media_summary[1]:
            raise SystemExit(1)
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
    if args.download_media or args.refresh:
        media_summary = download_media(args.workers, refresh=args.refresh, preferred_ids=set(args.ids) if args.ids is not None else None)
        print("media summary", media_summary, flush=True)
        if media_summary[1]:
            raise SystemExit(1)
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
