"""Normalize Astro4Edu records and reconcile them with prior translations."""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

BASE = "https://www.astro4edu.org"
API = f"{BASE}/oae-api"
StaleTranslationCallback = Callable[[str, str | None, str | None], None]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
            body = (
                description.get_text(" ", strip=True).removeprefix("Description:").strip()
                if description
                else ""
            )
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
        links: list[dict[str, str]] = []
        for anchor in section.select("a[href]"):
            href = anchor.get("href")
            if isinstance(href, str) and href.startswith(("/resources/", "https://")):
                links.append({"text": anchor.get_text(" ", strip=True), "url": urljoin(BASE, href)})
        media_ids: list[str] = []
        seen_media: set[str] = set()
        for link in links:
            path = urlparse(link["url"]).path.rstrip("/")
            if path in media_by_url:
                identifier = media_by_url[path]
                if identifier not in seen_media:
                    seen_media.add(identifier)
                    media_ids.append(identifier)
        result.append(
            {
                "id": section_id,
                "type": kind,
                "heading": heading,
                "body": body,
                "links": links,
                "media_ids": media_ids,
                "source_html": str(section),
            }
        )
    return result


@dataclass
class TermSources:
    term: dict
    page_url: str
    sections: list[dict]
    media: list[dict]
    snapshots: list[Path]
    sha256: str


def snapshot_hash(paths: list[Path]) -> str:
    source_digest = hashlib.sha256()
    for path in paths:
        content = path.read_bytes()
        source_digest.update(len(content).to_bytes(8, "big"))
        source_digest.update(content)
    return source_digest.hexdigest()


def merge_sections(sections: list[dict], prior: dict, keep_stale: StaleTranslationCallback) -> bool:
    old_sections = {section["id"]: section for section in prior.get("sections", [])}
    old_ru_blocks = prior.get("locales", {}).get("ru", {}).get("blocks", {})
    source_changed = False
    for index, section in enumerate(sections):
        previous = old_sections.get(section["id"], {})
        if prior and not previous:
            source_changed = True  # a new section has no translation or review yet
        for field in ("heading", "body"):
            if previous.get(field) == section.get(field):
                if field + "_ru" in previous:
                    section[field + "_ru"] = previous[field + "_ru"]
            elif previous:
                source_changed = True
                keep_stale(f"/sections/{index}/{field}_ru", previous.get(field), previous.get(field + "_ru"))
        old_links = {(link.get("url"), link.get("text")): link for link in previous.get("links", [])}
        new_links = {(link["url"], link["text"]) for link in section["links"]}
        if previous and set(old_links) != new_links:
            source_changed = True
        for link in section["links"]:
            old_link = old_links.get((link["url"], link["text"]), {})
            if "text_ru" in old_link:
                link["text_ru"] = old_link["text_ru"]
        for old_index, link in enumerate(previous.get("links", [])):
            if (link.get("url"), link.get("text")) not in new_links:
                keep_stale(
                    f"/sections/{index}/links/{old_index}/text_ru", link.get("text"), link.get("text_ru")
                )
    for old_id in old_sections.keys() - {section["id"] for section in sections}:
        source_changed = True
        old = old_sections[old_id]
        old_index = prior["sections"].index(old)
        old_block = old_ru_blocks.get(old_id, {})
        for field in ("heading", "body"):
            keep_stale(f"/sections/{old_index}/{field}_ru", old.get(field), old.get(f"{field}_ru"))
            keep_stale(f"/locales/ru/blocks/{old_id}/{field}", old.get(field), old_block.get(field))
        for link_index, link in enumerate(old.get("links", [])):
            keep_stale(
                f"/sections/{old_index}/links/{link_index}/text_ru", link.get("text"), link.get("text_ru")
            )
    return source_changed


def merge_media(media_rows: list[dict], prior: dict, keep_stale: StaleTranslationCallback) -> bool:
    old_media = {item["id"]: item for item in prior.get("media", [])}
    media_ids = {item["id"] for item in media_rows}
    source_changed = bool(prior and media_ids != old_media.keys())
    for old_id in old_media.keys() - media_ids:
        old = old_media[old_id]
        old_index = prior["media"].index(old)
        for field in ("title", "short_caption", "caption", "alt"):
            keep_stale(f"/media/{old_index}/{field}_ru", old.get(f"{field}_en"), old.get(f"{field}_ru"))
    for media_index, row in enumerate(media_rows):
        previous = old_media.get(row["id"], {})
        source_file_changed = bool(previous and row.get("source_url") != previous.get("source_url"))
        media_changed = source_file_changed
        for field in ("title", "short_caption", "caption", "alt"):
            translated = f"{field}_ru"
            if row.get(field + "_en") == previous.get(field + "_en"):
                if translated in previous:
                    row[translated] = previous[translated]
            elif previous:
                media_changed = True
                keep_stale(
                    f"/media/{media_index}/{translated}",
                    previous.get(field + "_en"),
                    previous.get(translated),
                )
        if not source_file_changed:
            for field in ("ru_path", "ru_sha256", "original_path", "sha256", "size"):
                if field in previous:
                    row[field] = previous[field]
        # Changes to the original file or captions invalidate media review.
        if media_changed:
            source_changed = True
        elif "review" in previous:
            row["review"] = previous["review"]
    return source_changed


def merge_term(number: int, sources: TermSources, prior: dict, *, refresh: bool) -> dict:
    term, page_url = sources.term, sources.page_url
    sections, media_rows, raw_paths = sources.sections, sources.media, sources.snapshots
    # A changed English source invalidates its translation, but an editor's Russian text is
    # never discarded: it is kept here for the next translation/review pass.
    stale = list(prior.get("stale_translations", []))

    def keep_stale(path: str, previous_en: str | None, previous_ru: str | None) -> None:
        if previous_ru and previous_ru.strip():
            stale.append(
                {
                    "path": path,
                    "previous_en": previous_en or "",
                    "previous_ru": previous_ru,
                    "detected_at": now(),
                }
            )

    sections_changed = merge_sections(sections, prior, keep_stale)
    media_changed = merge_media(media_rows, prior, keep_stale)
    source_changed = sections_changed or media_changed
    old_locales = prior.get("locales", {})
    old_en = old_locales.get("en", {})
    old_ru = old_locales.get("ru", {})
    old_metadata = prior.get("metadata", {})
    title = term.get("term_name") or ""
    aliases = term.get("alternate_terms") or []
    categories = term.get("categories", [])
    same_title = old_en.get("title") == title
    same_aliases = old_en.get("aliases") == aliases
    same_categories = old_metadata.get("categories") == categories

    blocks_en = {s["id"]: {"heading": s["heading"], "body": s["body"]} for s in sections}
    old_en_blocks = old_en.get("blocks", {})
    old_ru_blocks = old_ru.get("blocks", {})
    blocks_ru = {}
    for section in sections:
        section_id = section["id"]
        previous_en = old_en_blocks.get(section_id, {})
        previous_ru = old_ru_blocks.get(section_id, {})
        translated = {}
        for field, text in blocks_en[section_id].items():
            previous = previous_en.get(field)
            translated[field] = previous_ru.get(field, "") if previous == text else ""
            if previous and previous != text:
                source_changed = True
                keep_stale(f"/locales/ru/blocks/{section_id}/{field}", previous, previous_ru.get(field))
        blocks_ru[section_id] = translated

    if old_en and not same_title:
        source_changed = True
    if old_en and not same_aliases:
        source_changed = True
        translation = old_ru.get("aliases", [])
        for index, value in enumerate(old_en.get("aliases", [])):
            keep_stale(
                f"/locales/ru/aliases/{index}",
                value,
                translation[index] if index < len(translation) else None,
            )
    if old_metadata and not same_categories:
        source_changed = True
        translation = old_metadata.get("categories_ru", [])
        for index, value in enumerate(old_metadata.get("categories", [])):
            keep_stale(
                f"/metadata/categories_ru/{index}",
                value,
                translation[index] if index < len(translation) else None,
            )
    record = {
        "id": f"term:{number}",
        "kind": "term_page",
        "source": {
            "type": "astro4edu_glossary",
            "term_number": number,
            "url": page_url,
            "api_url": f"{API}/glossary-terms/?language=en&term-number={number}",
            "language": "en",
            "fetched_at": now() if refresh else prior.get("source", {}).get("fetched_at") or now(),
            "sha256": sources.sha256,
            "raw_snapshots": [path.as_posix() for path in raw_paths],
        },
        "metadata": {
            "approval_level": term.get("term_approval_level"),
            "categories": categories,
            "category_ids": term.get("category_ids", []),
            "categories_ru": old_metadata.get("categories_ru", []) if same_categories else [],
            "related_term_numbers": term.get("linked_terms", []),
            "redirect_term_number": term.get("override_term_number"),
            "credit": "IAU OAE",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/deed.en",
        },
        "sections": sections,
        "locales": {
            "en": {
                "title": title,
                "aliases": aliases,
                "blocks": blocks_en,
            },
            "ru": {
                "title": old_ru.get("title", "") if same_title else "",
                "aliases": old_ru.get("aliases", []) if same_aliases else [],
                "blocks": blocks_ru,
            },
        },
        "media": media_rows,
        "review": {
            "status": "draft",
            "reviewer": None,
            "notes": "Source changed; translation requires review",
        }
        if source_changed
        else prior.get("review", {"status": "draft", "reviewer": None}),
    }
    if old_en.get("title") and not same_title:
        keep_stale("/locales/ru/title", old_en.get("title"), old_ru.get("title"))
    if stale:
        record["stale_translations"] = stale
    return record
