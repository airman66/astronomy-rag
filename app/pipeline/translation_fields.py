"""Enumerate source/translation fields with their stable JSON paths."""

import re
from typing import Any, Iterator

TERM_LINK = re.compile(r"/resources/glossary/term/(\d+)/?$")


def get_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.strip("/").split("/"):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def set_path(obj: Any, path: str, value: str) -> None:
    parts = path.strip("/").split("/")
    cur = obj
    for part in parts[:-1]:
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    last = parts[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def fields(record: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Every non-empty translatable EN field with the path of its Russian counterpart."""
    locales = record.get("locales", {})
    en, ru = locales.get("en", {}), locales.get("ru", {})
    metadata = record.get("metadata", {})
    sections = record.get("sections", [])

    def item(kind: str, en_path: str, ru_path: str, source: Any, target: Any, **extra: Any):
        if isinstance(source, str) and source.strip():
            return {
                "kind": kind,
                "en_path": en_path,
                "ru_path": ru_path,
                "en": source,
                "ru": target if isinstance(target, str) else "",
                **extra,
            }
        return None

    candidates = [item("title", "/locales/en/title", "/locales/ru/title", en.get("title"), ru.get("title"))]
    ru_aliases = ru.get("aliases", [])
    for index, alias in enumerate(en.get("aliases", [])):
        candidates.append(
            item(
                "alias",
                f"/locales/en/aliases/{index}",
                f"/locales/ru/aliases/{index}",
                alias,
                ru_aliases[index] if index < len(ru_aliases) else "",
            )
        )
    categories_ru = metadata.get("categories_ru", [])
    for index, category in enumerate(metadata.get("categories", [])):
        candidates.append(
            item(
                "category",
                f"/metadata/categories/{index}",
                f"/metadata/categories_ru/{index}",
                category,
                categories_ru[index] if index < len(categories_ru) else "",
            )
        )
    section_types = {section["id"]: section["type"] for section in sections}
    for block_id, block in en.get("blocks", {}).items():
        target = ru.get("blocks", {}).get(block_id, {})
        for field in ("heading", "body"):
            kind = (
                "heading"
                if field == "heading"
                else ("definition" if section_types.get(block_id) == "definition" else "section_body")
            )
            candidates.append(
                item(
                    kind,
                    f"/locales/en/blocks/{block_id}/{field}",
                    f"/locales/ru/blocks/{block_id}/{field}",
                    block.get(field),
                    target.get(field),
                )
            )
    for index, section in enumerate(sections):
        for field in ("heading", "body"):
            kind = (
                "heading"
                if field == "heading"
                else ("definition" if section["type"] == "definition" else "section_body")
            )
            candidates.append(
                item(
                    kind,
                    f"/sections/{index}/{field}",
                    f"/sections/{index}/{field}_ru",
                    section.get(field),
                    section.get(f"{field}_ru"),
                )
            )
        for link_index, link in enumerate(section.get("links", [])):
            match = TERM_LINK.search(link.get("url", ""))
            candidates.append(
                item(
                    "link",
                    f"/sections/{index}/links/{link_index}/text",
                    f"/sections/{index}/links/{link_index}/text_ru",
                    link.get("text"),
                    link.get("text_ru"),
                    target_term=int(match.group(1)) if match else None,
                )
            )
    for index, media in enumerate(record.get("media", [])):
        for field in ("title", "short_caption", "caption", "alt"):
            candidates.append(
                item(
                    f"media_{field}",
                    f"/media/{index}/{field}_en",
                    f"/media/{index}/{field}_ru",
                    media.get(f"{field}_en"),
                    media.get(f"{field}_ru"),
                    media_id=media["id"],
                    media_reviewed=media.get("review", {}).get("status") == "reviewed",
                )
            )
    yield from (candidate for candidate in candidates if candidate)
