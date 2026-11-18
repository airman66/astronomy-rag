"""Rank search hits and prepare context for the astronomy assistant."""

from app.catalog import storage
from app.catalog.media import IMAGE_SUFFIXES, local_file, localized_filename
from app.catalog.search import search_with_mode


def image_for(record: dict, section: dict, locale: str) -> dict | None:
    illustration = record.get("illustration")
    if illustration and illustration.get("url"):
        full_illustration = storage.get_illustration(record["source"]["term_number"])
        filename = full_illustration.get("path", "") if full_illustration else ""
        path = local_file(storage.DATA, filename)
        if path:
            source = (full_illustration or {}).get("source") or {}
            return {
                "url": illustration["url"],
                "alt": illustration.get(f"alt_{locale}", ""),
                "caption": illustration.get(f"caption_{locale}", ""),
                "kind": "illustration",
                "credit": source.get("author"),
                "credit_url": source.get("file_page"),
                "license": source.get("license"),
                "license_url": source.get("license_url"),
                "modified": source.get("modified"),
            }
    media_ids = set(section.get("media_ids", []))
    media_items = record.get("media", [])
    for media in sorted(media_items, key=lambda item: item["id"] not in media_ids):
        path = local_file(storage.DATA, localized_filename(media, locale))
        if path and path.suffix.lower() in IMAGE_SUFFIXES:
            return {
                "url": f"/api/v1/media/{media['id']}/file?locale={locale}",
                "alt": media.get(f"alt_{locale}") or media.get(f"title_{locale}", ""),
                "caption": media.get(f"short_caption_{locale}") or media.get(f"caption_{locale}", ""),
                "kind": "source_media",
                "credit": media.get("credit"),
                "credit_url": media.get("credit_url"),
                "license": media.get("license"),
                "license_url": media.get("license_url"),
            }
    return None


def candidates_for(question: str, locale: str) -> list[dict]:
    results = search_with_mode(question, locale, limit=80)[0]
    question_words = storage.normalized_words(question)

    def title_match_fraction(title: str) -> float:
        words = [word for word in storage.normalized_words(title) if len(word) >= 4]
        matched = sum(
            any(query.startswith(word[: max(4, len(word) - 2)]) for query in question_words) for word in words
        )
        return matched / len(words) if words else 0.0

    results.sort(key=lambda row: row["score"] + 0.08 * title_match_fraction(row["title"]), reverse=True)
    unique: dict[str, dict] = {}
    for result in results:
        identifier = result["record_id"]
        if identifier in unique:
            continue
        record = (
            storage.get_term(int(identifier.split(":", 1)[1]))
            if identifier.startswith("term:")
            else storage.get_article(identifier)
        )
        if record is None:
            continue
        localized = record["locales"][locale]
        blocks = localized.get("blocks", {})
        sections = [section for section in record["sections"] if blocks.get(section["id"], {}).get("body")]
        if not sections:
            continue
        primary = next((section for section in sections if section["type"] == "definition"), sections[0])
        matched = next((section for section in sections if section["id"] == result["section_id"]), primary)
        image = image_for(record, primary, locale)
        unique[identifier] = {
            **result,
            "section_id": primary["id"],
            "title": localized["title"],
            "excerpt": blocks[primary["id"]]["body"][:1600],
            "matched_section_id": matched["id"],
            "matched_excerpt": blocks[matched["id"]]["body"][:1600],
            "kind": "article" if identifier.startswith("article:") else "term",
            "image": image,
            "matched_image": image if matched is primary else image_for(record, matched, locale),
            "url": f"/api/v1/articles/{identifier}"
            if identifier.startswith("article:")
            else f"/api/v1/terms/{record['source']['term_number']}",
        }
        if len(unique) == 8:
            break
    return list(unique.values())
