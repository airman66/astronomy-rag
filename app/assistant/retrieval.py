"""Rank search hits and prepare context for the astronomy assistant."""

import re
from dataclasses import dataclass
from typing import Literal

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


EVIDENCE_LIMIT = 6
FRAGMENT_LIMIT = 1800
CONTEXT_LIMIT = 8500
ARTICLE_REQUEST = re.compile(r"\b(?:articles?|стать\w*)\b", re.IGNORECASE)
NON_EVIDENCE_HEADINGS = (
    "activit", "related media", "related diagram", "external link", "further reading",
    "активност", "учебн", "образовательн", "связанн", "ссылк", "дополнительное чтение",
)
EvidenceKind = Literal["term", "article", "caption"]


@dataclass(frozen=True)
class Evidence:
    record: dict
    section: dict
    kind: EvidenceKind
    title: str
    heading: str
    text: str
    canonical_definition: str | None
    url: str
    review_status: str
    chunk_id: str | None

    def reference(self) -> dict[str, str | None]:
        return {"record_id": self.record["id"], "section_id": self.section["id"], "chunk_id": self.chunk_id}


def evidence_for(question: str, locale: str) -> dict[str, Evidence]:
    """Keep exact passages, diversify by record and reserve space for a matching article."""
    hits, _ = search_with_mode(question, locale, limit=80)
    records: dict[str, dict] = {}
    primary: dict[str, Evidence] = {}
    extras: dict[tuple[str, str, str], Evidence] = {}
    for hit in hits:
        record_id = hit["record_id"]
        if record_id not in records:
            if len(records) >= 12 and not record_id.startswith("article:"):
                continue
            record = storage.get_term(int(record_id[5:])) if record_id.startswith("term:") else storage.get_article(record_id)
            if record is None:
                continue
            records[record_id] = record
        record = records[record_id]
        localized = record["locales"][locale]
        sections = {section["id"]: section for section in record["sections"]}
        if record_id not in primary:
            definition = next((section for section in sections.values() if section["type"] == "definition"), None)
            if definition and (item := _section_evidence(record, localized, definition, {})):
                primary[record_id] = item

        section_id = hit["section_id"]
        section = sections.get(section_id)
        if section and section["type"] == "definition":
            if record_id in primary and not primary[record_id].chunk_id and hit.get("chunk_id"):
                item = _section_evidence(record, localized, section, hit)
                if item and item.chunk_id:
                    primary[record_id] = item
            continue
        if section and section["type"] != "media_collection":
            item = _section_evidence(record, localized, section, hit)
        else:
            media = next((item for item in record.get("media", []) if item["id"] == section_id), None)
            item = _caption_evidence(record, localized, media, locale, hit) if media else None
        if item:
            if record_id not in primary:
                primary[record_id] = item
            else:
                extras.setdefault((record_id, item.section["id"], item.text), item)

    ordered = list(primary.values())
    article = next((item for item in ordered if item.kind == "article"), None)
    lead = ordered[:4]
    if article and article not in lead:
        lead = [*lead[:3], article]
    lead_ids = {item.record["id"] for item in lead}
    supporting = [item for item in extras.values() if item.record["id"] in lead_ids]
    if article and ARTICLE_REQUEST.search(question):
        supporting.sort(key=lambda item: item.record["id"] != article.record["id"])
    lead.extend(supporting[:EVIDENCE_LIMIT - len(lead)])
    lead.extend(item for item in ordered if item not in lead)
    packed: dict[str, Evidence] = {}
    remaining = CONTEXT_LIMIT
    for item in lead:
        size = len(item.text) + len(item.title) + len(item.heading)
        if size <= remaining:
            packed[f"e{len(packed) + 1}"] = item
            remaining -= size
        if len(packed) == EVIDENCE_LIMIT:
            break
    return packed


def _section_evidence(record: dict, localized: dict, section: dict, hit: dict) -> Evidence | None:
    block = localized["blocks"].get(section["id"], {})
    canonical = block.get("body", "").strip()
    heading = block.get("heading") or section.get("heading") or ""
    if not canonical or (
        section["type"] != "definition" and any(marker in str(heading).casefold() for marker in NON_EVIDENCE_HEADINGS)
    ):
        return None
    kind = "article" if record["id"].startswith("article:") else "term"
    url = f"/api/v1/articles/{record['id']}" if kind == "article" else f"/api/v1/terms/{record['source']['term_number']}"
    return _evidence(record, localized["title"], section, canonical, kind, url, hit, heading=heading)


def _caption_evidence(record: dict, localized: dict, media: dict, locale: str, hit: dict) -> Evidence | None:
    canonical = next((media[f"{field}_{locale}"] for field in ("caption", "short_caption", "alt")
                      if media.get(f"{field}_{locale}")), "")
    if not canonical.strip():
        return None
    section = {"id": media["id"], "type": "caption", "media_ids": [media["id"]]}
    return _evidence(record, localized["title"], section, canonical, "caption",
                     f"/api/v1/media/{media['id']}", hit, heading=media.get(f"title_{locale}") or "",
                     review_status=media["review"]["status"])


def _evidence(record: dict, title: str, section: dict, canonical: str, kind: EvidenceKind, url: str,
              hit: dict, *, heading: str, review_status: str | None = None) -> Evidence:
    exact = hit.get("evidence_text")
    matches = bool(exact and " ".join(exact.split()) in " ".join(canonical.split()))
    fragment = exact if exact and matches else canonical
    if len(fragment) > FRAGMENT_LIMIT and section["type"] == "source_block":
        paragraphs = fragment.split("\n\n")
        fragment = paragraphs[0]
        for paragraph in paragraphs[1:]:
            if len(fragment) + len(paragraph) + 2 > FRAGMENT_LIMIT:
                break
            fragment += "\n\n" + paragraph
    return Evidence(
        record, section, kind, title, heading, text=fragment, url=url,
        canonical_definition=canonical if section["type"] == "definition" else None,
        review_status=review_status or record["review"]["status"], chunk_id=hit.get("chunk_id") if matches else None,
    )
