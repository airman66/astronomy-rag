"""Rank search hits and prepare context for the astronomy assistant."""

import re
from dataclasses import dataclass
from itertools import zip_longest
from typing import Literal, TypedDict

from app.assistant.source_policy import ARTICLE_ID, ARTICLE_REQUEST, ArticleScope, allows_glossary, explicit_articles
from app.catalog import storage
from app.catalog.media import IMAGE_SUFFIXES, local_file, localized_digest, localized_filename, media_file_url
from app.catalog.search import search_with_mode


def image_for(record: dict, section: dict, locale: str) -> dict | None:
    illustration = record.get("illustration")
    caption = section["type"] == "caption"
    if not caption and illustration and illustration.get("url"):
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
    media_items = [item for item in record.get("media", []) if not caption or item["id"] in media_ids]
    for media in sorted(media_items, key=lambda item: item["id"] not in media_ids):
        path = local_file(storage.DATA, localized_filename(media, locale))
        if path and path.suffix.lower() in IMAGE_SUFFIXES:
            return {
                "url": media_file_url(media["id"], locale, localized_digest(media, locale)),
                "alt": media.get(f"alt_{locale}") or media.get(f"title_{locale}", ""),
                "caption": media.get(f"short_caption_{locale}") or media.get(f"caption_{locale}", ""),
                "kind": "source_media",
                "credit": media.get("credit"),
                "credit_url": media.get("credit_url"),
                "license": media.get("license"),
                "license_url": media.get("license_url"),
            }
    return None


EVIDENCE_LIMIT = 8
FRAGMENT_LIMIT = 1800
CONTEXT_LIMIT = 8500
PACKING_VERSION = "2026-11-25.3"
ARTICLE_PASSAGES_REQUEST = re.compile(r"\b(?:по статье|из статьи|use (?:the |an? )?article|from (?:the |an? )?article)\b", re.IGNORECASE)
DEFINITION_REQUEST = re.compile(r"\b(?:define(?:\s+(?:an?|the))?|дай(?:те)?\s+определение)\s+", re.IGNORECASE)
NON_EVIDENCE_HEADINGS = (
    "activit", "related media", "related diagram", "external link", "further reading",
    "активност", "учебн", "образовательн", "связанн", "ссылк", "дополнительное чтение",
)
EvidenceKind = Literal["term", "article", "caption"]


class EvidenceReference(TypedDict):
    record_id: str
    section_id: str
    chunk_id: str | None


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

    def reference(self) -> EvidenceReference:
        return {"record_id": self.record["id"], "section_id": self.section["id"], "chunk_id": self.chunk_id}


def _page_evidence(record_id: str, locale: str) -> tuple[dict, Evidence] | None:
    """Resolve a user-supplied page hint against the local catalogue only."""
    record: dict | None
    if record_id.startswith("term:") and len(record_id) <= 24 and record_id[5:].isdigit():
        number = int(record_id[5:])
        if number > 2**63 - 1:
            return None
        record = storage.get_term(number)
    elif record_id.startswith("article:") and len(record_id) <= 180 and re.fullmatch(
        r"article:[A-Za-z0-9_.:-]+", record_id,
    ):
        record = storage.get_article(record_id)
    else:
        return None
    if record is None or locale not in record.get("locales", {}):
        return None
    localized = record["locales"][locale]
    ordered = sorted(record["sections"], key=lambda section: section["type"] != "definition")
    for section in ordered:
        if item := _section_evidence(record, localized, section, {}):
            return record, item
    return None


def evidence_for(
    question: str, locale: str, *, prefer_article: bool = False,
    context_record_id: str | None = None, trace: dict | None = None, allow_index_load: bool = True,
    article_scope: ArticleScope | None = None,
) -> dict[str, Evidence]:
    """Keep exact passages, diversify by record and reserve space for a matching article."""
    scope = article_scope if article_scope is not None else explicit_articles(question, context_record_id)
    hits, search_mode = search_with_mode(question, locale, limit=80, allow_index_load=allow_index_load)
    if scope:
        hits = _scoped_hits(hits, question, locale, scope)
    if trace is not None:
        trace["search_mode"] = search_mode
        trace["vector_hits"] = sum(bool(hit.get("chunk_id")) for hit in hits)
    records: dict[str, tuple[dict, dict, dict[str, dict], dict[str, dict]]] = {}
    primary: dict[str, Evidence] = {}
    extras: dict[tuple[str, str, str], Evidence] = {}
    for hit in hits:
        record_id = hit["record_id"]
        if record_id not in records:
            is_term = record_id.startswith("term:")
            if len(records) >= 12 and is_term:
                continue
            record = storage.get_record(record_id)
            if record is None:
                continue
            localized = record["locales"][locale]
            sections = {section["id"]: section for section in record["sections"]}
            media = {item["id"]: item for item in record.get("media", [])}
            records[record_id] = record, localized, sections, media
            definition = next((section for section in sections.values() if section["type"] == "definition"), None)
            if definition and (item := _section_evidence(record, localized, definition, {})):
                primary[record_id] = item

        record, localized, sections, media = records[record_id]
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
            item = _caption_evidence(record, localized, media[section_id], locale, hit) if section_id in media else None
        if item:
            if record_id not in primary:
                primary[record_id] = item
            else:
                extras.setdefault((record_id, item.section["id"], item.text), item)

    if scope:
        return _pack_scoped(scope, primary, extras, locale)

    ordered = list(primary.values())
    if context_record_id:
        # Page-relative questions (e.g. "explain this") often have no useful
        # lexical keywords. Keep one actual section from the open page in view.
        pinned = next((item for item in ordered if item.record["id"] == context_record_id), None)
        if pinned is not None:
            ordered.remove(pinned)
        elif page := _page_evidence(context_record_id, locale):
            record, pinned = page
            records[context_record_id] = (
                record, record["locales"][locale],
                {section["id"]: section for section in record["sections"]},
                {item["id"]: item for item in record.get("media", [])},
            )
        if pinned is not None:
            ordered.insert(0, pinned)
    if match := DEFINITION_REQUEST.search(question):
        requested = " ".join(storage.normalized_words(question[match.end():]))
        definition = next((item for item in ordered if item.canonical_definition and item.title
                           and (requested + " ").startswith(" ".join(storage.normalized_words(item.title)) + " ")), None)
        if definition:
            ordered.remove(definition)
            ordered.insert(0, definition)
    article = next((item for item in ordered if item.kind == "article"), None)
    prioritize_article = prefer_article or ARTICLE_REQUEST.search(question) is not None
    lead = ordered[:4]
    if article and (prefer_article or ARTICLE_PASSAGES_REQUEST.search(question)):
        record, localized, sections, _ = records[article.record["id"]]
        direct = [article]
        seen = {article.section["id"]}
        for hit in storage.lexical_search(question, locale, 6, record_id=record["id"]):
            section = sections.get(hit["section_id"])
            if not section or section["id"] in seen:
                continue
            if item := _section_evidence(record, localized, section, hit):
                direct.append(item)
                seen.add(section["id"])
            if len(direct) == 3:
                break
        lead = [*direct, *(item for item in ordered if item is not article)][:4]
    elif article and prioritize_article:
        lead = [article, *(item for item in ordered if item is not article)][:4]
    elif article and article not in lead:
        lead = [*lead[:3], article]
    lead_ids = {item.record["id"] for item in lead}
    lead_passages = {(item.record["id"], item.section["id"], item.text) for item in lead}
    supporting = [item for key, item in extras.items() if item.record["id"] in lead_ids and key not in lead_passages]
    if article and prioritize_article:
        supporting.sort(key=lambda item: item.record["id"] != article.record["id"])
    lead.extend(supporting[:EVIDENCE_LIMIT - len(lead)])
    lead.extend(item for item in ordered if item.record["id"] not in lead_ids)
    packed = _pack_context(lead)
    if prefer_article and article and not any(item.kind == "term" for item in packed.values()):
        glossary = next((item for item in ordered if item.kind == "term"), None)
        if glossary is not None:
            return _pack_context([lead[0], glossary, *(item for item in lead[1:] if item is not glossary)])
    return packed


def _scoped_hits(hits: list[dict], question: str, locale: str, scope: ArticleScope) -> list[dict]:
    query = ARTICLE_ID.sub(" ", question)
    selected = []
    for record_id in scope.record_ids[:EVIDENCE_LIMIT]:
        local = storage.lexical_search(query, locale, limit=EVIDENCE_LIMIT, record_id=record_id)
        global_hits = [hit for hit in hits if hit["record_id"] == record_id]
        exact = {hit["section_id"]: hit for hit in global_hits if hit.get("chunk_id")}
        # Start with the within-article ranking. Prefer exact indexed excerpts
        # over whole sections when the same section also ranked globally.
        selected.extend(exact.get(hit["section_id"], hit) for hit in local)
        seen = {hit["section_id"] for hit in local}
        selected.extend(hit for hit in global_hits if hit["section_id"] not in seen)
    if allows_glossary(question):
        selected.extend(hit for hit in hits if hit["record_id"].startswith("term:"))
    return selected


def _pack_scoped(
    scope: ArticleScope, primary: dict[str, Evidence], extras: dict[tuple[str, str, str], Evidence], locale: str,
) -> dict[str, Evidence]:
    groups: list[list[Evidence]] = []
    for record_id in scope.record_ids[:EVIDENCE_LIMIT]:
        passages = [item for (rid, _, _), item in extras.items() if rid == record_id]
        if item := primary.get(record_id):
            passages.insert(0, item)
        if not any(item.kind == "article" for item in passages) and (page := _page_evidence(record_id, locale)):
            passages.insert(0, page[1])
        if passages:
            groups.append(sorted(passages, key=lambda item: item.kind != "article"))
    glossary = [item for record_id, item in primary.items() if record_id.startswith("term:")]
    for (record_id, _, _), item in extras.items():
        if record_id.startswith("term:"):
            glossary.append(item)
    # Give every requested article a turn before packing additional passages.
    candidates = [item for row in zip_longest(*groups, glossary) for item in row if item is not None]
    return _pack_context(candidates)


def _pack_context(candidates: list[Evidence]) -> dict[str, Evidence]:
    packed: dict[str, Evidence] = {}
    remaining = CONTEXT_LIMIT
    for item in candidates:
        size = len(item.text) + len(item.title) + len(item.heading)
        if size <= remaining:
            packed[f"e{len(packed) + 1}"] = item
            remaining -= size
        if len(packed) == EVIDENCE_LIMIT:
            break
    return packed


def _section_evidence(record: dict, localized: dict, section: dict, hit: dict) -> Evidence | None:
    block = localized["blocks"].get(section["id"], {})
    canonical = (block.get("body") or "").strip()
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
              hit: dict, *, heading: str, review_status: str | None = None) -> Evidence | None:
    exact = hit.get("evidence_text") or ""
    is_definition = section["type"] == "definition"
    normalized = " ".join(canonical.split())
    indexed = " ".join(exact.split())
    matches = bool(indexed and (indexed == normalized if is_definition else indexed in normalized))
    fragment = exact if matches and not is_definition else canonical
    chunk_id = hit.get("chunk_id") if matches else None
    if len(fragment) > FRAGMENT_LIMIT and section["type"] == "source_block":
        paragraphs = re.split(r"\n\s*\n", fragment)
        kept: list[str] = []
        size = 0
        for paragraph in paragraphs:
            size += len(paragraph) + (2 if kept else 0)
            if size > FRAGMENT_LIMIT:
                break
            kept.append(paragraph)
        if not kept:
            return None
        fragment = "\n\n".join(kept)
        chunk_id = None
    return Evidence(
        record, section, kind, title, heading, text=fragment, url=url,
        canonical_definition=canonical if is_definition else None,
        review_status=review_status or record["review"]["status"], chunk_id=chunk_id,
    )
