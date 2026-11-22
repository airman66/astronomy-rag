"""Semantic chunks derived solely from normalized source records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from functools import lru_cache

from tokenizers import Tokenizer

from app.infra.config import data_directory
from app.infra.files import write_bytes, write_json

DATA = data_directory()

CHUNK_VERSION = "3"
TOKENIZER_MODEL = "Qwen/Qwen3-Embedding-8B"


@lru_cache(maxsize=1)
def tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained(TOKENIZER_MODEL)


def token_count(text: str) -> int:
    return len(tokenizer().encode(text).ids)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _split_prose(paragraph: str, budget: int) -> list[str]:
    """Split overlong prose by sentences first, and by words only if needed.

    Preserve formulas, tables and multi-line lists as indivisible units; those
    may exceed the budget rather than corrupt their mathematical structure.
    """
    if token_count(paragraph) <= budget:
        return [paragraph]
    if ("\\begin{" in paragraph or "$$" in paragraph
            or re.search(r"(?m)^\s*(?:[-*•]|\d+[.)])\s", paragraph)
            or re.search(r"(?m)^\s*\|[^\n]*\|", paragraph)):
        return [paragraph]

    def split_sentence(sentence: str) -> list[str]:
        if token_count(sentence) <= budget:
            return [sentence]
        pieces: list[str] = []
        words: list[str] = []
        for word in sentence.split():
            if words and token_count(" ".join([*words, word])) > budget:
                pieces.append(" ".join(words))
                words = []
            if token_count(word) > budget:
                # Do not clip identifiers or scientific words in the middle.
                if words:
                    pieces.append(" ".join(words))
                    words = []
                pieces.append(word)
            else:
                words.append(word)
        if words:
            pieces.append(" ".join(words))
        return pieces

    sentences: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        if sentence.strip():
            for piece in split_sentence(sentence):
                sentences.append(piece.strip())
    result: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if current and token_count(" ".join([*current, sentence])) > budget:
            result.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        result.append(" ".join(current))
    return result

def paragraphs(body: str, budget: int, overlap: float) -> list[str]:
    """Pack article prose at paragraph/sentence boundaries with bounded overlap."""
    if budget <= 0 or not 0 <= overlap < 1:
        raise ValueError("budget must be positive and overlap in [0, 1)")
    parts: list[str] = []
    for paragraph in re.split(r"\n\s*\n", body):
        if paragraph.strip():
            parts.extend(_split_prose(paragraph.strip(), budget))
    if not parts:
        return []
    groups: list[str] = []
    current: list[str] = []
    for part in parts:
        count = token_count(part)
        if current and token_count("\n\n".join([*current, part])) > budget:
            groups.append("\n\n".join(current))
            # Retain complete previous paragraphs only while both overlap
            # and the next group remain inside the token budget.
            retained: list[str] = []
            for old in reversed(current):
                candidate = [old, *retained]
                if (token_count("\n\n".join(candidate)) > budget * overlap
                        or token_count("\n\n".join([*candidate, part])) > budget):
                    break
                retained = candidate
            current = retained
        current.append(part)
        if count > budget:
            # Large unsplittable scientific blocks intentionally overflow.
            groups.append("\n\n".join(current))
            current = []
    if current:
        groups.append("\n\n".join(current))
    return groups


def make_chunk(record_id: str, section_id: str, locale: str, title: str, heading: str, body: str, kind: str, url: str, status: str, ordinal: int = 0, media_id: str | None = None, source_record_id: str | None = None) -> dict:
    identifier = f"{record_id}/{section_id}/{locale}/{ordinal}"
    searchable = " → ".join(filter(None, (title, heading))) + "\n" + body
    return {
        "id": identifier,
        "content_hash": sha(searchable),
        "record_id": record_id,
        "section_id": section_id,
        "locale": locale,
        "title": title,
        "heading": heading,
        "text": body,
        "search_text": searchable,
        "kind": kind,
        "url": url,
        "review_status": status,
        "media_id": media_id,
        # Media captions are indexed once; search maps them back to the first record that shows them.
        "source_record_id": source_record_id,
    }


def build_chunks(budget: int = 400, overlap: float = 0.1) -> list[dict]:
    result: list[dict] = []
    seen_media: set[tuple[str, str]] = set()
    files = (
        sorted((DATA / "processed" / "terms").glob("*.json"))
        + sorted((DATA / "processed" / "articles").glob("*.json"))
    )
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        record_id = record["id"]
        kind = "article" if record_id.startswith("article:") else "term"
        route = f"/api/v1/articles/{record_id}" if kind == "article" else f"/api/v1/terms/{record['source']['term_number']}"
        for locale in ("en", "ru"):
            loc = record["locales"][locale]
            for section in record["sections"]:
                block = loc.get("blocks", {}).get(section["id"], {})
                body = (block.get("body") or "").strip()
                if not body:
                    continue
                if kind == "article" and section["type"] not in ("definition", "formula", "table"):
                    groups = paragraphs(body, budget, overlap)
                else:
                    groups = [body]
                for ordinal, part in enumerate(groups):
                    result.append(make_chunk(record_id, section["id"], locale, loc["title"], block.get("heading") or "", part, kind, route, record["review"]["status"], ordinal))
            for item in record.get("media", []):
                key = item["id"], locale
                if key in seen_media:
                    continue
                seen_media.add(key)
                body = (item.get(f"caption_{locale}") or item.get(f"short_caption_{locale}") or item.get(f"alt_{locale}") or "").strip()
                if body:
                    result.append(make_chunk(item["id"], "caption", locale, item.get(f"title_{locale}") or "", "", body, item["kind"], f"/api/v1/media/{item['id']}", item["review"]["status"], media_id=item["id"], source_record_id=record_id))
    return sorted(result, key=lambda x: x["id"])


def write_chunks(budget: int = 400, overlap: float = 0.1) -> dict:
    chunks = build_chunks(budget, overlap)
    destination = DATA / "index" / "chunks.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in chunks).encode()
    write_bytes(destination, content)
    manifest = {"chunk_version": CHUNK_VERSION, "budget_tokens": budget, "overlap": overlap, "tokenizer_model": TOKENIZER_MODEL, "chunk_count": len(chunks), "chunks_sha256": hashlib.sha256(content).hexdigest()}
    write_json(DATA / "index" / "chunk-manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=400)
    parser.add_argument("--overlap", type=float, default=0.1)
    args = parser.parse_args()
    print(json.dumps(write_chunks(args.budget, args.overlap)))
