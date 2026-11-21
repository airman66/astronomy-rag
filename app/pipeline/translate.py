"""Resumable EN→RU translation for normalized Astro4Edu records."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import requests

from app.assistant.client import AssistantNotConfigured, LLMClient
from app.infra.config import LLMSettings, ROOT, data_directory, load_env
from app.infra.files import write_json
from app.pipeline.translation_fields import fields

DATA = data_directory(ROOT / "data")
TERMS = DATA / "processed/terms"
TRANSLATION_CACHE = DATA / "processed/.translation-cache.json"
MODEL = "deepseek-flash"
API_URL = "https://api.deepseek.com/chat/completions"
NUMBER_RE = re.compile(r"(?<!\d)\d+(?:[.,]\d+)*(?!\d)")
UNIT_RE = re.compile(r"(?<=\d)(?:\s*(?:%|°[CFK]|km|cm|mm|AU|ly|pc|kg|M☉|R☉)|\s+(?:s|min|h|yr|g|m)\b)")
NUMBER_SPAN_RE = re.compile(r"(?<!\d)\d+(?:[.,\u00a0\u202f ]\d+)*(?:\s*(?:%|°[CFK]|km|cm|mm|AU|ly|pc|kg|M☉|R☉)|\s+(?:s|min|h|yr|g|m)\b)?")
MARKER_RE = re.compile(r"⟦Q\d+⟧")
ROMAN_RE = re.compile(r"(?<!\w)(?:IV|IX|VI{0,3}|V|I{2,3}|X|L|C|D|M)(?!\w)")
LAYOUT_RE = re.compile(r"https?://[^\s<>]+|\n+|" + NUMBER_SPAN_RE.pattern + "|" + ROMAN_RE.pattern)
CACHE: dict[str, str] = {}
CACHE_LOCK = threading.Lock()


def translation_cache_key(text: str, context: str) -> str:
    """A translation is scoped to its source record and field, not just its words."""
    digest = hashlib.sha256(json.dumps([context, text], ensure_ascii=False).encode()).hexdigest()
    return f"v2:{digest}"


def protected_tokens(text: str) -> list[str]:
    # Grouping separators may change to spaces; decimal comma and point are equivalent.
    normalized = re.sub(r"(?<=\d)[\u00a0\u202f ](?=\d{3}(?:\D|$))", "", text)
    normalized = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", normalized)
    numbers = [value.replace(",", ".") for value in NUMBER_RE.findall(normalized)]
    units = [value.strip() for value in UNIT_RE.findall(normalized)]
    return [f"N:{value}" for value in numbers] + [f"U:{value}" for value in units]


def string_slots(record: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Return missing Russian translations in the order used for batch requests."""
    slots = []
    categories = []
    for field in fields(record):
        if field["ru"].strip() or field.get("media_reviewed"):
            continue
        slot = field["en_path"], field["en"], field["ru_path"]
        if field["kind"] == "category":
            categories.append(slot)
        else:
            slots.append(slot)
    return slots + categories


def validate_parity(record: dict[str, Any], *, complete: bool = False) -> None:
    en = record.get("locales", {}).get("en", {})
    ru = record.get("locales", {}).get("ru", {})
    en_blocks, ru_blocks = en.get("blocks", {}), ru.get("blocks", {})
    if set(en_blocks) != set(ru_blocks):
        raise ValueError(f"Locale block ID mismatch: en={sorted(en_blocks)}, ru={sorted(ru_blocks)}")
    for block_id, source in en_blocks.items():
        if set(source) != set(ru_blocks[block_id]):
            raise ValueError(f"Locale field mismatch in block {block_id}")
    if complete and len(en.get("aliases", [])) != len(ru.get("aliases", [])):
        raise ValueError("Locale alias count mismatch")
    if complete and len(record.get("metadata", {}).get("categories", [])) != len(record.get("metadata", {}).get("categories_ru", [])):
        raise ValueError("Category translation count mismatch")


def set_path(obj: dict[str, Any], path: str, value: str) -> None:
    parts = path.strip("/").split("/")
    cur: Any = obj
    for index, part in enumerate(parts[:-1]):
        key: Any = int(part) if isinstance(cur, list) else part
        next_is_index = parts[index + 1].isdigit()
        if isinstance(cur, dict) and key not in cur:
            cur[key] = [] if next_is_index else {}
        cur = cur[key]
    key = int(parts[-1]) if isinstance(cur, list) else parts[-1]
    if isinstance(cur, list):
        while len(cur) <= key:
            cur.append("")
    cur[key] = value


def translate_batch(items: list[dict[str, str]], key: str, client: LLMClient | None = None) -> dict[str, str]:
    resolved: dict[str, str] = {}
    unique: dict[tuple[str, str], str] = {}
    for item in items:
        source = item["text"]
        context = item.get("context", "")
        cached = CACHE.get(translation_cache_key(source, context))
        if cached is not None:
            resolved[item["id"]] = cached
        else:
            unique.setdefault((source, context), f"u{len(unique)}")
    if not unique:
        return resolved
    source_by_id = {identifier: pair for pair, identifier in unique.items()}
    request_items: list[dict[str, str]] = []
    markers: dict[str, dict[str, str]] = {}
    for (source, context), identifier in unique.items():
        replacements: dict[str, str] = {}

        def mask(match: re.Match[str]) -> str:
            marker = f"⟦Q{len(replacements)}⟧"
            replacements[marker] = match.group(0)
            return marker

        if client:
            masked = LAYOUT_RE.sub(mask, source)
        else:
            masked = NUMBER_SPAN_RE.sub(mask, source)
            masked = ROMAN_RE.sub(mask, masked)
        markers[identifier] = replacements
        request_items.append({"id": identifier, "text": masked, "context": context})
    prompt = (
        "Translate every provided English astronomy/education string into natural, accurate Russian. "
        "Return ONLY a JSON object mapping each id to its translation. The context is metadata for "
        "disambiguating terminology, not text to translate; never include it in the answer. "
        "Preserve IDs, all numbers, symbols, units, URLs, markup-like tokens, and meaning. "
        "Keep every ⟦Q0⟧-style protected token unchanged; it represents a source number, unit, URL "
        "or line break. Translate labels, alt text, and captions. Do not omit, summarize, "
        "or leave English words untranslated unless they are proper names/acronyms."
    )
    last: Exception | None = None
    for attempt in range(5):
        try:
            if client:
                result = asyncio.run(client.complete(prompt, {"items": request_items})).data
            else:
                response = requests.post(
                    API_URL,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": MODEL, "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": json.dumps(request_items, ensure_ascii=False)},
                    ], "temperature": 0.1, "max_tokens": 8192,
                        "thinking": {"type": "disabled"}, "response_format": {"type": "json_object"}},
                    timeout=120,
                )
                response.raise_for_status()
                result = json.loads(response.json()["choices"][0]["message"]["content"])
            if set(result) != {item["id"] for item in request_items} or not all(
                isinstance(v, str) and v.strip() for v in result.values()
            ):
                raise ValueError("Model response keys/values do not match request")
            for item in request_items:
                identifier = item["id"]
                expected = markers[identifier]
                value = result[identifier]
                if sorted(MARKER_RE.findall(value)) != sorted(expected):
                    raise ValueError(f"Protected number/unit markers changed for {identifier}")
                for marker, original in expected.items():
                    value = value.replace(marker, original)
                result[identifier] = value
                source, _context = source_by_id[identifier]
                before = protected_tokens(source)
                after = protected_tokens(value)
                if before != after:
                    raise ValueError(f"Numeric/unit token mismatch for {item['id']}: {before!r} != {after!r}")
            with CACHE_LOCK:
                for (source, context), identifier in unique.items():
                    CACHE[translation_cache_key(source, context)] = result[identifier]
                write_json(TRANSLATION_CACHE, CACHE)
            for item in items:
                resolved[item["id"]] = CACHE[translation_cache_key(item["text"], item.get("context", ""))]
            return resolved
        except AssistantNotConfigured:
            raise
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in {401, 402, 403}:
                raise
            last = exc
            time.sleep(min(2 ** attempt, 16))
        except Exception as exc:
            last = exc
            time.sleep(min(2 ** attempt, 16))
    raise RuntimeError(f"Batch failed after retries: {last}")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=TERMS)
    parser.add_argument("--glob", default="*.json", help="record filenames to translate")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-chars", type=int, default=10_000)
    parser.add_argument("--provider", choices=("deepseek", "runpod"), default="deepseek")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    load_env()
    if TRANSLATION_CACHE.exists():
        CACHE.update(json.loads(TRANSLATION_CACHE.read_text()))
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    client = None
    if args.provider == "runpod":
        os.environ["ASSISTANT_PROVIDER"] = "runpod"
        settings = replace(LLMSettings.from_env(), max_tokens=8192, thinking="disabled", reasoning_effort="none")
        client = LLMClient(settings=settings, read_timeout=300)
        client.configured_settings()
    elif not key:
        raise SystemExit("DEEPSEEK_API_KEY is missing from environment/.env")
    total = [0, 0, 0]
    paths = sorted(
        args.dir.glob(args.glob),
        key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1 + p.stem.startswith("oae-document-"), p.stem),
    )
    records: dict[Path, dict[str, Any]] = {}
    pending: dict[tuple[str, str], list[tuple[Path, str]]] = {}
    for path in paths:
        try:
            record = json.loads(path.read_text())
            records[path] = record
            if record.get("review", {}).get("status") == "reviewed":
                continue
            locale_en = record.get("locales", {}).get("en", {})
            locale_ru = record.get("locales", {}).get("ru", {})
            for field in ("aliases",):
                source_values = locale_en.get(field, [])
                target_values = locale_ru.setdefault(field, [])
                while len(target_values) < len(source_values):
                    target_values.append("")
            source_categories = record.get("metadata", {}).get("categories", [])
            target_categories = record.setdefault("metadata", {}).setdefault("categories_ru", [])
            while len(target_categories) < len(source_categories):
                target_categories.append("")
            validate_parity(record)
            for source_path, source, target in string_slots(record):
                context = f"{record['id']} | {locale_en.get('title', '')} | {source_path}"
                cache_key = translation_cache_key(source, context)
                if cache_key in CACHE:
                    set_path(record, target, CACHE[cache_key])
                    total[0] += 1
                else:
                    pending.setdefault((source, context), []).append((path, target))
        except Exception as exc:
            total[2] += 1
            print(f"{path.name}: {exc}", flush=True)
    # Persist every cache-assisted update before making network calls.
    for path, record in records.items():
        if record.get("review", {}).get("status") != "reviewed":
            write_json(path, record)
    packages: list[list[tuple[str, str]]] = []
    size = 0
    for pair in pending:
        source, context = pair
        if not packages or len(packages[-1]) >= args.batch_size or size + len(source) + len(context) > args.max_chars:
            packages.append([])
            size = 0
        packages[-1].append(pair)
        size += len(source) + len(context)

    def send(package_number: int, pairs: list[tuple[str, str]]) -> tuple[int, list[tuple[str, str]], dict[str, str] | Exception]:
        request = [{"id": str(index), "text": source, "context": context}
                   for index, (source, context) in enumerate(pairs)]
        try:
            return package_number, pairs, translate_batch(request, key, client)
        except Exception as exc:
            return package_number, pairs, exc

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(send, number, pairs) for number, pairs in enumerate(packages, 1)]
        for future in as_completed(futures):
            package_number, pairs, result = future.result()
            completed += 1
            if isinstance(result, Exception):
                failed_fields = sum(len(pending[pair]) for pair in pairs)
                total[2] += failed_fields
                print(f"package {package_number}: {result}", flush=True)
                continue
            touched: set[Path] = set()
            for index, pair in enumerate(pairs):
                translated = result[str(index)]
                for path, target in pending[pair]:
                    set_path(records[path], target, translated)
                    touched.add(path)
                    total[0] += 1
            for path in touched:
                write_json(path, records[path])
            print(f"packages={completed}/{len(packages)} translated={total[0]} errors={total[2]}", flush=True)
    for path, record in records.items():
        if record.get("review", {}).get("status") == "reviewed":
            continue
        remaining = len(string_slots(record))
        try:
            validate_parity(record, complete=remaining == 0)
        except Exception as exc:
            total[2] += 1
            print(f"{path.name}: parity check: {exc}", flush=True)
            continue
        total[1] += remaining
    print(f"files={len(paths)} translated_fields={total[0]} untranslated_fields={total[1]} errors={total[2]}", flush=True)
    if total[1] or total[2]:
        raise SystemExit(1)
