"""Resumable EN→RU translation for normalized Astro4Edu term records."""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from app.config import ROOT, data_directory, load_env
from app.fileio import write_json
from app.translation_fields import fields

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
CACHE: dict[str, str] = {}
CACHE_LOCK = threading.Lock()


def protected_tokens(text: str) -> list[str]:
    # Grouping separators may change to spaces; decimal comma and point are equivalent.
    normalized = re.sub(r"(?<=\d)[\u00a0\u202f ](?=\d)", "", text)
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


def translate_batch(items: list[dict[str, str]], key: str) -> dict[str, str]:
    resolved: dict[str, str] = {}
    unique: dict[str, str] = {}
    for item in items:
        text = item["text"]
        if text in CACHE:
            resolved[item["id"]] = CACHE[text]
        else:
            unique.setdefault(text, f"u{len(unique)}")
    if not unique:
        return resolved
    source_by_id = {identifier: text for text, identifier in unique.items()}
    request_items = []
    markers: dict[str, dict[str, str]] = {}
    for source, identifier in unique.items():
        replacements: dict[str, str] = {}
        def mask(match: re.Match[str]) -> str:
            marker = f"⟦Q{len(replacements)}⟧"
            replacements[marker] = match.group(0)
            return marker
        masked = NUMBER_SPAN_RE.sub(mask, source)
        masked = ROMAN_RE.sub(mask, masked)
        markers[identifier] = replacements
        request_items.append({"id": identifier, "text": masked})
    prompt = (
        "Translate every provided English astronomy/education string into natural, accurate Russian. "
        "Return ONLY a JSON object mapping each id to its translation. Preserve IDs, all numbers, "
        "symbols, units, URLs, markup-like tokens, and meaning. Keep every ⟦Q0⟧-style protected token unchanged; "
        "it represents exact source numbers/units. Translate labels, alt text, and captions. "
        "Do not omit, summarize, or leave English words untranslated unless they are proper names/acronyms."
    )
    last: Exception | None = None
    for attempt in range(5):
        try:
            response = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(request_items, ensure_ascii=False)}], "temperature": 0.1, "max_tokens": 8192, "thinking": {"type": "disabled"}, "response_format": {"type": "json_object"}},
                timeout=120,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            result = json.loads(content)
            if set(result) != {item["id"] for item in request_items} or not all(isinstance(v, str) and v.strip() for v in result.values()):
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
                source = source_by_id[identifier]
                before = protected_tokens(source)
                after = protected_tokens(value)
                if before != after:
                    raise ValueError(f"Numeric/unit token mismatch for {item['id']}: {before!r} != {after!r}")
            with CACHE_LOCK:
                for source, identifier in unique.items():
                    CACHE[source] = result[identifier]
                write_json(TRANSLATION_CACHE, CACHE)
            for item in items:
                resolved[item["id"]] = CACHE[item["text"]]
            return resolved
        except Exception as exc:
            last = exc
            time.sleep(min(2 ** attempt, 16))
    raise RuntimeError(f"Batch failed after retries: {last}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=TERMS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    load_env()
    if TRANSLATION_CACHE.exists():
        CACHE.update(json.loads(TRANSLATION_CACHE.read_text()))
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY is missing from environment/.env")
    total = [0, 0, 0]
    paths = sorted(args.dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    records: dict[Path, dict[str, Any]] = {}
    pending: dict[str, list[tuple[Path, str]]] = {}
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
            for _, source, target in string_slots(record):
                if source in CACHE:
                    set_path(record, target, CACHE[source])
                    total[0] += 1
                else:
                    pending.setdefault(source, []).append((path, target))
        except Exception as exc:
            total[2] += 1
            print(f"{path.name}: {exc}", flush=True)
    # Persist every cache-assisted update before making network calls.
    for path, record in records.items():
        if record.get("review", {}).get("status") != "reviewed":
            write_json(path, record)
    unique_texts = list(pending)
    packages: list[tuple[int, list[str]]] = [
        (start // args.batch_size + 1, unique_texts[start:start + args.batch_size])
        for start in range(0, len(unique_texts), args.batch_size)
    ]
    def send(package_number: int, sources: list[str]) -> tuple[int, list[str], dict[str, str] | Exception]:
        request = [{"id": str(index), "text": source} for index, source in enumerate(sources)]
        try:
            return package_number, sources, translate_batch(request, key)
        except Exception as exc:
            return package_number, sources, exc

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(send, number, sources) for number, sources in packages]
        for future in as_completed(futures):
            package_number, sources, result = future.result()
            completed += 1
            if isinstance(result, Exception):
                failed_fields = sum(len(pending[source]) for source in sources)
                total[2] += failed_fields
                print(f"package {package_number}: {result}", flush=True)
                continue
            touched: set[Path] = set()
            for index, source in enumerate(sources):
                translated = result[str(index)]
                for path, target in pending[source]:
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


if __name__ == "__main__":
    main()
