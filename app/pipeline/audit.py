"""Validate source IDs, translation coverage, links, and local image hashes."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from app.catalog.media import local_file
from app.infra.config import data_directory
from app.infra.files import checksum
from app.pipeline.importer import current_snapshot
from app.pipeline.translation_fields import fields
from urllib.parse import urlsplit

DATA = data_directory()


def check(*, skip_original_media: bool = False) -> dict:
    files = sorted((DATA / "processed" / "terms").glob("*.json"))
    records = [json.loads(path.read_text()) for path in files]
    ids = {record["source"]["term_number"] for record in records}
    errors = [] if records else ["catalog: no canonical terms found"]
    media: dict[str, dict] = {}
    for record in records:
        number = record["source"]["term_number"]
        if record["id"] != f"term:{number}" or not record["source"]["url"].rstrip("/").endswith(f"/{number}"):
            errors.append(f"term {number}: source ID mismatch")
        if set(record["locales"]["en"]["blocks"]) != set(record["locales"]["ru"]["blocks"]):
            errors.append(f"term {number}: block ID mismatch")
        for linked in record["metadata"].get("related_term_numbers", []):
            if linked not in ids:
                errors.append(f"term {number}: broken related term {linked}")
        for locale in ("en", "ru"):
            if not record["locales"][locale]["title"].strip():
                errors.append(f"term {number}: empty {locale} title")
        en, ru = record["locales"]["en"], record["locales"]["ru"]
        if len(en.get("aliases", [])) != len(ru.get("aliases", [])):
            errors.append(f"term {number}: aliases untranslated")
        if len(record["metadata"].get("categories", [])) != len(record["metadata"].get("categories_ru", [])):
            errors.append(f"term {number}: categories untranslated")
        media_ids = {item["id"] for item in record["media"]}
        for section in record["sections"]:
            for identifier in section["media_ids"]:
                if identifier not in media_ids:
                    errors.append(f"term {number}: broken media {identifier}")
        for field in fields(record):
            if field["kind"] not in ("title", "alias", "category") and not field["ru"]:
                errors.append(f"term {number}: untranslated {field['ru_path']}")
        for item in record["media"]:
            old = media.setdefault(item["id"], item)
            for field in ("title_ru", "caption_ru", "short_caption_ru", "alt_ru"):
                if old.get(field) and item.get(field) and old[field] != item[field]:
                    errors.append(f"media {item['id']}: inconsistent {field}")
    for identifier, item in media.items():
        for field, hash_field, label in (("original_path", "sha256", "original"), ("ru_path", "ru_sha256", "RU")):
            if (skip_original_media and field == "original_path") or not item.get(field):
                continue
            path = local_file(DATA, item[field])
            if path is None or checksum(path) != item.get(hash_field):
                errors.append(f"media {identifier}: missing/changed {label} image")
    return {"records": len(records), "unique_ids": len(ids), "unique_media": len(media), "image_files": sum(bool(item.get("original_path")) for item in media.values()), "translated_titles": sum(bool(record["locales"]["ru"]["title"]) for record in records), "errors": errors, "error_counts": dict(Counter(error.split(":")[-1].strip() for error in errors))}


def check_reviews(*, skip_original_media: bool = False) -> dict:
    terms = {int(path.stem) for path in (DATA / "processed" / "terms").glob("*.json")}
    records = {
        f"article:{path.stem}": json.loads(path.read_bytes())
        for path in (DATA / "processed" / "articles").glob("oae-*.json")
    }
    errors = [] if records else ["reviews: no review records found"]
    media_checked = set()
    missing = 0
    types = Counter()
    for record_id, record in records.items():
        source = record["source"]
        kind = source["type"]
        types[kind] += 1
        snapshot_url = source.get("pdf_url") or source["url"]
        filename = Path(urlsplit(snapshot_url).path).name
        if kind == "oae_review_topic":
            filename += ".html"
        snap = current_snapshot(DATA / "raw" / "reviews" / filename)
        if record["id"] != record_id or not snap.is_file() or checksum(snap) != source["sha256"]:
            errors.append(f"{record_id}: source ID or checksum mismatch")
        metadata = record["metadata"]
        for relation in ("document_ids", "topic_ids"):
            for other in metadata.get(relation, []):
                inverse = "topic_ids" if relation == "document_ids" else "document_ids"
                if other not in records or record_id not in records[other]["metadata"].get(inverse, []):
                    errors.append(f"{record_id}: missing reciprocal reference to {other}")
        for number in metadata.get("term_numbers", []):
            if number not in terms:
                errors.append(f"{record_id}: unknown term {number}")
        section_ids = {item["id"] for item in record["sections"]}
        for locale in ("ru", "en"):
            if section_ids != set(record["locales"][locale]["blocks"]):
                errors.append(f"{record_id}: section ID mismatch in {locale}")
        missing += sum(not field["ru"].strip() for field in fields(record))
        attached = {identifier for section in record["sections"] for identifier in section["media_ids"]}
        if attached != {item["id"] for item in record["media"]}:
            errors.append(f"{record_id}: media links differ from attached media")
        composite_ids = {item["id"] for item in record["media"] if item.get("component_ids")}
        for item in record["media"]:
            components = set(item.get("component_ids", []))
            if components and (
                components & composite_ids
                or any(
                    not components.issubset(section["media_ids"])
                    for section in record["sections"]
                    if item["id"] in section["media_ids"]
                )
            ):
                errors.append(f"{record_id}: invalid media components for {item['id']}")
            for filename_field, hash_field in (("original_path", "sha256"), ("ru_path", "ru_sha256")):
                expected = item.get(filename_field)
                if not expected or expected in media_checked or (skip_original_media and filename_field == "original_path"):
                    continue
                media_checked.add(expected)
                path = local_file(DATA, expected)
                if path is None or checksum(path) != item.get(hash_field):
                    errors.append(f"{record_id}: bad media {item['id']} {filename_field}")
    if missing:
        errors.append(f"reviews: {missing} untranslated fields")
    return {"records": len(records), "types": dict(types), "media_files_checked": len(media_checked),
            "untranslated_fields": missing, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-original-media", action="store_true", help="Check a fresh Git checkout without downloaded originals")
    parser.add_argument("--reviews", action="store_true")
    args = parser.parse_args()
    run = check_reviews if args.reviews else check
    report = run(skip_original_media=args.skip_original_media)
    print(json.dumps({key: value for key, value in report.items() if key != "errors"}, ensure_ascii=False, indent=2))
    if report["errors"]:
        print("first errors:", *report["errors"][:30], sep="\n")
        raise SystemExit(1)
