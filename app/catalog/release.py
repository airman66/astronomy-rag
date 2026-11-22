"""Validate the immutable artifacts selected for this process."""

import hashlib
import json
from pathlib import Path

from app.catalog.corpus import corpus_digest
from app.catalog.media import inspect_media_files, local_file, review_pdf
from app.infra.files import checksum

RELEASE_MANIFEST = "release-manifest.json"
RELEASE_SCHEMA_VERSION = 2
INDEX_FILES = ("chunks.jsonl", "chunk-manifest.json", "index-manifest.json", "index.faiss", "vector-map.json")


def _illustrations_digest(root: Path) -> str | None:
    manifest = root / "illustrations" / "manifest.json"
    try:
        content = manifest.read_bytes()
    except FileNotFoundError:
        return None

    for item in json.loads(content).get("items", []):
        if item["status"] in ("generated", "approved"):
            path = local_file(root, item.get("path"))
            number = item["term_number"]
            if path is None:
                raise ValueError(f"Published cover for term {number} is missing")
            if checksum(path) != item.get("sha256"):
                raise ValueError(f"Published cover for term {number} has a checksum mismatch")
    return hashlib.sha256(content).hexdigest()


def _review_pdf_digests(root: Path) -> dict[str, str]:
    digests = {}
    for file in sorted((root / "processed" / "articles").glob("*.json")):
        record = json.loads(file.read_bytes())
        if record["source"].get("type") != "oae_review_document":
            continue
        for locale in ("en", "ru"):
            path = review_pdf(root, record, locale)
            if path is None:
                raise ValueError(f"Published PDF {locale} for {record['id']} is missing")
            digest = checksum(path)
            if locale == "en" and digest != record["source"].get("sha256"):
                raise ValueError(f"Published PDF en for {record['id']} has a checksum mismatch")
            digests[path.relative_to(root).as_posix()] = digest
    return digests


def release_manifest(root: Path) -> dict:
    root = root.resolve()
    return {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "corpus_sha256": corpus_digest(root),
        "illustrations_sha256": _illustrations_digest(root),
        "review_pdf_sha256": _review_pdf_digests(root),
        "catalog_sha256": checksum(root / "glossary.db"),
        "index_sha256": {name: checksum(root / "index" / name) for name in INDEX_FILES},
    }


def verify_release(root: Path) -> None:
    root = root.resolve(strict=True)
    expected = json.loads((root / RELEASE_MANIFEST).read_bytes())
    if expected.get("schema_version") != RELEASE_SCHEMA_VERSION:
        raise ValueError("Unsupported data release manifest version; publish a new release")
    actual = release_manifest(root)
    if actual["review_pdf_sha256"] != expected.get("review_pdf_sha256"):
        raise ValueError("Published PDF checksums do not match the release manifest")
    if expected != actual:
        raise ValueError("Published data release does not match its manifest")
    verify_media_files(root)


def verify_media_files(root: Path) -> None:
    report = inspect_media_files(root, require_hashes=True)
    if report["missing"]:
        raise ValueError(f"Published media file is missing: {report['missing'][0]['path']}")
    if report["invalid"]:
        raise ValueError(f"Published media file has a missing or incorrect checksum: {report['invalid'][0]['path']}")
