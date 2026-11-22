"""Validate the immutable artifacts selected for this process."""

import hashlib
import json
import os
import re
from pathlib import Path

from app.catalog import storage
from app.catalog.corpus import corpus_digest
from app.catalog.index import ARTIFACT_HASH_FIELDS, INDEX_FILES, check_chunk_manifest, read_manifest
from app.catalog.media import inspect_media_files, local_file, review_pdf
from app.infra.files import checksum

RELEASE_MANIFEST = "release-manifest.json"
RELEASE_SCHEMA_VERSION = 3


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


def release_manifest(root: Path, *, schema_version: int = RELEASE_SCHEMA_VERSION) -> dict:
    root = root.resolve()
    digest = corpus_digest(root)
    index = read_manifest(root / "index")
    check_chunk_manifest(root / "index", index)
    if index["corpus_sha256"] != digest or storage.database_corpus(root / "glossary.db") != digest:
        raise ValueError("Release catalogue and index must match the canonical corpus")
    index_sha = {name: checksum(root / "index" / name) for name in INDEX_FILES}
    for name, field in ARTIFACT_HASH_FIELDS.items():
        if index_sha[name] != index[field]:
            raise ValueError(f"Index file checksum mismatch: {name}")
    manifest = {
        "schema_version": schema_version,
        "corpus_sha256": digest,
        "illustrations_sha256": _illustrations_digest(root),
        "review_pdf_sha256": _review_pdf_digests(root),
        "catalog_sha256": checksum(root / "glossary.db"),
        "index_sha256": index_sha,
    }
    if schema_version == RELEASE_SCHEMA_VERSION:
        models = os.readlink(root / "models")
        if Path(models).is_absolute():
            raise ValueError("Release model link must be relative")
        manifest["models_path"] = models
    return manifest


def verify_release(root: Path, *, expected_id: str | None = None) -> None:
    root = root.resolve(strict=True)
    content = (root / RELEASE_MANIFEST).read_bytes()
    expected = json.loads(content)
    version = expected.get("schema_version")
    if version not in (2, RELEASE_SCHEMA_VERSION):
        raise ValueError("Unsupported data release manifest version; publish a new release")
    actual = release_manifest(root, schema_version=version)
    if actual["review_pdf_sha256"] != expected.get("review_pdf_sha256"):
        raise ValueError("Published PDF checksums do not match the release manifest")
    if expected != actual:
        raise ValueError("Published data release does not match its manifest")
    if expected_id is None and re.fullmatch(r"[0-9a-f]{64}", root.name):
        expected_id = root.name
    if expected_id is not None and hashlib.sha256(content).hexdigest() != expected_id:
        raise ValueError("Published manifest does not match its release ID")
    verify_media_files(root)


def verify_media_files(root: Path) -> None:
    report = inspect_media_files(root, require_hashes=True)
    if report["missing"]:
        raise ValueError(f"Published media file is missing: {report['missing'][0]['path']}")
    if report["invalid"]:
        raise ValueError(f"Published media file has a missing or incorrect checksum: {report['invalid'][0]['path']}")
