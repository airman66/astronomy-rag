"""Validate the immutable artifacts selected for this process."""

import json
from pathlib import Path

from app.catalog.corpus import corpus_digest
from app.infra.files import checksum

RELEASE_MANIFEST = "release-manifest.json"
INDEX_FILES = ("chunks.jsonl", "chunk-manifest.json", "index-manifest.json", "index.faiss", "vector-map.json")


def release_manifest(root: Path) -> dict:
    illustrations = root / "illustrations" / "manifest.json"
    return {
        "schema_version": 1,
        "corpus_sha256": corpus_digest(root),
        "illustrations_sha256": checksum(illustrations) if illustrations.exists() else None,
        "catalog_sha256": checksum(root / "glossary.db"),
        "index_sha256": {name: checksum(root / "index" / name) for name in INDEX_FILES},
    }


def verify_release(root: Path) -> None:
    expected = json.loads((root / RELEASE_MANIFEST).read_bytes())
    if expected != release_manifest(root):
        raise ValueError("Published data release does not match its manifest")
