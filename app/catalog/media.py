"""Resolve paths recorded in raw media metadata and illustration manifests."""

import json
from pathlib import Path

from app.catalog.corpus import record_files
from app.infra.files import checksum

IMAGE_SUFFIXES = frozenset((".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"))
REVIEW_SOURCE_PDFS = Path("raw/reviews")
REVIEW_TRANSLATED_PDFS = Path("media/ru/review-pdf")


def local_file(data_root: Path, filename: str | None) -> Path | None:
    if not filename:
        return None
    root = data_root.resolve()
    stored = Path(filename)
    if stored.is_absolute():
        path = stored
    elif stored.parts[:1] == ("data",):
        path = root.joinpath(*stored.parts[1:])
    elif data_root.parts and stored.parts[: len(data_root.parts)] == data_root.parts:
        path = stored
    else:
        path = root / stored
    path = path.resolve()
    if path.is_relative_to(root) and path.is_file():
        return path
    # Older imports sometimes recorded the path with the original DATA_DIR prefix.
    if ".." not in stored.parts and stored.parts[-3:-1] in (("media", "original"), ("media", "ru")):
        relocated = root.joinpath(*stored.parts[-3:]).resolve()
        if relocated.is_relative_to(root) and relocated.is_file():
            return relocated
    return None


def review_pdf(root: Path, record: dict, locale: str) -> Path | None:
    source = record["source"]
    if source.get("type") != "oae_review_document":
        return None
    if locale == "ru":
        directory = REVIEW_TRANSLATED_PDFS
        filename = str((root / directory / (record["id"].removeprefix("article:") + ".pdf")).absolute())
    else:
        directory = REVIEW_SOURCE_PDFS
        filename = source.get("pdf_path")
    path = local_file(root, filename)
    if path and path.suffix == ".pdf" and path.is_relative_to((root / directory).resolve()):
        return path
    return None


def localized_filename(item: dict, locale: str) -> str | None:
    if locale == "ru" and item.get("ru_path"):
        return item["ru_path"]
    return item.get("original_path")


def localized_digest(item: dict, locale: str) -> str | None:
    return item.get("ru_sha256") if locale == "ru" and item.get("ru_path") else item.get("sha256")


def media_file_url(identifier: str, locale: str, digest: str | None) -> str:
    url = f"/api/v1/media/{identifier}/file?locale={locale}"
    return f"{url}&v={digest}" if digest else url


def inspect_media_files(root: Path, *, verify_hashes: bool = False, require_hashes: bool = False) -> dict:
    """Check media references, hashing each deployed path at most once."""
    paths: dict[tuple[str, str], Path | None] = {}
    checksums: dict[tuple[str, str], str] = {}
    missing: list[dict[str, str]] = []
    invalid: list[dict[str, str]] = []
    for file in record_files(root):
        record = json.loads(file.read_text(encoding="utf-8"))
        for media in record.get("media", []):
            for field, digest_field in (("original_path", "sha256"), ("ru_path", "ru_sha256")):
                if not (filename := media.get(field)):
                    continue
                item = {"record_id": record["id"], "media_id": media["id"], "field": field, "path": filename}
                key = field, filename
                if key not in paths:
                    paths[key] = local_file(root, filename)
                    if paths[key] is None:
                        missing.append(item)
                local = paths[key]
                if local is None or not (verify_hashes or require_hashes):
                    continue
                if expected := media.get(digest_field):
                    if key not in checksums:
                        checksums[key] = checksum(local)
                    if checksums[key] != expected:
                        invalid.append(item)
                elif require_hashes:
                    invalid.append(item)
    return {
        "expected_files": len(paths), "present_files": len(paths) - len(missing), "hashes_checked": len(checksums),
        "missing_count": len(missing), "checksum_errors": len(invalid),
        "missing": missing, "invalid": invalid,
    }
