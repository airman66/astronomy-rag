"""Resolve paths recorded in raw media metadata and illustration manifests."""

from pathlib import Path

IMAGE_SUFFIXES = frozenset((".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"))


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
    return path if path.is_relative_to(root) and path.is_file() else None


def localized_filename(item: dict, locale: str) -> str | None:
    if locale == "ru" and item.get("ru_path"):
        return item["ru_path"]
    return item.get("original_path")


def localized_digest(item: dict, locale: str) -> str | None:
    return item.get("ru_sha256") if locale == "ru" and item.get("ru_path") else item.get("sha256")


def media_file_url(identifier: str, locale: str, digest: str | None) -> str:
    url = f"/api/v1/media/{identifier}/file?locale={locale}"
    return f"{url}&v={digest}" if digest else url
