"""Content identity shared by the canonical JSON, SQLite and FAISS."""

import hashlib
from pathlib import Path

from app.infra.files import checksum


def record_files(root: Path) -> list[Path]:
    return sorted((root / "processed" / "terms").glob("*.json")) + sorted(
        (root / "processed" / "articles").glob("*.json")
    )


def corpus_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in record_files(root):
        entry = f"{path.relative_to(root).as_posix()} {checksum(path)}\n"
        digest.update(entry.encode())
    return digest.hexdigest()
