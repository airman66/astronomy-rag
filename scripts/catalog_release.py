"""Build, verify and atomically publish a coordinated catalogue release."""

import argparse
import fcntl
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from app.catalog import storage
from app.catalog.index import INDEX_FILES
from app.catalog.release import RELEASE_MANIFEST, release_manifest, verify_media_files, verify_release
from app.infra.config import data_directory
from app.infra.files import checksum, write_json


def prepare_catalog() -> None:
    storage.DB.parent.mkdir(parents=True, exist_ok=True)
    with storage.DB.with_name(f".{storage.DB.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        storage.ensure_database(force_check=True, allow_stale=False)


def publish(source: Path, releases: Path) -> Path:
    from scripts.index.verify import verify_index

    releases.mkdir(parents=True, exist_ok=True)
    current = releases / "current"
    if current.exists() and not current.is_symlink():
        raise ValueError("Release pointer must be a symlink")
    with tempfile.TemporaryDirectory(prefix=".build-", dir=releases) as directory:
        stage = Path(directory) / "release"
        for relative in ("processed/terms", "processed/articles", "media", "raw/reviews"):
            original = source / relative
            if original.is_dir():
                shutil.copytree(original, stage / relative)
        illustrations = source / "illustrations" / "manifest.json"
        if illustrations.is_file():
            (stage / "illustrations").mkdir()
            shutil.copy2(illustrations, stage / "illustrations" / illustrations.name)
        (stage / "index").mkdir()
        for name in INDEX_FILES:
            shutil.copy2(source / "index" / name, stage / "index" / name)
        # Model downloads are independent of the corpus; the runtime verifies GGUF bytes.
        models = os.path.relpath((source / "models").resolve(), releases.resolve() / "version")
        (stage / "models").symlink_to(models, target_is_directory=True)
        verify_index(stage / "index", require_current_chunking=True)
        verify_media_files(stage)
        storage.build_database(stage, stage / "glossary.db", immutable=True)
        write_json(stage / RELEASE_MANIFEST, release_manifest(stage))
        release = releases / checksum(stage / RELEASE_MANIFEST)
        if not release.exists():
            try:
                stage.rename(release)
            except OSError:
                if not release.is_dir():
                    raise
        if stage.exists():  # The version already existed and may have changed since it was built.
            verify_release(release, expected_id=release.name)
        link = releases / f".current-{uuid.uuid4().hex}"
        try:
            link.symlink_to(release.name, target_is_directory=True)
            link.replace(current)
        finally:
            link.unlink(missing_ok=True)
    return release


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    build = commands.add_parser("publish")
    build.add_argument("releases", type=Path)
    build.add_argument("--source", default=data_directory(), type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("release", type=Path)
    args = cli.parse_args()
    if args.command == "publish":
        print(json.dumps({"release": str(publish(args.source, args.releases))}))
    elif args.command == "verify":
        verify_release(args.release)
    else:
        prepare_catalog()


if __name__ == "__main__":
    main()
