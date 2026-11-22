"""Check that the deployed media files match the canonical records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.catalog.media import local_file
from app.infra.config import data_directory
from app.infra.files import checksum, write_json


def inspect(root: Path, *, verify_hashes: bool = False) -> dict:
    seen: set[str] = set()
    missing: list[dict[str, str]] = []
    invalid: list[dict[str, str]] = []
    present = 0
    checked = 0
    for folder in ("processed/terms", "processed/articles"):
        for file in sorted((root / folder).glob("*.json")):
            record = json.loads(file.read_text(encoding="utf-8"))
            for media in record.get("media", []):
                for field, digest_field in (("original_path", "sha256"), ("ru_path", "ru_sha256")):
                    filename = media.get(field)
                    if not filename:
                        continue
                    key = f"{field}:{filename}"
                    if key in seen:
                        continue
                    seen.add(key)
                    local = local_file(root, filename)
                    if local is None:
                        missing.append({
                            "record_id": record["id"], "media_id": media["id"],
                            "field": field, "path": filename,
                        })
                    else:
                        present += 1
                        expected = media.get(digest_field)
                        if verify_hashes and expected:
                            checked += 1
                            if checksum(local) != expected:
                                invalid.append({
                                    "record_id": record["id"], "media_id": media["id"],
                                    "field": field, "path": filename,
                                })
    return {
        "expected_files": len(seen), "present_files": present, "hashes_checked": checked,
        "missing_count": len(missing), "checksum_errors": len(invalid),
        "missing": missing, "invalid": invalid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=data_directory())
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = inspect(args.data_dir, verify_hashes=args.verify_hashes)
    if args.report:
        write_json(args.report, result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"missing", "invalid"}}))
    for item in [*result["missing"][:5], *result["invalid"][:5]]:
        print(json.dumps(item, ensure_ascii=False))
    if args.strict and (result["missing_count"] or result["checksum_errors"]):
        raise SystemExit("Media deployment integrity check failed")


if __name__ == "__main__":
    main()
