"""Check that the deployed media files match the canonical records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.catalog.media import inspect_media_files as inspect
from app.infra.config import data_directory
from app.infra.files import write_json


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
