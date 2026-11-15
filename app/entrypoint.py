"""Adopt the former .NET cache volume, then run the API as the existing non-root user."""

import os
import sys
from pathlib import Path


def main():
    if os.getuid() == 0:
        directory = Path(os.getenv("CATALOG_DATA_DIR", "/app/catalog"))
        directory.mkdir(parents=True, exist_ok=True)
        # Only the known catalog snapshots belong to this migration.
        for path in [
            directory,
            *(
                directory / name
                for name in (
                    "nasa-planets.json",
                    "hwc.json",
                    "nasa-planets.json.tmp",
                    "hwc.json.tmp",
                )
            ),
        ]:
            if path.exists() and not path.is_symlink():
                os.chown(path, 10001, 10001)
        os.setgroups([])
        os.setgid(10001)
        os.setuid(10001)
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
