"""Find and import individually licensed Wikimedia covers for glossary terms.

Each request addresses one article or file. No paid image generation or batch API is used.
The candidate cache makes discovery and downloads resumable.
"""

import argparse

from scripts.covers.importers import (
    import_all,
    import_astro,
    import_curated,
    import_related,
)
from scripts.covers.report import write_review
from scripts.covers.sources import discover_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "discover",
            "import",
            "astro-fallback",
            "astro-curated",
            "curated",
            "related",
            "report",
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "discover":
        discover_all(args.workers)
    elif args.command == "astro-fallback":
        import_astro()
    elif args.command == "astro-curated":
        import_astro(curated=True)
    elif args.command == "report":
        write_review()
    elif args.command == "related":
        import_related()
    elif args.command == "curated":
        import_curated()
    else:
        import_all(args.workers)


if __name__ == "__main__":
    main()
