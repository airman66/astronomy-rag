"""Command entry point for python -m scripts.illustrations."""

import sys

from scripts.covers.manifest import main

if __name__ == "__main__":
    sys.exit(main())
