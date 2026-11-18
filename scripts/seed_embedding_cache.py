"""Command entry point for python -m scripts.seed_embedding_cache."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.index.cache import seed_main

if __name__ == "__main__":
    seed_main()
