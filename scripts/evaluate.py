"""Command entry point for python -m scripts.evaluate."""

import asyncio

from scripts.evaluation.retrieval import main

if __name__ == "__main__":
    asyncio.run(main())
