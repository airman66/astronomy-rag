"""Command entry point for python -m scripts.eval_scope."""

import asyncio

from scripts.evaluation.scope import main

if __name__ == "__main__":
    asyncio.run(main())
