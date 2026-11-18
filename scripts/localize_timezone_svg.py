"""Command entry point for python -m scripts.localize_timezone_svg."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.media.timezone import main

if __name__ == "__main__":
    main()
