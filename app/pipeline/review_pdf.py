"""Extract ordered page text, using OCR where a PDF's text layer is broken."""

import os
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory


def page_links(path: Path) -> dict[int, list[dict[str, str]]]:
    output = subprocess.check_output(["pdfinfo", "-url", path], text=True)
    pages: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for line in output.splitlines():
        if match := re.fullmatch(r"\s*(\d+)\s+Annotation\s+(https?://\S+)\s*", line):
            number = int(match[1])
            url = re.sub(r"^(https?://)(?:%20)+(?=[A-Za-z0-9])", r"\1", match[2])
            pages[number][url] = {"url": url, "text": url, "text_ru": url}
    return {number: list(urls.values()) for number, urls in pages.items()}


def extract_pages(path: Path) -> list[dict[str, str]]:
    text = subprocess.check_output(["pdftotext", "-enc", "UTF-8", path, "-"], text=True)
    pages = text.split("\f")[:-1]
    if not pages:
        raise ValueError(f"Cannot extract PDF pages from {path}")

    with TemporaryDirectory(prefix="oae-review-ocr-") as directory:
        prefix = Path(directory)

        def read_page(item: tuple[int, str]) -> dict[str, str]:
            number, page = item
            if "\ufffd" not in page:
                return {"text": page, "method": "embedded"}
            image = prefix / f"page-{number}"
            subprocess.run(
                [
                    "pdftoppm",
                    "-f",
                    str(number),
                    "-l",
                    str(number),
                    "-singlefile",
                    "-r",
                    "180",
                    "-gray",
                    "-png",
                    path,
                    image,
                ],
                check=True,
            )
            recovered = subprocess.check_output(
                ["tesseract", image.with_suffix(".png"), "stdout", "-l", "eng"],
                env={**os.environ, "OMP_THREAD_LIMIT": "1"},
                text=True,
            )
            return {"text": recovered, "method": "ocr"}

        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(read_page, enumerate(pages, 1)))
