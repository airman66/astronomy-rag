"""Inventory visible labels before ChatGPT image localization."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from app.infra.config import data_directory
from app.infra.files import write_json


def scan(path: Path) -> dict:
    import pytesseract
    from PIL import Image
    from pytesseract import Output

    try:
        with Image.open(path) as image:
            dimensions = image.size
            image.thumbnail((2400, 2400))
            recognized = pytesseract.image_to_data(
                image.convert("RGB"), lang="eng", output_type=Output.DICT, config="--psm 11", timeout=45
            )
        words = []
        for i, value in enumerate(recognized["text"]):
            if value.strip() and float(recognized["conf"][i]) >= 45:
                words.append(
                    {
                        "text": value.strip(),
                        "confidence": float(recognized["conf"][i]),
                        "box": [recognized[key][i] for key in ("left", "top", "width", "height")],
                    }
                )
        return {"path": str(path), "size": dimensions, "words": words, "error": None}
    except Exception as exc:
        return {"path": str(path), "words": [], "error": str(exc)[:160]}


def labels(path: Path, language: str) -> list[str]:
    import pytesseract
    from PIL import Image

    with Image.open(path) as image:
        text = pytesseract.image_to_string(image.convert("RGB"), lang=language, config="--psm 11")
    return [line.strip() for line in text.splitlines() if line.strip()]


def ocr_latin(path: Path) -> str:
    result = subprocess.run(
        ["tesseract", str(path), "stdout", "-l", "eng+rus", "tsv"],
        capture_output=True,
        text=True,
        timeout=25,
        check=False,
    )
    words = []
    for line in result.stdout.splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) >= 12:
            try:
                confidence = float(fields[10])
            except ValueError:
                continue
            word = fields[11].strip()
            if confidence >= 60 and re.search(r"[A-Za-z]{4,}", word):
                words.append(word)
    return " ".join(words[:12])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("diagram", "media", "all"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    data = data_directory()
    paths = sorted(p for p in (data / "media/original").glob("*") if p.is_file())
    if args.kind != "all":
        paths = [path for path in paths if path.name.startswith(args.kind + "_")]
    existing_file = data / "image_ocr.json"
    existing = (
        {row["path"]: row for row in json.loads(existing_file.read_text())} if existing_file.exists() else {}
    )
    pending = [path for path in paths if str(path) not in existing]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(scan, path) for path in pending]):
            row = future.result()
            existing[row["path"]] = row
            if len(existing) % 20 == 0:
                write_json(existing_file, list(existing.values()))
                print(f"audited {len(existing)}", flush=True)
    write_json(existing_file, list(existing.values()))
    print(f"audited={len(paths)} with_text={sum(bool(existing[str(path)]['words']) for path in paths)}")
