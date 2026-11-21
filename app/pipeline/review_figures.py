"""Extract the artwork of OAE PDFs and attach it to its originating pages."""

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from app.catalog.media import local_file
from app.infra.config import data_directory
from app.infra.files import write_json
from app.pipeline.importer import digest, write_immutable

EXOPLANET_DIAGRAMS = {
    2: [(180, 108, 430, 369)],
    4: [(90, 124, 529, 374)],
    5: [(150, 92, 485, 262)],
    8: [(120, 104, 506, 434)],
    9: [(133, 171, 476, 304)],
}


def cropped_vector_art(pdf: Any, page: Any, filename: str) -> list[tuple[bytes, str]]:
    if filename == "mini-review-exoplanets.pdf":
        rectangles = [
            pdf.Rect(box) for box in EXOPLANET_DIAGRAMS.get(page.number + 1, [])
        ]
    else:
        rectangles = []
        for box in page.cluster_drawings(x_tolerance=6, y_tolerance=6):
            if (
                box.width < 35
                or box.height < 25
                or box.get_area() >= 0.75 * page.rect.get_area()
            ):
                continue
            if box.y1 < 65 or box.y0 > page.rect.y1 - 85:
                continue
            rectangles.append(
                pdf.Rect(box.x0 - 8, box.y0 - 8, box.x1 + 8, box.y1 + 8) & page.rect
            )
    return [
        (page.get_pixmap(clip=box, dpi=160).tobytes("png"), "png") for box in rectangles
    ]


def embedded_images(pdf: Any, page: Any) -> list[tuple[bytes, str]]:
    result = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 1:
            continue
        data, extension = block["image"], block["ext"]
        mask = block.get("mask")
        if mask or extension not in {"png", "jpeg", "jpg", "webp", "gif"}:
            pixels = pdf.Pixmap(data)
            if mask and not pixels.alpha:
                pixels = pdf.Pixmap(pixels, pdf.Pixmap(mask))
            if pixels.n - pixels.alpha > 3:
                pixels = pdf.Pixmap(pdf.csRGB, pixels)
            data, extension = pixels.tobytes("png"), "png"
        result.append((data, extension))
    return result


def attach_figures(pdf: Any, data: Path, record: dict) -> int:
    original = local_file(data, record["source"]["pdf_path"])
    if not original:
        raise ValueError(f"PDF missing for {record['id']}")
    slug = record["id"].removeprefix("article:oae-document-")
    prefix = f"media:oae-pdf-{slug}-"
    media = {m["id"]: m for m in record["media"]}
    sections = {s["id"]: s for s in record["sections"]}
    for section in sections.values():
        section["media_ids"] = [
            i for i in section["media_ids"] if not i.startswith(prefix)
        ]
    new = 0
    with pdf.open(original) as document:
        for page in document:
            number = page.number + 1
            figures = [
                *embedded_images(pdf, page),
                *cropped_vector_art(pdf, page, original.name),
            ]
            if not figures:
                continue
            section_id = f"page-{number}"
            if section_id not in sections:
                en, ru = f"Page {number}", f"Страница {number}"
                section = {
                    "id": section_id,
                    "type": "media_collection",
                    "heading": en,
                    "body": "",
                    "heading_ru": ru,
                    "body_ru": "",
                    "links": [],
                    "media_ids": [],
                }
                sections[section_id] = section
                for locale, heading in (("en", en), ("ru", ru)):
                    record["locales"][locale]["blocks"][section_id] = {
                        "heading": heading,
                        "body": "",
                    }
            ids = sections[section_id]["media_ids"]
            for content, ext in figures:
                checksum = digest(content)
                identifier = f"{prefix}{checksum[:20]}"
                if identifier not in media:
                    relative = Path("media/original/review-pdf") / f"{checksum}.{ext}"
                    write_immutable(data / relative, content)
                    media[identifier] = {
                        "id": identifier,
                        "kind": "media",
                        "file_type": "img",
                        "source_url": record["source"]["pdf_url"] + f"#page={number}",
                        "record_url": record["source"]["url"],
                        "original_path": relative.as_posix(),
                        "sha256": checksum,
                        "size": len(content),
                        "review": {"status": "draft", "reviewer": None},
                    }
                    new += 1
                if identifier not in ids:
                    ids.append(identifier)
                if (
                    media[identifier].get("title_en")
                    == "Illustration from the original PDF"
                ):
                    del media[identifier]["title_en"]
                    media[identifier].pop("title_ru", None)
    record["sections"] = sorted(
        sections.values(), key=lambda item: int(item["id"].removeprefix("page-"))
    )
    attached = {
        identifier
        for section in sections.values()
        for identifier in section["media_ids"]
    }
    record["media"] = [
        item for identifier, item in media.items() if identifier in attached
    ]
    return new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", default="oae-document-*.json")
    args = parser.parse_args()
    pdf = importlib.import_module("pymupdf")
    data = data_directory()
    for path in sorted((data / "processed" / "articles").glob(args.glob)):
        record = json.loads(path.read_bytes())
        if record["source"]["type"] != "oae_review_document":
            continue
        added = attach_figures(pdf, data, record)
        write_json(path, record)
        print(
            f"{record['id']}: {added} added, {len(record['media'])} total", flush=True
        )


if __name__ == "__main__":
    main()
