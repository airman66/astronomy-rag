"""Export printable Russian editions from the normalized OAE documents."""

import argparse
import importlib
import json
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlsplit

from app.catalog.media import local_file, localized_filename
from app.infra.config import data_directory

CSS = """
body { font-family: sans-serif; font-size: 10pt; line-height: 1.35; color: #172032; }
h1 { font-size: 22pt; line-height: 1.15; color: #172d49; }
h2 { font-size: 13pt; color: #254c74; margin-top: 12pt; }
p { margin: 0 0 6pt; }
small { font-size: 8pt; color: #45596a; }
a { color: #174f87; }
.picture { margin: 6pt 0; }
"""


def image_html(pdf: Any, data: Path, item: dict) -> str:
    path = local_file(data, localized_filename(item, "ru"))
    if path is None:
        raise ValueError(f"Missing artwork: {item['id']}")
    pixels = pdf.Pixmap(str(path))
    width, height = pixels.width, pixels.height
    image_width = min(480, width, 470 * width / height)
    source = escape(path.relative_to(data).as_posix())
    alt = escape(item.get("alt_ru") or "")
    caption = escape(item.get("caption_ru") or item.get("short_caption_ru") or item.get("alt_ru") or "")
    return (
        f'<p class="picture"><img src="{source}" alt="{alt}" width="{image_width:.1f}pt">'
        f'<br><small>{caption}</small></p>'
    )


def web_link(url: str, label: str) -> str:
    if urlsplit(url).scheme not in {"https", "http"}:
        return escape(label)
    return f'<a href="{escape(url)}">{escape(label)}</a>'


def export_record(
    pdf: Any, data: Path, record: dict, archive: Any, destination: Path
) -> tuple[Path, int]:
    title = record["locales"]["ru"]["title"]
    source = record["source"]
    authors = source.get("authors") or ""
    if isinstance(authors, list):
        authors = ", ".join(authors)
    attribution = " · ".join(filter(None, (authors, source.get("license"))))
    body = [
        f"<h1>{escape(title)}</h1>",
        f"<p>{escape(attribution)}</p>",
        f'<p><small>{web_link(source["pdf_url"], "Оригинал Astro4Edu")}'
        f' · Статус перевода: {escape(record["review"]["status"])}</small></p>',
    ]
    images = {item["id"]: item for item in record["media"]}
    image_cache: dict[tuple[str, str, str], str] = {}
    for number, section in enumerate(record["sections"], 1):
        page = section["id"].removeprefix("page-")
        heading = section.get("heading_ru") or f"Страница {page}"
        body.append(f'<h2 id="source-page-{number}">{escape(heading)}</h2>')
        translation = section.get("body_ru") or ""
        if section.get("body") and not translation.strip():
            raise ValueError(f"Missing translation: {record['id']}/{section['id']}")
        for paragraph in translation.split("\n\n"):
            if paragraph.strip():
                lines = "<br>".join(escape(line) for line in paragraph.splitlines())
                body.append(f"<p>{lines}</p>")
        covered_components: set[str] = set()
        for media_id in section.get("media_ids", []):
            covered_components.update(images[media_id].get("component_ids", []))
        shown_images: set[tuple[str, str, str]] = set()
        for media_id in section.get("media_ids", []):
            if media_id in covered_components:
                continue
            item = images[media_id]
            key = (
                localized_filename(item, "ru") or "",
                item.get("alt_ru") or "",
                item.get("caption_ru") or item.get("short_caption_ru") or "",
            )
            if key in shown_images:
                continue
            shown_images.add(key)
            if key not in image_cache:
                image_cache[key] = image_html(pdf, data, item)
            body.append(image_cache[key])
        links = {
            link["url"]: link.get("text_ru") or link["url"]
            for link in section.get("links", [])
        }
        if links:
            body.append("<p><small>" + " · ".join(web_link(url, label) for url, label in links.items()) + "</small></p>")
    paper = pdf.paper_rect("a4")
    area = paper + (52, 48, -52, -48)
    outline: dict[str, list] = {}

    def position(item: Any) -> None:
        if item.id and item.id.startswith("source-page-") and item.open_close & 1:
            outline[item.id] = [1, item.text, item.page_num]

    story = pdf.Story(html="".join(body), user_css=CSS, archive=archive)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / (record["id"].removeprefix("article:") + ".pdf")
    with story.write_with_links(lambda _i, _filled: (paper, area, None), position) as document:
        document.set_metadata({"title": title, "author": authors, "subject": source["pdf_url"]})
        document.set_toc(list(outline.values()))
        document.subset_fonts()
        pages = len(document)
        with TemporaryDirectory(dir=destination) as directory:
            temporary = Path(directory) / path.name
            document.save(temporary, garbage=4, deflate=True, no_new_id=True)
            temporary.replace(path)
    return path, pages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", default="oae-document-*.json")
    parser.add_argument("--output-dir", type=Path, help="export to a separate directory")
    args = parser.parse_args()
    pdf = importlib.import_module("pymupdf")
    data = data_directory().resolve()
    destination = args.output_dir or data / "media" / "ru" / "review-pdf"
    archive = pdf.Archive(str(data))
    for path in sorted((data / "processed" / "articles").glob(args.glob)):
        record = json.loads(path.read_text(encoding="utf-8"))
        output, pages = export_record(pdf, data, record, archive, destination)
        print(f"{output.name}: {pages} pages", flush=True)


if __name__ == "__main__":
    main()
