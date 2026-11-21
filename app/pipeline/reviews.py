"""Import Astro4Edu OAE review topics and first-party documents."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup, Tag

from app.infra.config import data_directory
from app.infra.files import write_json
from app.pipeline.importer import current_snapshot, digest, raw_bytes, save_snapshot
from app.pipeline.records import BASE, merge_media, merge_sections, now
from app.pipeline.review_media import (
    INLINE_IMAGE,
    cover_media,
    glossary_media,
    inline_media,
    review_media,
)
from app.pipeline.review_pdf import extract_pages, page_links

CATALOG = f"{BASE}/oae-reviews/"
TOPIC_PATH = re.compile(r"^/oae-reviews/([a-z0-9-]+)/?$")
TERM_PATH = re.compile(r"^/resources/glossary/term/(\d+)/?$")
DOCUMENT_PATH = re.compile(r"^/resources/document/([A-Za-z0-9]+)/?$")
PDF_PATH = re.compile(r"^/media/documents/([A-Za-z0-9_.-]+\.pdf)$")
HEADING = ("h2", "h3", "h4", "h5")
DRAFT = {"status": "draft", "reviewer": None, "notes": None}


def soup(content: bytes) -> BeautifulSoup:
    return BeautifulSoup(content, "html.parser")


def clean(text: str) -> str:
    return " ".join(text.split())


def direct_text(tag: Tag) -> str:
    return clean(
        " ".join(str(node) for node in tag.find_all(string=True, recursive=False))
    )


def links(root: Tag, url: str) -> list[dict]:
    result = {}
    for anchor in root.select("a[href]"):
        href = str(anchor["href"])
        if not href or href.startswith(("#", "javascript:")):
            continue
        target = urljoin(url, href)
        if urlsplit(target).scheme not in {"http", "https"}:
            continue
        label = clean(anchor.get_text(" ", strip=True))
        if not label and (image := anchor.find("img")):
            label = clean(str(image.get("alt", "")))
        if label:
            result.setdefault(target, {"url": target, "text": label, "text_ru": ""})
    return list(result.values())


def section(
    identifier: str, kind: str, heading: str, body: str, refs: list[dict]
) -> dict:
    return {
        "id": identifier,
        "type": kind,
        "heading": heading,
        "body": body,
        "heading_ru": "",
        "body_ru": "",
        "links": refs,
        "media_ids": [],
    }


def html_sections(
    page: BeautifulSoup, url: str, inline_images: dict[str, list[Tag]] | None = None
) -> list[dict]:
    result = []
    for area in page.select(".container[id]"):
        area_id = str(area["id"])
        if area_id == "other":
            break
        headings = [
            h
            for h in area.find_all(HEADING)
            if h.find_parent(class_="container", id=True) is area
        ]
        if not headings:
            continue
        # Older topic pages omit closing h4 tags. Rebuild each heading's content
        # from its slice of the original DOM, including nested list and card markup.
        pieces = re.split(r"(?=<h[2-5]\b)", area.decode_contents())
        current_id = area_id
        for piece in pieces[1:]:
            if "container" in piece:
                piece = re.split(
                    r"<div\b[^>]*\bclass=\"[^\"]*\bcontainer\b[^\"]*\"[^>]*\bid=",
                    piece,
                    maxsplit=1,
                )[0]
            fragment = BeautifulSoup(piece, "html.parser")
            heading = fragment.find(HEADING)
            if not heading:
                continue
            title = direct_text(heading) or clean(heading.get_text(" ", strip=True))
            if not title:
                continue
            display = [
                clean(block.get_text(" ", strip=True))
                for block in fragment.find_all("p")
                if not block.find("p")
            ]
            if area_id == "panel":
                for group in fragment.select(".staff"):
                    for line_break in group.find_all("br"):
                        line_break.replace_with("\n")
                    visible_lines = group.get_text(" ", strip=False).splitlines()
                    display.extend(
                        clean(line).replace("\u200b", "") for line in visible_lines
                    )
            refs = links(fragment, url)
            if not display and not refs and not fragment.find("img"):
                continue
            identifier = (
                current_id
                if current_id not in {s["id"] for s in result}
                else f"{current_id}-{len(result) + 1}"
            )
            kind = (
                "article_body"
                if area_id == "about"
                else "media_collection"
                if fragment.find("img")
                else "related_resources"
            )
            result.append(
                section(
                    identifier,
                    kind,
                    title,
                    "\n\n".join(dict.fromkeys(filter(None, display))),
                    refs,
                )
            )
            if inline_images is not None:
                for image in fragment.select(INLINE_IMAGE):
                    inline_images.setdefault(identifier, []).append(image)
    return result


def record(
    slug: str,
    title: str,
    source: dict,
    metadata: dict,
    sections: list[dict],
    media: list[dict],
    previous: dict,
) -> dict:
    old_source = previous.get("source", {})
    old_locales = previous.get("locales", {})
    old_en, old_ru = old_locales.get("en", {}), old_locales.get("ru", {})
    stale = dict(old_source.get("stale_translations", {}))

    def keep(path: str, en: str | None, ru: str | None) -> None:
        if ru:
            old = {"en": en, "ru": ru}
            history = stale.setdefault(path, [])
            if old not in history:
                history.append(old)

    changed = merge_sections(sections, previous, keep)
    old_media = {item["id"]: item for item in previous.get("media", [])}
    for item in media:
        original = old_media.get(item["id"], {})
        if (
            "source_alt_en" in original
            and item.get("alt_en") == original["source_alt_en"]
            and item.get("sha256") == original.get("sha256")
        ):
            item["source_alt_en"] = original["source_alt_en"]
            item["alt_en"] = original["alt_en"]
    changed |= merge_media(media, previous, keep)
    title_ru = old_ru.get("title", "") if title == old_en.get("title") else ""
    if not title_ru and old_ru.get("title"):
        keep("/locales/ru/title", old_en.get("title"), old_ru["title"])
        changed = True
    locales = {
        "en": {"title": title, "blocks": {}},
        "ru": {"title": title_ru, "blocks": {}},
    }
    for item in sections:
        for lang, suffix in (("en", ""), ("ru", "_ru")):
            locales[lang]["blocks"][item["id"]] = {
                field: item[field + suffix] for field in ("heading", "body")
            }
    for lang, old_locale in (("en", old_en), ("ru", old_ru)):
        if "aliases" in old_locale:
            locales[lang]["aliases"] = old_locale["aliases"]
    if "categories_ru" in previous.get("metadata", {}):
        metadata["categories_ru"] = previous["metadata"]["categories_ru"]
    if stale:
        source["stale_translations"] = stale
    if source.get("sha256") == old_source.get("sha256"):
        source["imported_at"] = old_source["imported_at"]
        for field in ("authors", "editorial_references"):
            if field in old_source:
                source.setdefault(field, old_source[field])
    return {
        "id": f"article:{slug}",
        "kind": "article",
        "source": source,
        "metadata": metadata,
        "sections": sections,
        "locales": locales,
        "media": media,
        "review": previous.get("review", DRAFT) if not changed else DRAFT,
    }


def pdf_body(text: str) -> str:
    paragraphs = []
    for block in re.split(r"\n\s*\n", text.strip()):
        # Keep lists and lines aligned as tables; unwrap the typographic line
        # breaks of prose without joining words separated by real hyphens.
        if re.search(r"(?m)^\s*(?:[•●▪]|\d+[.)])\s|\S {3,}\S", block):
            paragraphs.append(block.strip())
        elif block.strip():
            paragraphs.append(clean(block))
    return "\n\n".join(paragraphs)


def provenance(url: str, content: bytes, kind: str) -> dict:
    return {"type": kind, "url": url, "imported_at": now(), "sha256": digest(content)}


def write_record(
    destination: Path,
    slug: str,
    title: str,
    source: dict,
    metadata: dict,
    sections: list[dict],
    media: list[dict],
) -> None:
    path = destination / f"{slug}.json"
    previous = json.loads(path.read_bytes()) if path.exists() else {}
    write_json(path, record(slug, title, source, metadata, sections, media, previous))


def capture(
    path: Path, url: str, mirror: Path | None, mirror_name: str | None = None
) -> bytes:
    if mirror:
        content = (mirror / (mirror_name or path.name)).read_bytes()
        save_snapshot(path, content)
        return content
    return raw_bytes(path, url)


def discover_documents(
    page: BeautifulSoup, url: str, mirror: Path | None, snapshots: Path
) -> dict:
    found = {}
    for item in links(page, url):
        path = urlsplit(item["url"]).path
        document = DOCUMENT_PATH.fullmatch(path)
        if document:
            data = capture(
                snapshots / f"{document[1]}.html", item["url"], mirror, document[1]
            )
            details = soup(data)
            heading = details.find("h2")
            if not heading or not heading.parent:
                raise ValueError(f"Cannot find document heading: {item['url']}")
            title = direct_text(heading)
            pdf = next(
                link["url"]
                for link in links(details, item["url"])
                if PDF_PATH.fullmatch(urlsplit(link["url"]).path)
            )
            license_link = details.select_one("a[rel=license]")
            license_info = (
                {
                    "license": clean(license_link.get_text(" ", strip=True)),
                    "license_url": str(license_link["href"]),
                }
                if license_link
                else {}
            )
            authors = heading.parent.find("b", string=re.compile(r"^Authors?:$"))
            author_names = []
            if authors:
                for sibling in authors.next_siblings:
                    if isinstance(sibling, Tag) and sibling.name == "br":
                        break
                    author_names.append(
                        sibling.get_text(" ", strip=True)
                        if isinstance(sibling, Tag)
                        else str(sibling)
                    )
            credit = {"authors": clean("".join(author_names))} if author_names else {}
            found[pdf] = {
                "title": title,
                "record_url": item["url"],
                **license_info,
                **credit,
            }
        elif PDF_PATH.fullmatch(path):
            found.setdefault(item["url"], {"title": item["text"]})
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mirror", type=Path, nargs="?", help="use pre-downloaded source snapshots"
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--topics-only", action="store_true")
    scope.add_argument("--documents-only", action="store_true")
    parser.add_argument(
        "--media",
        action="store_true",
        help="mirror the original images linked by the topics",
    )
    args = parser.parse_args()
    data = data_directory()
    snapshots = data / "raw" / "reviews"
    destination = data / "processed" / "articles"
    catalog = capture(snapshots / "index.html", CATALOG, args.mirror)
    catalog_page = soup(catalog)
    topic_links = {
        match[1]: a
        for a in catalog_page.select("h3 a[href]")
        if (match := TOPIC_PATH.fullmatch(str(a["href"])))
    }
    topics = {
        topic: clean(link.get_text(" ", strip=True))
        for topic, link in topic_links.items()
    }
    if not topics:
        raise SystemExit("No review topics found in catalog")

    documents: dict[str, dict] = {}
    document_topics: dict[str, list[str]] = defaultdict(list)
    glossary = glossary_media(data) if args.media else {}
    for topic, title in topics.items():
        slug = f"oae-review-{topic}"
        url = CATALOG + topic
        content = capture(snapshots / f"{topic}.html", url, args.mirror)
        page = soup(content)
        docs = discover_documents(page, url, args.mirror, snapshots)
        for pdf_url, details in docs.items():
            filename = Path(urlsplit(pdf_url).path).name
            previous = documents.get(filename, {})
            documents[filename] = {**previous, **details, "pdf_url": pdf_url}
            document_topics[filename].append(f"article:{slug}")
        if args.documents_only:
            continue
        term_numbers = sorted(
            {
                int(match[1])
                for item in links(page, url)
                if (match := TERM_PATH.fullmatch(urlsplit(item["url"]).path))
            }
        )
        inline_images: dict[str, list[Tag]] = {}
        sections = html_sections(page, url, inline_images)
        if not sections or sections[0]["id"] != "about":
            raise ValueError(f"Could not extract About section for {url}")
        metadata = {
            "term_numbers": term_numbers,
            "document_ids": [
                f"article:oae-document-{Path(urlsplit(pdf).path).stem.lower()}"
                for pdf in docs
            ],
        }
        media = []
        if args.media:
            with ThreadPoolExecutor(max_workers=8) as pool:
                media = review_media(data, sections, pool, glossary)
                by_source = {
                    str(image["src"]): image
                    for images in inline_images.values()
                    for image in images
                }
                mirrors = dict(
                    zip(
                        by_source,
                        pool.map(
                            lambda image: inline_media(data, image, url),
                            by_source.values(),
                        ),
                    )
                )
            media.extend(mirrors.values())
            for item in sections:
                media_ids = (
                    mirrors[str(image["src"])]["id"]
                    for image in inline_images.get(item["id"], [])
                )
                item["media_ids"] = list(
                    dict.fromkeys([*item["media_ids"], *media_ids])
                )
            card = topic_links[topic].find_parent(class_="row")
            image = card.find("img") if card else None
            if image and image.get("src"):
                cover = cover_media(data, topic, image, title, catalog_page)
                media.insert(0, cover)
                sections[0]["media_ids"].insert(0, cover["id"])
            print(f"topic={topic} media={len(media)}", flush=True)
        elif (destination / f"{slug}.json").exists():
            saved = json.loads((destination / f"{slug}.json").read_bytes())
            media = saved.get("media", [])
            attached = {item["id"]: item["media_ids"] for item in saved["sections"]}
            for item in sections:
                item["media_ids"] = attached.get(item["id"], [])
        write_record(
            destination,
            slug,
            title,
            provenance(url, content, "oae_review_topic"),
            metadata,
            sections,
            media,
        )

    if not args.topics_only:
        for filename, details in documents.items():
            pdf_url = details["pdf_url"]
            pdf = snapshots / filename
            content = capture(pdf, pdf_url, args.mirror)
            extracted = current_snapshot(pdf).with_suffix(".pages.json")
            ready = args.mirror / f"{filename}.pages.json" if args.mirror else extracted
            if ready.exists():
                pages = json.loads(ready.read_bytes())
            else:
                pages = extract_pages(current_snapshot(pdf))
            if not extracted.exists():
                write_json(extracted, pages)
            slug = f"oae-document-{Path(filename).stem.lower()}"
            topic_ids = document_topics[filename]
            title = details.pop("title")
            if title.lower().startswith("mini-review ("):
                topic = topic_ids[0].removeprefix("article:oae-review-")
                title = f"{topics[topic]}: Mini-Review"
            source = {
                **provenance(
                    details.get("record_url", pdf_url), content, "oae_review_document"
                ),
                **details,
                "pdf_path": current_snapshot(pdf).relative_to(data).as_posix(),
            }
            metadata = {
                "topic_ids": topic_ids,
                "page_count": len(pages),
                "ocr_pages": [
                    n for n, p in enumerate(pages, 1) if p["method"] == "ocr"
                ],
            }
            refs = page_links(current_snapshot(pdf))
            sections = [
                section(f"page-{n}", "article_body", f"Page {n}", body, refs.get(n, []))
                for n, p in enumerate(pages, 1)
                if (body := pdf_body(p["text"]))
            ]
            if not sections:
                raise ValueError(f"No extractable text in {filename}")
            prior_path = destination / f"{slug}.json"
            prior = json.loads(prior_path.read_bytes()) if prior_path.exists() else {}
            media = []
            if prior.get("source", {}).get("sha256") == source["sha256"]:
                for key in ("authors", "license", "license_url"):
                    if key not in source and key in prior["source"]:
                        source[key] = prior["source"][key]
                media = prior.get("media", [])
                previous_sections = {item["id"]: item for item in prior["sections"]}
                for item in sections:
                    previous = previous_sections.get(item["id"], {})
                    item["media_ids"] = previous.get("media_ids", [])
                    links_by_url = {
                        link["url"]: link for link in previous.get("links", [])
                    }
                    for link in item["links"]:
                        links_by_url.setdefault(link["url"], link)
                    item["links"] = list(links_by_url.values())
                    if item["body"] == previous.get("extracted_body"):
                        item["extracted_body"] = item["body"]
                        item["body"] = previous["body"]
                text_ids = {item["id"] for item in sections}
                sections.extend(
                    item
                    for item in prior["sections"]
                    if item["id"] not in text_ids and item["type"] == "media_collection"
                )
                sections.sort(key=lambda item: int(item["id"].removeprefix("page-")))
            write_record(destination, slug, title, source, metadata, sections, media)
    print(f"topics={len(topics)} documents={len(documents)}")


if __name__ == "__main__":
    main()
