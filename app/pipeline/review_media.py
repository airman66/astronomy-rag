"""Reuse glossary media or mirror the originals linked from an OAE review."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup, Tag

from app.catalog.media import IMAGE_SUFFIXES, local_file
from app.pipeline.importer import digest, raw_bytes
from app.pipeline.records import BASE, normalize_media

RESOURCE = re.compile(r"^/resources/(media|diagram)/([A-Za-z0-9]+)/?$")
INLINE_IMAGE = 'img[src^="/media/astroedu/images/"], img[src^="/media/headshots/"]'


def label_text(root: Tag, name: str) -> str:
    label = root.find(
        "b", string=re.compile(rf"^(?:Image|Diagram|Video) {name}:$", re.I)
    )
    if not label:
        return ""
    parts = []
    for node in label.next_siblings:
        if isinstance(node, Tag) and (
            node.name in ("b", "p") or str(node.get("href", "")).startswith("#")
        ):
            break
        if (
            not isinstance(node, Tag)
            or node.name != "a"
            or node.get_text(strip=True) != "Credit Link"
        ):
            parts.append(
                node.get_text(" ", strip=True) if isinstance(node, Tag) else str(node)
            )
    return " ".join(" ".join(parts).split())


def media_record(data: Path, path: str, glossary: dict[str, dict]) -> dict:
    if path in glossary:
        item = dict(glossary[path])
    else:
        match = RESOURCE.fullmatch(path)
        if not match:
            raise ValueError(f"Invalid OAE media URL path: {path}")
        kind, slug = match.groups()
        url = BASE + path + "/"
        raw = raw_bytes(data / "raw" / "reviews" / f"{kind}-{slug}.html", url)
        page = BeautifulSoup(raw, "html.parser")
        download = next(
            a
            for a in page.select("a[href]")
            if a.get_text(" ", strip=True) == "Download File"
        )
        source = urljoin(url, str(download["href"]))
        heading = page.find("h2")
        if not heading:
            raise ValueError(f"Cannot find media title: {url}")
        title = heading.get_text(" ", strip=True)
        license_link = page.select_one("a[rel=license]")
        license_name = (
            license_link.get_text(" ", strip=True)
            if license_link
            else label_text(page, "license")
        )
        caption = label_text(page, "caption")
        credit = label_text(page, "credit")
        image = page.select_one("a[href] img[alt][src*='/media/']")
        alt = str(image.get("alt", "")) if image else ""
        if kind == "diagram":
            item = normalize_media(
                {
                    "diagram_slug": slug,
                    "diagram_url": url,
                    "multimedia_file": source,
                    "diagram_title": title,
                    "caption": caption,
                    "alt_text": alt,
                    "credit_text": credit,
                    "license": license_name,
                },
                "diagram",
            )
        else:
            file_type = (
                "img"
                if Path(urlsplit(source).path).suffix.lower() in IMAGE_SUFFIXES
                else "vid"
            )
            item = normalize_media(
                {
                    "record_url": url,
                    "file_url": source,
                    "file_type": file_type,
                    "captions": [
                        {
                            "language_code": "en",
                            "file_title": title,
                            "caption": caption,
                            "alt_text": alt,
                            "credit": credit,
                        }
                    ],
                    "license": license_name,
                },
                "media",
            )
        if license_link:
            item["license_url"] = str(license_link["href"])
    if not item["source_url"]:
        return item
    suffix = Path(urlsplit(item["source_url"]).path).suffix.lower()
    if suffix not in IMAGE_SUFFIXES or local_file(data, item.get("original_path")):
        return item
    content = raw_bytes(
        data / "media" / "original" / f"{item['id'].replace(':', '_')}{suffix}",
        item["source_url"],
    )
    checksum = digest(content)
    if item.get("sha256") and item["sha256"] != checksum:
        raise ValueError(f"Original checksum differs for {item['id']}")
    item["original_path"] = f"media/original/{item['id'].replace(':', '_')}{suffix}"
    item["sha256"] = checksum
    item["size"] = len(content)
    return item


def glossary_media(data: Path) -> dict[str, dict]:
    items = {}
    for record in (data / "processed" / "terms").glob("*.json"):
        for item in json.loads(record.read_bytes())["media"]:
            if item.get("record_url"):
                items.setdefault(urlsplit(item["record_url"]).path.rstrip("/"), item)
    return items


def review_media(
    data: Path, sections: list[dict], pool: ThreadPoolExecutor, glossary: dict
) -> list[dict]:
    paths: list[str] = []
    seen_paths: set[str] = set()
    for section in sections:
        for link in section["links"]:
            path = urlsplit(link["url"]).path.rstrip("/")
            if RESOURCE.fullmatch(path) and path not in seen_paths:
                seen_paths.add(path)
                paths.append(path)
    if not paths:
        return []

    media = dict(
        zip(paths, pool.map(lambda path: media_record(data, path, glossary), paths))
    )
    for section in sections:
        media_ids: list[str] = []
        seen_ids: set[str] = set()
        for link in section["links"]:
            path = urlsplit(link["url"]).path.rstrip("/")
            if path in media:
                identifier = media[path]["id"]
                if identifier not in seen_ids:
                    seen_ids.add(identifier)
                    media_ids.append(identifier)
        section["media_ids"] = media_ids
    return list(media.values())


def cover_media(data: Path, topic: str, image: Tag, title: str, catalog: Tag) -> dict:
    source_url = urljoin(BASE, str(image["src"]))
    filename = f"oae-review-{topic}{Path(urlsplit(source_url).path).suffix}"
    relative = Path("media/original") / filename
    content = raw_bytes(data / relative, source_url)
    label = catalog.find("b", string=re.compile(rf"^{re.escape(title)}:$", re.I))
    credit = " ".join(str(label.next_sibling).strip(" ;\n").split()) if label else None
    return {
        "id": f"media:oae-review-{topic}",
        "kind": "media",
        "file_type": "img",
        "source_url": source_url,
        "record_url": f"{BASE}/oae-reviews/{topic}",
        "original_path": relative.as_posix(),
        "sha256": digest(content),
        "size": len(content),
        "alt_en": str(image.get("alt", "")),
        "alt_ru": "",
        "credit": credit,
        "title_en": "",
        "title_ru": "",
        "caption_en": "",
        "caption_ru": "",
        "short_caption_en": "",
        "short_caption_ru": "",
        "review": {"status": "draft", "reviewer": None},
    }


def inline_media(data: Path, image: Tag, page_url: str) -> dict:
    source_url = urljoin(page_url, str(image["src"]))
    suffix = Path(urlsplit(source_url).path).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported OAE image: {source_url}")
    slug = f"oae-inline-{digest(source_url.encode())[:20]}"
    relative = Path("media/original/review-topics") / f"{slug}{suffix}"
    content = raw_bytes(data / relative, source_url)
    link = image.find_parent("a", href=True)
    return {
        "id": f"media:{slug}",
        "kind": "media",
        "file_type": "img",
        "source_url": source_url,
        "record_url": urljoin(page_url, str(link["href"])) if link else page_url,
        "original_path": relative.as_posix(),
        "sha256": digest(content),
        "size": len(content),
        "alt_en": str(image.get("alt", "")),
        "alt_ru": "",
        "title_en": "",
        "title_ru": "",
        "caption_en": "",
        "caption_ru": "",
        "short_caption_en": "",
        "short_caption_ru": "",
        "review": {"status": "draft", "reviewer": None},
    }
