"""Small on-disk catalogues for HTTP and retrieval tests."""

import json
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from app.catalog import storage, vector


def temporary_dir(case: TestCase) -> Path:
    return Path(case.enterContext(tempfile.TemporaryDirectory()))


def use_vector_root(case: TestCase, root: Path) -> None:
    case.enterContext(
        patch.multiple(
            vector,
            DATA=root,
            _loaded=None,
            _failure={},
            _stamp={"key": "", "checked": 0.0},
        )
    )


def use_catalog(case: TestCase) -> Path:
    root = temporary_dir(case)
    case.enterContext(patch.multiple(storage, DATA=root, DB=root / "glossary.db", _checked={}))
    use_vector_root(case, root)
    return root


def make_term(
    number: int, title_en: str, title_ru: str, body_en: str, body_ru: str, categories=("Stars",)
) -> dict:
    return {
        "id": f"term:{number}",
        "kind": "term_page",
        "source": {
            "type": "astro4edu_glossary",
            "term_number": number,
            "url": f"https://astro4edu.org/resources/glossary/term/{number}/",
        },
        "metadata": {
            "categories": list(categories),
            "categories_ru": ["Звёзды"] * len(categories),
            "related_term_numbers": [],
        },
        "sections": [
            {
                "id": "definition",
                "type": "definition",
                "heading": "",
                "body": body_en,
                "media_ids": [],
                "links": [],
            }
        ],
        "locales": {
            "en": {
                "title": title_en,
                "aliases": [],
                "blocks": {"definition": {"heading": "", "body": body_en}},
            },
            "ru": {
                "title": title_ru,
                "aliases": [],
                "blocks": {"definition": {"heading": "", "body": body_ru}},
            },
        },
        "media": [],
        "review": {"status": "draft", "reviewer": None},
    }


def write_term(root: Path, term: dict) -> None:
    folder = root / "processed/terms"
    folder.mkdir(parents=True, exist_ok=True)
    number = term["source"]["term_number"]
    (folder / f"{number}.json").write_text(json.dumps(term, ensure_ascii=False), encoding="utf-8")


def seed_star(root: Path) -> Path:
    path = root / "media/original/example.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not-a-real-png")
    term = make_term(1, "A-type Star", "Звезда класса A", "A hot star.", "Горячая звезда.")
    term["sections"][0]["media_ids"] = ["media:example"]
    term["media"].append(
        {
            "id": "media:example",
            "kind": "media",
            "file_type": "img",
            "original_path": str(path),
            "title_en": "Example",
            "title_ru": "Пример",
            "caption_en": "A star.",
            "caption_ru": "Звезда.",
            "alt_en": "Star",
            "alt_ru": "Звезда",
            "review": {"status": "draft", "reviewer": None},
        }
    )
    write_term(root, term)
    storage.rebuild_database()
    return path


def seed_search_terms(root: Path) -> None:
    for term in (
        make_term(
            1,
            "Black Hole",
            "Чёрная дыра",
            "A region of space from which nothing escapes.",
            "Область пространства, из которой ничто не может вырваться.",
        ),
        make_term(
            2,
            "Stars",
            "Звёзды",
            "Hot balls of gas.",
            "Горячие газовые шары, светящиеся за счёт термоядерных реакций.",
        ),
        make_term(
            3,
            "Comet",
            "Комета",
            "An icy body.",
            "Ледяное тело, у которого возле Солнца появляется хвост.",
            categories=("Solar System",),
        ),
    ):
        write_term(root, term)
    storage.rebuild_database()
