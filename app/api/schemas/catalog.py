from typing import Any, Literal

from pydantic import BaseModel, Field


class ListItem(BaseModel):
    id: str
    title: str
    title_en: str | None = None
    url: str
    review_status: Literal["draft", "reviewed"]
    categories: list[str]
    categories_ru: list[str] = Field(default_factory=list)
    illustration_url: str | None = None


class ListResponse(BaseModel):
    items: list[ListItem]
    total: int
    page: int
    page_size: int


class FacetLetter(BaseModel):
    letter: str
    count: int


class FacetCategory(BaseModel):
    id: str
    title: str
    count: int


class FacetsResponse(BaseModel):
    locale: Literal["ru", "en"]
    total: int
    letters: list[FacetLetter]
    categories: list[FacetCategory]


class ReviewInfo(BaseModel):
    status: Literal["draft", "reviewed"]
    reviewer: str | None = None
    notes: str | None = None


class Section(BaseModel):
    id: str
    type: str
    heading: str
    body: str
    heading_ru: str | None = None
    body_ru: str | None = None
    links: list[dict[str, str]] = Field(default_factory=list)
    media_ids: list[str] = Field(default_factory=list)
    source_html: str | None = None


class LocaleContent(BaseModel):
    title: str
    aliases: list[str] = Field(default_factory=list)
    blocks: dict[str, dict[str, str]]


class MediaDetail(BaseModel):
    id: str
    kind: str
    file_type: str | None = None
    source_url: str | None = None
    record_url: str | None = None
    original_path: str | None = None
    ru_path: str | None = None
    ru_sha256: str | None = None
    sha256: str | None = None
    size: int | None = None
    title_en: str = ""
    title_ru: str = ""
    short_caption_en: str = ""
    short_caption_ru: str = ""
    caption_en: str = ""
    caption_ru: str = ""
    alt_en: str = ""
    alt_ru: str = ""
    credit: str | None = None
    credit_url: str | None = None
    license: str | None = None
    license_url: str | None = None
    term_numbers: list[int] = Field(default_factory=list)
    source_language: str = "en"
    review: ReviewInfo


class IllustrationInfo(BaseModel):
    """Cover image for a term, stored apart from Astro4Edu media."""

    term_number: int
    url: str
    width: int | None = None
    height: int | None = None
    alt_ru: str = ""
    alt_en: str = ""
    caption_ru: str = ""
    kind: Literal["conceptual", "diagram"] = "conceptual"
    status: Literal["generated", "approved"]
    generator: str = ""
    source_type: str = "generated"
    file_page: str | None = None
    article_url: str | None = None
    author: str | None = None
    license: str | None = None
    license_url: str | None = None
    modified: str | None = None


class RecordDetail(BaseModel):
    id: str
    kind: str
    source: dict[str, Any]
    metadata: dict[str, Any]
    sections: list[Section]
    locales: dict[str, LocaleContent]
    media: list[MediaDetail] = Field(default_factory=list)
    illustration: IllustrationInfo | None = None
    review: ReviewInfo
