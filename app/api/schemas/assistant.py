from typing import Any, Literal

from pydantic import BaseModel, Field


class AnswerRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    mode: Literal["auto", "term", "article", "explain", "general"] = "auto"
    locale: Literal["ru", "en"] = "ru"
    context: dict[str, Any] | None = None


class AnswerSource(BaseModel):
    record_id: str
    section_id: str
    url: str
    review_status: str
    chunk_id: str | None = None


class AnswerImage(BaseModel):
    record_id: str
    url: str
    alt: str
    caption: str
    kind: Literal["illustration", "source_media"]
    credit: str | None = None
    credit_url: str | None = None
    license: str | None = None
    license_url: str | None = None
    modified: str | None = None


class AnswerBlock(BaseModel):
    text: str
    source_indices: list[int]


class AnswerResponse(BaseModel):
    mode: Literal["term", "article", "explain", "general", "out_of_scope"]
    grounding: Literal["knowledge_base", "model", "mixed", "policy"]
    answer: str
    primary_url: str | None
    sources: list[AnswerSource]
    images: list[AnswerImage] = Field(default_factory=list)
    review_status: str | None
    diagnostics: dict[str, str]
    coverage: Literal["complete", "partial", "insufficient"] = "complete"
    blocks: list[AnswerBlock] = Field(default_factory=list)
