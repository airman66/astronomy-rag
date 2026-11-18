from typing import Literal

from pydantic import BaseModel, Field


class SearchItem(BaseModel):
    record_id: str
    section_id: str
    kind: str
    title: str
    excerpt: str
    url: str
    review_status: str
    score: float


class SearchResponse(BaseModel):
    items: list[SearchItem]
    total: int
    page: int
    page_size: int
    mode: Literal["hybrid", "lexical"] = Field(
        "hybrid",
        description="lexical: семантический индекс или сервер эмбеддингов недоступен, выдача только по FTS",
    )
