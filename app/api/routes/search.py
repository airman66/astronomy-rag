from typing import Literal

from fastapi import APIRouter, Query

from app.api.responses import LIMITS
from app.api.schemas.search import SearchResponse
from app.catalog.search import paged_search

router = APIRouter()
SEARCH_POOL = 200


@router.get("/search", response_model=SearchResponse, responses=LIMITS)
def search(
    q: str = Query(min_length=1, max_length=200),
    locale: Literal["ru", "en"] = "ru",
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> SearchResponse:
    """Гибридный поиск (FTS5 со стеммингом + векторный индекс). `total` — размер пула релевантных результатов (не больше 200)."""
    matches, mode = paged_search(q, locale, limit=SEARCH_POOL)
    return SearchResponse.model_validate(
        {
            "items": matches[(page - 1) * page_size : page * page_size],
            "total": len(matches),
            "page": page,
            "page_size": page_size,
            "mode": mode,
        }
    )
