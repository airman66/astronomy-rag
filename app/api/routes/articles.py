from typing import Literal

from fastapi import APIRouter, Query

from app import storage
from app.api.errors import error
from app.api.responses import LIMITS
from app.api.schemas.catalog import ListResponse, RecordDetail
from app.api.schemas.common import ErrorResponse

router = APIRouter(prefix="/articles")


@router.get("", response_model=ListResponse, responses=LIMITS)
def articles(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    q: str = Query("", max_length=200),
    status: Literal["", "draft", "reviewed"] = "",
    locale: Literal["ru", "en"] = "ru",
) -> ListResponse:
    items, total = storage.list_records(
        "article", page=page, page_size=page_size, locale=locale, query=q, status=status
    )
    return ListResponse(items=items, total=total, page=page, page_size=page_size)


@router.get("/{article_id}", response_model=RecordDetail, responses={404: {"model": ErrorResponse}, **LIMITS})
def article_detail(article_id: str):
    record = storage.get_article(article_id)
    return (
        RecordDetail.model_validate(record)
        if record
        else error(404, "article_not_found", "Статья не найдена")
    )
