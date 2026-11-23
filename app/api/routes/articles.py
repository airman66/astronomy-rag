from typing import Literal

from fastapi import APIRouter, Query, Response
from fastapi.responses import FileResponse, JSONResponse

from app.api.errors import error
from app.api.responses import LIMITS
from app.api.schemas.catalog import ListResponse, RecordDetail
from app.api.schemas.common import ErrorResponse
from app.catalog import storage
from app.catalog.media import review_pdf

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
def article_detail(article_id: str) -> RecordDetail | JSONResponse:
    record = storage.get_article(article_id)
    if record is None:
        return error(404, "article_not_found", "Статья не найдена")
    return RecordDetail.model_validate(record)


@router.get(
    "/{article_id}/source.pdf",
    response_class=FileResponse,
    responses={200: {"content": {"application/pdf": {}}}, 404: {"model": ErrorResponse}, **LIMITS},
)
def article_source_pdf(article_id: str) -> Response:
    record = storage.get_article(article_id)
    if not record:
        return error(404, "article_not_found", "Статья не найдена")
    path = review_pdf(storage.DATA, record, "en")
    if path is None:
        return error(404, "article_file_not_found", "Оригинальный PDF не найден")
    return FileResponse(path, media_type="application/pdf", filename=path.name)


@router.get(
    "/{article_id}/translation.pdf",
    response_class=FileResponse,
    responses={200: {"content": {"application/pdf": {}}}, 404: {"model": ErrorResponse}, **LIMITS},
)
def article_translation_pdf(article_id: str) -> Response:
    record = storage.get_article(article_id)
    if not record:
        return error(404, "article_not_found", "Статья не найдена")
    path = review_pdf(storage.DATA, record, "ru")
    if path:
        filename = record["id"].removeprefix("article:") + ".pdf"
        return FileResponse(path, media_type="application/pdf", filename=filename)
    return error(404, "article_file_not_found", "Русский PDF не найден")
