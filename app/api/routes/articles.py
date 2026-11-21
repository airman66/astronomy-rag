from typing import Literal

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from app.api.errors import error
from app.api.responses import LIMITS
from app.api.schemas.catalog import ListResponse, RecordDetail
from app.api.schemas.common import ErrorResponse
from app.catalog import storage
from app.catalog.media import local_file

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
    return ListResponse.model_validate({"items": items, "total": total, "page": page, "page_size": page_size})


@router.get("/{article_id}", response_model=RecordDetail, responses={404: {"model": ErrorResponse}, **LIMITS})
def article_detail(article_id: str):
    record = storage.get_article(article_id)
    return (
        RecordDetail.model_validate(record)
        if record
        else error(404, "article_not_found", "Статья не найдена")
    )


@router.get(
    "/{article_id}/source.pdf",
    response_class=FileResponse,
    responses={200: {"content": {"application/pdf": {}}}, 404: {"model": ErrorResponse}, **LIMITS},
)
def article_source_pdf(article_id: str):
    record = storage.get_article(article_id)
    if not record:
        return error(404, "article_not_found", "Статья не найдена")
    source = record["source"]
    filename = source.get("pdf_path") if source.get("type") == "oae_review_document" else None
    path = local_file(storage.DATA, filename)
    archive = (storage.DATA / "raw" / "reviews").resolve()
    if path is None or not path.is_relative_to(archive) or path.suffix != ".pdf":
        return error(404, "article_file_not_found", "Оригинальный PDF не найден")
    return FileResponse(path, media_type="application/pdf", filename=path.name)


@router.get(
    "/{article_id}/translation.pdf",
    response_class=FileResponse,
    responses={200: {"content": {"application/pdf": {}}}, 404: {"model": ErrorResponse}, **LIMITS},
)
def article_translation_pdf(article_id: str):
    record = storage.get_article(article_id)
    if not record:
        return error(404, "article_not_found", "Статья не найдена")
    archive = storage.DATA / "media" / "ru" / "review-pdf"
    if record["source"].get("type") == "oae_review_document":
        filename = record["id"].removeprefix("article:") + ".pdf"
        path = local_file(storage.DATA, str(archive.resolve() / filename))
        if path and path.is_relative_to(archive.resolve()):
            return FileResponse(path, media_type="application/pdf", filename=filename)
    return error(404, "article_file_not_found", "Русский PDF не найден")
