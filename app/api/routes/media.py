from typing import Literal

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse

from app.api.errors import error
from app.api.media_files import media_response
from app.api.responses import IMAGE_TYPES, LIMITS
from app.api.schemas.catalog import MediaDetail
from app.api.schemas.common import ErrorResponse
from app.catalog import storage
from app.catalog.media import local_file, localized_digest, localized_filename

router = APIRouter(prefix="/media")


@router.get("/{media_id}", response_model=MediaDetail, responses={404: {"model": ErrorResponse}, **LIMITS})
def media_detail(media_id: str):
    item = storage.get_media(media_id)
    return MediaDetail.model_validate(item) if item else error(404, "media_not_found", "Медиа не найдено")


@router.get(
    "/{media_id}/file",
    response_class=FileResponse,
    responses={
        200: {"content": IMAGE_TYPES, "description": "Файл изображения на выбранном языке"},
        304: {"description": "Версия файла совпадает с кэшем"},
        404: {"model": ErrorResponse},
        **LIMITS,
    },
)
def media_file(request: Request, media_id: str, locale: Literal["ru", "en"] = "ru",
               v: str | None = Query(None, max_length=64)):
    item = storage.get_media(media_id)
    if not item:
        return error(404, "media_not_found", "Медиа не найдено")
    path = local_file(storage.DATA, localized_filename(item, locale))
    if path is None:
        return error(404, "media_file_not_found", "Файл медиа не найден")
    digest = localized_digest(item, locale)
    if v is not None and v != digest:
        return error(404, "media_file_not_found", "Версия медиа не найдена")
    return media_response(request, path, digest, v)
