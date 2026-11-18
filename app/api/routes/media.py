from typing import Literal

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.api.errors import error
from app.api.responses import IMAGE_TYPES, LIMITS
from app.api.schemas.catalog import MediaDetail
from app.api.schemas.common import ErrorResponse
from app.catalog import storage
from app.catalog.media import local_file, localized_filename

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
        404: {"model": ErrorResponse},
        **LIMITS,
    },
)
def media_file(media_id: str, locale: Literal["ru", "en"] = "ru"):
    item = storage.get_media(media_id)
    if not item:
        return error(404, "media_not_found", "Медиа не найдено")
    path = local_file(storage.DATA, localized_filename(item, locale))
    if path is None:
        return error(404, "media_file_not_found", "Файл медиа не найден")
    return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})
