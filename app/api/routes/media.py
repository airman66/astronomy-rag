from typing import Literal

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from app.api.errors import error
from app.api.media_files import media_not_found, media_response
from app.api.responses import IMAGE_TYPES, LIMITS
from app.api.schemas.catalog import MediaDetail
from app.api.schemas.common import ErrorResponse
from app.catalog import storage
from app.catalog.media import local_file, localized_digest, localized_filename

router = APIRouter(prefix="/media")


@router.get("/{media_id}", response_model=MediaDetail, responses={404: {"model": ErrorResponse}, **LIMITS})
def media_detail(media_id: str) -> MediaDetail | JSONResponse:
    item = storage.get_media(media_id)
    if item is None:
        return error(404, "media_not_found", "Медиа не найдено")
    return MediaDetail.model_validate(item)


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
def media_file(
    request: Request,
    media_id: str,
    locale: Literal["ru", "en"] = "ru",
    v: str | None = Query(None, max_length=64),
) -> Response:
    item = storage.get_media(media_id)
    if not item:
        return media_not_found("media_not_found", "Медиа не найдено", v)
    path = local_file(storage.DATA, localized_filename(item, locale))
    if path is None:
        return media_not_found("media_file_not_found", "Файл медиа не найден", v)
    digest = localized_digest(item, locale)
    if v is not None and v != digest:
        return media_not_found("media_file_not_found", "Версия медиа не найдена", v)
    return media_response(request, path, digest, v, error_code="media_file_not_found")
