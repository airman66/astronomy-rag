from typing import Literal

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from app import storage
from app.api.errors import error
from app.api.responses import IMAGE_TYPES, LIMITS
from app.api.schemas.catalog import FacetsResponse, ListResponse, RecordDetail
from app.api.schemas.common import ErrorResponse
from app.media_files import local_file

router = APIRouter(prefix="/terms")


@router.get("", response_model=ListResponse, responses=LIMITS)
def terms(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    q: str = Query("", max_length=200),
    category: str = "",
    letter: str = Query("", max_length=1),
    status: Literal["", "draft", "reviewed"] = "",
    locale: Literal["ru", "en"] = "ru",
) -> ListResponse:
    """Список терминов. Поиск по названию и алиасам и фильтр по букве не зависят от регистра и «ё»; сортировка — по названию на языке `locale`."""
    items, total = storage.list_records(
        "term",
        page=page,
        page_size=page_size,
        locale=locale,
        query=q,
        category=category,
        letter=letter,
        status=status,
    )
    return ListResponse(items=items, total=total, page=page, page_size=page_size)


@router.get("/facets", response_model=FacetsResponse, responses=LIMITS)
def term_facets(locale: Literal["ru", "en"] = "ru") -> FacetsResponse:
    """Первые буквы названий и категории с числом терминов — для алфавитного указателя и фильтров."""
    letters = storage.letters(locale)
    return FacetsResponse(
        locale=locale,
        total=sum(item["count"] for item in letters),
        letters=letters,
        categories=storage.categories(locale),
    )


@router.get(
    "/{term_number}", response_model=RecordDetail, responses={404: {"model": ErrorResponse}, **LIMITS}
)
def term_detail(term_number: int):
    record = storage.get_term(term_number)
    return RecordDetail.model_validate(record) if record else error(404, "term_not_found", "Термин не найден")


@router.get(
    "/{term_number}/illustration",
    response_class=FileResponse,
    responses={
        200: {"content": IMAGE_TYPES, "description": "Иллюстрация термина"},
        404: {"model": ErrorResponse},
        **LIMITS,
    },
)
def term_illustration(term_number: int):
    """Файл иллюстрации команды; 404 `illustration_not_found`, пока она не сгенерирована и не проверена."""
    item = storage.get_illustration(term_number)
    if not item:
        return error(404, "illustration_not_found", "Иллюстрация ещё не готова")
    path = local_file(storage.DATA, item["path"])
    if path is None:
        return error(404, "illustration_not_found", "Файл иллюстрации не найден")
    return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})
