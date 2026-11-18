from typing import Annotated, Literal

from fastapi import APIRouter, Path, Query
from fastapi.responses import FileResponse

from app.api.errors import error
from app.api.responses import IMAGE_TYPES, LIMITS
from app.api.schemas.catalog import FacetsResponse, ListResponse, RecordDetail
from app.api.schemas.common import ErrorResponse
from app.catalog import storage
from app.catalog.media import local_file

router = APIRouter(prefix="/terms")
TermNumber = Annotated[int, Path(ge=-(2**63), le=2**63 - 1)]


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
    return ListResponse.model_validate({"items": items, "total": total, "page": page, "page_size": page_size})


@router.get("/facets", response_model=FacetsResponse, responses=LIMITS)
def term_facets(locale: Literal["ru", "en"] = "ru") -> FacetsResponse:
    """Первые буквы названий и категории с числом терминов — для алфавитного указателя и фильтров."""
    letters = storage.letters(locale)
    return FacetsResponse.model_validate(
        {
            "locale": locale,
            "total": sum(item["count"] for item in letters),
            "letters": letters,
            "categories": storage.categories(locale),
        }
    )


@router.get(
    "/{term_number}", response_model=RecordDetail, responses={404: {"model": ErrorResponse}, **LIMITS}
)
def term_detail(term_number: TermNumber):
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
def term_illustration(term_number: TermNumber):
    """Файл обложки термина. Если обложка недоступна, возвращает 404 `illustration_not_found`."""
    item = storage.get_illustration(term_number)
    if not item:
        return error(404, "illustration_not_found", "Иллюстрация ещё не готова")
    path = local_file(storage.DATA, item["path"])
    if path is None:
        return error(404, "illustration_not_found", "Файл иллюстрации не найден")
    return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})
