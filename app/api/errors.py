from typing import Any, Mapping

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.api.schemas.common import ErrorResponse


def error(
    status: int,
    code: str,
    message: str,
    details: Any = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    body = ErrorResponse(code=code, message=message, details=details)
    return JSONResponse(status_code=status, content=body.model_dump(), headers=headers)


async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    return error(422, "validation_error", "Некорректный запрос", exc.errors())


async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
    code = "not_found" if exc.status_code == 404 else "http_error"
    return error(exc.status_code, code, str(exc.detail), headers=exc.headers)
