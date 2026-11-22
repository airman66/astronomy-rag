"""Cache headers and conditional responses for published media."""

from pathlib import Path

from fastapi import Request, Response
from fastapi.responses import FileResponse


def media_response(request: Request, path: Path, digest: str | None, version: str | None) -> Response:
    cache = "public, max-age=31536000, immutable" if version else "public, max-age=0, must-revalidate"
    headers = {"Cache-Control": cache}
    if digest:
        etag = f'"{digest}"'
        headers["ETag"] = etag
        validators = request.headers.get("If-None-Match", "").split(",")
        if any(value.strip().removeprefix("W/") in (etag, "*") for value in validators):
            return Response(status_code=304, headers=headers)
    return FileResponse(path, headers=headers)
