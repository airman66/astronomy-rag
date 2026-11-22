"""Cache headers and conditional responses for published media."""

import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from starlette.types import Receive, Scope, Send

from app.api.errors import error
from app.infra.config import RELEASE_DIR


def _file_response(
    request: Request, path: Path, digest: str | None, version: str | None, error_code: str,
) -> Response:
    if version is not None and version != digest:
        return error(404, error_code, "Версия файла не найдена")
    cache = "public, max-age=31536000, immutable" if version else "public, max-age=0, must-revalidate"
    headers = {"Cache-Control": cache}
    if digest:
        etag = f'"{digest}"'
        headers["ETag"] = etag
        validators = request.headers.get("If-None-Match", "").split(",")
        if any(value.strip().removeprefix("W/") in (etag, "*") for value in validators):
            return Response(status_code=304, headers=headers)
    return FileResponse(path, headers=headers)


def _snapshot(path: Path, destination: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source, destination.open("wb") as copy:
        stat = os.fstat(source.fileno())
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            copy.write(block)
    os.utime(destination, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    return digest.hexdigest()


class _MutableMediaResponse(Response):
    def __init__(self, request: Request, path: Path, version: str | None, error_code: str) -> None:
        super().__init__()
        self.request = request
        self.path = path
        self.version = version
        self.error_code = error_code

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        with TemporaryDirectory(prefix="astronomy-media-") as directory:
            snapshot = Path(directory) / self.path.name
            try:
                digest = await run_in_threadpool(_snapshot, self.path, snapshot)
            except FileNotFoundError:
                response = error(404, self.error_code, "Файл не найден")
            else:
                response = _file_response(self.request, snapshot, digest, self.version, self.error_code)
            # Keep the snapshot alive until FileResponse has sent the body itself.
            extensions = {name: value for name, value in scope.get("extensions", {}).items()
                          if name != "http.response.pathsend"}
            await response({**scope, "extensions": extensions}, receive, send)


def media_response(
    request: Request, path: Path, digest: str | None, version: str | None, *, error_code: str,
) -> Response:
    if RELEASE_DIR is not None:
        return _file_response(request, path, digest, version, error_code)
    return _MutableMediaResponse(request, path, version, error_code)
