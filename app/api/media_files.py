"""Cache headers and conditional responses for published media."""

import hashlib
import os
import uuid
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import NamedTuple

from fastapi import Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from starlette.types import Receive, Scope, Send

from app.api.errors import error
from app.infra.config import RELEASE_DIR

MAX_CACHED_BYTES = 256 * 1024 * 1024
MAX_CACHED_FILES = 128


def _generation(stat: os.stat_result) -> tuple[int, ...]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class _CachedSnapshot(NamedTuple):
    generation: tuple[int, ...]
    path: Path
    digest: str
    size: int


class _SnapshotCache:
    def __init__(self) -> None:
        self._directory: TemporaryDirectory | None = None
        self._entries: OrderedDict[Path, _CachedSnapshot] = OrderedDict()
        self._bytes = 0
        self._lock = Lock()
        self._copy_locks = tuple(Lock() for _ in range(8))

    def link(self, path: Path, destination: Path) -> str:
        # Serialize copies of the same path. Lookups for other files can proceed during a copy.
        with self._copy_locks[hash(path) % len(self._copy_locks)]:
            stat = path.stat()
            generation = _generation(stat)
            with self._lock:
                cached = self._entries.get(path)
                if cached and cached.generation == generation:
                    # Keep a request-owned link so eviction cannot interrupt streaming.
                    os.link(cached.path, destination)
                    self._entries.move_to_end(path)
                    return cached.digest
                if cached:
                    self._discard(path)

            digest = _snapshot(path, destination)
            try:
                unchanged = generation == _generation(path.stat())
            except FileNotFoundError:
                unchanged = False
            if unchanged and stat.st_size <= MAX_CACHED_BYTES:
                self._store(path, destination, generation, digest, stat.st_size)
            return digest

    def _store(self, path: Path, snapshot: Path, generation: tuple[int, ...], digest: str, size: int) -> None:
        with self._lock:
            while self._entries and (
                self._bytes + size > MAX_CACHED_BYTES or len(self._entries) >= MAX_CACHED_FILES
            ):
                self._discard(next(iter(self._entries)))
            if self._directory is None:
                self._directory = TemporaryDirectory(prefix="astronomy-media-cache-")
            cached_path = Path(self._directory.name) / uuid.uuid4().hex
            os.link(snapshot, cached_path)
            self._entries[path] = _CachedSnapshot(generation, cached_path, digest, size)
            self._bytes += size

    def _discard(self, path: Path) -> None:
        cached = self._entries.pop(path)
        self._bytes -= cached.size
        cached.path.unlink(missing_ok=True)


_snapshots = _SnapshotCache()


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
                digest = await run_in_threadpool(_snapshots.link, self.path, snapshot)
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
