from app.api.schemas.common import ErrorResponse

RETRY_HEADER = {"Retry-After": {"description": "Секунды до повторной попытки", "schema": {"type": "integer"}}}
LIMITS = {
    429: {"model": ErrorResponse, "description": "Лимит запросов превышен", "headers": RETRY_HEADER},
    503: {
        "model": ErrorResponse,
        "description": "Сервис ограничений временно недоступен",
        "headers": RETRY_HEADER,
    },
}
IMAGE_TYPES = {"image/png": {}, "image/jpeg": {}, "image/webp": {}, "image/gif": {}, "image/svg+xml": {}}
