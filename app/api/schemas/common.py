from typing import Any, Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str


class ReadyResponse(BaseModel):
    status: Literal["ready", "degraded", "unavailable"]
    catalogue: Literal["ready", "unavailable"]
    semantic: Literal["ready", "unavailable"]
    hybrid_required: bool


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Any = None
