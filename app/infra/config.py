"""Environment configuration shared by the service and data tools."""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CALL_LEASE_SECONDS = 150
_release = os.getenv("DATA_RELEASE")
RELEASE_DIR = Path(_release).resolve(strict=True) if _release else None


def data_directory(default: Path = Path("data")) -> Path:
    return RELEASE_DIR or Path(os.getenv("DATA_DIR", str(default)))


def cors_origins() -> list[str]:
    return [origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()]


def assistant_deadline() -> float:
    deadline = float(os.getenv("ASSISTANT_DEADLINE") or "100")
    if not math.isfinite(deadline) or not 0 < deadline < CALL_LEASE_SECONDS:
        raise ValueError(f"ASSISTANT_DEADLINE must be greater than 0 and less than {CALL_LEASE_SECONDS}")
    return deadline


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    api_key: str = field(repr=False)
    endpoint_id: str = ""
    max_tokens: int = 0
    thinking: str = "default"
    reasoning_effort: str = ""

    @classmethod
    def from_env(cls) -> "LLMSettings":
        provider = (os.getenv("ASSISTANT_PROVIDER") or "deepseek").lower()
        max_tokens = int(os.getenv("ASSISTANT_MAX_OUTPUT_TOKENS") or "0")
        if max_tokens and not 256 <= max_tokens <= 4096:
            raise ValueError("ASSISTANT_MAX_OUTPUT_TOKENS must be 256–4096 or 0")
        if provider == "runpod":
            return cls(
                provider,
                os.getenv("RUNPOD_MODEL") or "glm-5.3",
                os.getenv("RUNPOD_API_KEY", ""),
                os.getenv("RUNPOD_ENDPOINT_ID", ""),
                max_tokens=max_tokens,
                thinking=os.getenv("RUNPOD_THINKING") or "default",
                reasoning_effort=os.getenv("RUNPOD_REASONING_EFFORT") or "",
            )
        return cls(provider, os.getenv("DEEPSEEK_MODEL") or "deepseek-flash", os.getenv("DEEPSEEK_API_KEY", ""), max_tokens=max_tokens)


def redis_url() -> str:
    return os.getenv("REDIS_URL", "redis://localhost:6379/0")


def load_env() -> None:
    """Load the local CLI environment, preserving variables already set."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
