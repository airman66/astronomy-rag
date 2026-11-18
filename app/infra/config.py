"""Environment configuration shared by the service and data tools."""

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def data_directory(default: Path = Path("data")) -> Path:
    return Path(os.getenv("DATA_DIR", str(default)))


def cors_origins() -> list[str]:
    return [origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()]


def assistant_deadline() -> float:
    return float(os.getenv("ASSISTANT_DEADLINE", "100"))


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    api_key: str = field(repr=False)
    endpoint_id: str = ""

    @classmethod
    def from_env(cls) -> "LLMSettings":
        provider = os.getenv("ASSISTANT_PROVIDER", "deepseek").lower()
        if provider == "runpod":
            return cls(
                provider,
                os.getenv("RUNPOD_MODEL", "glm-5.3"),
                os.getenv("RUNPOD_API_KEY", ""),
                os.getenv("RUNPOD_ENDPOINT_ID", ""),
            )
        return cls(provider, os.getenv("DEEPSEEK_MODEL", "deepseek-flash"), os.getenv("DEEPSEEK_API_KEY", ""))


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
