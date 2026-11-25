"""GigaChat 3 Ultra adapter for the assistant provider experiment."""

import json
import ssl
import time
from typing import Any
from uuid import uuid4

import httpx

from app.assistant.client import AssistantNotConfigured, AssistantUnavailable, Completion
from experiments.rag_ab.baseline import ModelReply

MODEL = "GigaChat-3-Ultra"
OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
CHAT_URL = "https://api.giga.chat/v1/chat/completions"
RESPONSE_FORMAT = {"type": "json_schema", "schema": ModelReply.model_json_schema(), "strict": True}


def tls_context(ca_file: str) -> ssl.SSLContext:
    context = ssl.create_default_context()
    if ca_file:
        context.load_verify_locations(cafile=ca_file)
    return context


class GigaChatClient:
    def __init__(self, http: httpx.AsyncClient, key: str, max_tokens: int) -> None:
        self.http = http
        self.key = key
        self.max_tokens = max_tokens
        self.token = ""
        self.expires_at = 0.0

    async def access_token(self) -> str:
        if self.token and self.expires_at > time.time() + 30:
            return self.token
        response = await self.http.post(
            OAUTH_URL,
            headers={"Authorization": f"Basic {self.key}", "RqUID": str(uuid4()), "Accept": "application/json"},
            data={"scope": "GIGACHAT_API_PERS"},
        )
        response.raise_for_status()
        body = response.json()
        token, expires_at = body["access_token"], body["expires_at"]
        if not isinstance(token, str) or not token or type(expires_at) is not int:
            raise ValueError("Invalid GigaChat OAuth response")
        self.token = token
        self.expires_at = expires_at / 1000 if expires_at > 10**11 else expires_at
        return self.token

    async def complete(self, system: str, payload: dict[str, Any]) -> Completion:
        usage: dict[str, Any] = {}
        try:
            token = await self.access_token()
            response = await self.http.post(
                CHAT_URL,
                headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
                    ],
                    "response_format": RESPONSE_FORMAT,
                    "max_tokens": self.max_tokens,
                    "stream": False,
                },
            )
            response.raise_for_status()
            body = response.json()
            if isinstance(body.get("usage"), dict):
                usage = body["usage"]
            choice = body["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise AssistantUnavailable(f"GigaChat finish_reason: {choice.get('finish_reason')}", usage=usage)
            return Completion(json.loads(choice["message"]["content"]), body["model"], usage)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in (401, 402, 403):
                raise AssistantNotConfigured(f"GigaChat: HTTP {status}", status) from exc
            raise AssistantUnavailable(f"GigaChat: HTTP {status}") from exc
        except (httpx.HTTPError, AttributeError, KeyError, IndexError, ValueError, TypeError) as exc:
            raise AssistantUnavailable(f"GigaChat: {type(exc).__name__}", usage=usage) from exc
