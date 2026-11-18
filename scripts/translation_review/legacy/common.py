"""Shared files, batching and model request for the original translation-cache review."""

import json

import requests

from scripts.translation_review.common import DATA

CACHE = DATA / "processed/.translation-cache.json"
REVIEW_REPORT = DATA / "eval/translation_review.json"
RESOLUTIONS = DATA / "eval/translation_resolutions.json"
DECISIONS = DATA / "eval/translation_applied.json"


def batches(rows: list[dict], max_chars: int) -> list[list[dict]]:
    out: list[list[dict]] = []
    current: list[dict] = []
    size = 0
    for row in rows:
        length = len(row["en"]) + len(row["ru"])
        if current and size + length > max_chars:
            out.append(current)
            current = []
            size = 0
        current.append(row)
        size += length
    if current:
        out.append(current)
    return out


def request_json(system: str, payload: list[dict], key: str) -> dict:
    response = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "deepseek-flash",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "max_tokens": 8192,
            "temperature": 0,
        },
        timeout=120,
    )
    response.raise_for_status()
    return json.loads(response.json()["choices"][0]["message"]["content"])
