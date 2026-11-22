"""Run model reviews and independent verification over pending units."""

import asyncio
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

from app.assistant.client import LLMClient
from app.infra.config import LLMSettings
from scripts.translation_review.collect import definition
from scripts.translation_review.common import (
    ISSUE_TYPES,
    REVIEW,
    REVIEW_MODEL,
    REVIEW_PROMPT_VERSION,
    REVIEW_VERSIONS,
    VERDICTS,
    VERIFY,
    VERIFY_MODEL,
    VERIFY_PROMPT_VERSION,
    VERIFY_VERSIONS,
    append_jsonl,
    latest,
    now,
    read_jsonl,
)
from scripts.translation_review.prompts import REVIEW_SYSTEM, VERIFY_SYSTEM

API_URL = "https://api.deepseek.com/chat/completions"


def complete(system: str, payload: dict, model: str, key: str, backend: str, max_tokens: int) -> dict:
    if backend == "runpod":
        settings = LLMSettings(
            "runpod", model, os.environ["RUNPOD_API_KEY"], os.environ["RUNPOD_ENDPOINT_ID"],
            max_tokens=max_tokens, thinking="disabled", reasoning_effort="none",
        )
        return asyncio.run(LLMClient(settings=settings, read_timeout=300).complete(system, payload)).data
    if backend == "ollama":
        response = requests.post(
            "http://127.0.0.1:11434/api/chat",
            json={
                "model": model, "format": "json", "stream": False, "think": False,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                "options": {"num_predict": max_tokens, "num_ctx": 16384, "temperature": 0},
            },
            timeout=300,
        )
        response.raise_for_status()
        return json.loads(response.json()["message"]["content"])
    return deepseek(system, payload, model, key, max_tokens)


def deepseek(system: str, payload: Any, model: str, key: str, max_tokens: int = 16000) -> dict[str, Any]:
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "enabled"},
            "max_tokens": max_tokens,
            "stream": False,
        },
        timeout=600,
    )
    response.raise_for_status()
    body = response.json()
    return json.loads(body["choices"][0]["message"]["content"])


def review_batches(
    units: dict[str, dict[str, Any]], records: dict[str, dict[str, Any]], done: set[str], max_chars: int
) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units.values():
        if unit["id"] not in done:
            groups[unit["context_record"]].append(unit)
    order = sorted(groups, key=lambda rid: (0, int(rid[5:])) if rid.startswith("term:") else (1, rid))
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    size = 0
    for record_id in order:
        for unit in groups[record_id]:
            length = len(unit["en"]) + len(unit["ru"]) + 200
            if current and size + length > max_chars:
                batches.append(current)
                current, size = [], 0
            current.append(unit)
            size += length
    if current:
        batches.append(current)
    return batches


def review_payload(batch: list[dict[str, Any]], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for unit in batch:
        record = records[unit["context_record"]]
        group = groups.setdefault(
            unit["context_record"],
            {
                "term_en": record["locales"]["en"]["title"],
                "term_ru": record["locales"]["ru"]["title"],
                "definition_en": definition(record, "en"),
                "items": [],
            },
        )
        entry = {"id": unit["id"][:16], "kind": "/".join(unit["kinds"]), "en": unit["en"], "ru": unit["ru"]}
        if unit.get("target_ru_title"):
            entry["target_en_title"] = unit["target_en_title"]
            entry["target_ru_title"] = unit["target_ru_title"]
        group["items"].append(entry)
    return {"groups": list(groups.values())}


def verification_item(
    review: dict[str, Any],
    units: dict[str, dict[str, Any]],
    records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Use the same input fields for online and exported verification."""
    unit = units[review["unit"]]
    record = records[unit["context_record"]]
    return {
        "id": review["unit"][:16],
        "kind": "/".join(unit["kinds"]),
        "term_en": record["locales"]["en"]["title"],
        "en": unit["en"],
        "ru_current": unit["ru"],
        "ru_proposed": review["corrected_ru"],
        "reviewer_issues": review["issues"],
        "target_ru_title": unit.get("target_ru_title"),
    }


def validate_review(result: dict[str, Any], batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected = {unit["id"][:16]: unit for unit in batch}
    items = result.get("items")
    if not isinstance(items, list):
        raise ValueError("items is not a list")
    seen: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict) or item.get("id") not in expected or item["id"] in seen:
            raise ValueError("unexpected or duplicate id")
        verdict = item.get("verdict")
        if verdict not in VERDICTS:
            raise ValueError(f"bad verdict {verdict!r}")
        issues = item.get("issues") or []
        if not isinstance(issues, list) or any(not isinstance(issue, dict) for issue in issues):
            raise ValueError("bad issues")
        corrected = item.get("corrected_ru") or ""
        if not isinstance(corrected, str):
            raise ValueError("bad corrected_ru")
        unit = expected[item["id"]]
        if verdict == "fix" and (not corrected.strip() or corrected == unit["ru"]):
            verdict = "uncertain" if issues else "ok"
        seen[item["id"]] = {
            "unit": unit["id"],
            "verdict": verdict,
            "issues": [
                {
                    "type": issue.get("type") if issue.get("type") in ISSUE_TYPES else "other",
                    "note": str(issue.get("note", ""))[:600],
                }
                for issue in issues
            ][:8],
            "corrected_ru": corrected if verdict == "fix" else "",
            "note": str(item.get("note") or "")[:800],
        }
    if set(seen) != set(expected):
        raise ValueError(f"missing ids: {sorted(set(expected) - set(seen))[:5]}")
    return list(seen.values())


def run_review(
    units: dict[str, dict[str, Any]],
    records: dict[str, dict[str, Any]],
    key: str,
    workers: int,
    max_chars: int,
    limit: int | None = None,
) -> None:
    done = {row["unit"] for row in read_jsonl(REVIEW) if row.get("prompt_version") in REVIEW_VERSIONS}
    batches = review_batches(units, records, done, max_chars)[:limit]
    print(
        f"review: units={len(units)} done={len(done)} batches={len(batches)} model={REVIEW_MODEL}", flush=True
    )

    def work(batch: list[dict[str, Any]]) -> int:
        last: Exception | None = None
        for attempt in range(4):
            try:
                result = complete(REVIEW_SYSTEM, review_payload(batch, records), REVIEW_MODEL, key,
                                  os.getenv("AUDIT_REVIEW_BACKEND", "deepseek"), 16000)
                rows = validate_review(result, batch)
                stamp = now()
                append_jsonl(
                    REVIEW,
                    [
                        {
                            **row,
                            "model": REVIEW_MODEL,
                            "prompt_version": REVIEW_PROMPT_VERSION,
                            "reviewed_at": stamp,
                        }
                        for row in rows
                    ],
                )
                return len(rows)
            except Exception as exc:  # network, JSON and contract errors are retried
                last = exc
                time.sleep(min(4 * 2**attempt, 60))
        if len(batch) > 1:  # isolate a problematic item instead of failing the whole package
            middle = len(batch) // 2
            return work(batch[:middle]) + work(batch[middle:])
        print(f"review failed for unit {batch[0]['id'][:12]}: {type(last).__name__}: {last}", flush=True)
        return 0

    completed = len(done)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, batch) for batch in batches]
        for future in as_completed(futures):
            completed += future.result()
            print(f"reviewed {completed}/{len(units)}", flush=True)


def run_verify(
    units: dict[str, dict[str, Any]], records: dict[str, dict[str, Any]], key: str, workers: int
) -> None:
    reviews = latest(read_jsonl(REVIEW), REVIEW_VERSIONS)
    done = {row["unit"] for row in read_jsonl(VERIFY) if row.get("prompt_version") in VERIFY_VERSIONS}
    pending = [
        reviews[unit_id]
        for unit_id in reviews
        if reviews[unit_id]["verdict"] == "fix" and unit_id not in done and unit_id in units
    ]
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    size = 0
    for row in pending:
        unit = units[row["unit"]]
        length = len(unit["en"]) + 2 * len(unit["ru"]) + 300
        if current and (size + length > 9000 or len(current) >= 12):
            batches.append(current)
            current, size = [], 0
        current.append(row)
        size += length
    if current:
        batches.append(current)
    print(
        f"verify: proposals={len(pending) + len(done)} done={len(done)} batches={len(batches)} model={VERIFY_MODEL}",
        flush=True,
    )

    def work(batch: list[dict[str, Any]]) -> int:
        payload = {"items": [verification_item(row, units, records) for row in batch]}
        expected = {row["unit"][:16]: row["unit"] for row in batch}
        last: Exception | None = None
        for attempt in range(4):
            try:
                result = complete(VERIFY_SYSTEM, payload, VERIFY_MODEL, key,
                                  os.getenv("AUDIT_VERIFY_BACKEND", "deepseek"), 12000)
                results = result.get("results")
                if not isinstance(results, list):
                    raise ValueError("results is not a list")
                decisions = {}
                for item in results:
                    if (
                        not isinstance(item, dict)
                        or item.get("id") not in expected
                        or item.get("decision") not in {"accept", "reject", "uncertain"}
                    ):
                        raise ValueError("bad verification item")
                    decisions[item["id"]] = item
                if set(decisions) != set(expected):
                    raise ValueError("missing verification ids")
                stamp = now()
                append_jsonl(
                    VERIFY,
                    [
                        {
                            "unit": expected[i],
                            "decision": d["decision"],
                            "reason": str(d.get("reason") or "")[:800],
                            "model": VERIFY_MODEL,
                            "prompt_version": VERIFY_PROMPT_VERSION,
                            "verified_at": stamp,
                        }
                        for i, d in decisions.items()
                    ],
                )
                return len(decisions)
            except Exception as exc:
                last = exc
                time.sleep(min(4 * 2**attempt, 60))
        if len(batch) > 1:
            middle = len(batch) // 2
            return work(batch[:middle]) + work(batch[middle:])
        print(f"verify failed for unit {batch[0]['unit'][:12]}: {type(last).__name__}: {last}", flush=True)
        return 0

    completed = len(done)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(work, batch) for batch in batches]):
            completed += future.result()
            print(f"verified {completed}/{len(pending) + len(done)}", flush=True)
