"""Mode selection and grounded answer generation."""

import asyncio
from dataclasses import dataclass
from typing import Any

from app.assistant_prompts import (
    ANSWER_PROMPT,
    GENERAL_PROMPT,
    OUT_OF_SCOPE,
    PROMPT_VERSION,
    SELECT_PROMPT,
)
from app.config import LLMSettings
from app.llm import AssistantUnavailable, CompletionClient
from app.retrieval import candidates_for


def _decision(value: Any) -> tuple[str, str, list[str]]:
    """Validate the mode-selection JSON; json_object mode does not guarantee field types."""
    if not isinstance(value, dict):
        raise AssistantUnavailable("Модель вернула некорректный выбор режима")
    scope = value.get("scope")
    if scope not in ("in_scope", "out_of_scope"):
        raise AssistantUnavailable("Модель вернула некорректную область вопроса")
    mode = value.get("mode")
    identifiers = value.get("record_ids") or []
    if not isinstance(identifiers, list):
        identifiers = []
    return scope, (mode if isinstance(mode, str) else "general"), [x for x in identifiers if isinstance(x, str)]


def _answer(value: Any) -> tuple[str, set[str]]:
    if not isinstance(value, dict) or not isinstance(value.get("answer"), str) or not value["answer"].strip():
        raise AssistantUnavailable("Модель вернула некорректный ответ")
    used = value.get("used_record_ids") or []
    if not isinstance(used, list):
        used = []
    return value["answer"], {x for x in used if isinstance(x, str)}


def matched_context(candidate: dict) -> dict:
    return {
        **candidate,
        "section_id": candidate.get("matched_section_id", candidate["section_id"]),
        "excerpt": candidate.get("matched_excerpt", candidate["excerpt"]),
        "image": candidate.get("matched_image", candidate.get("image")),
    }


@dataclass(frozen=True)
class Selection:
    mode: str
    sources: list[dict]
    suggested_mode: str
    invalid_ids: list[str]


async def select_context(
    question: str, mode: str, locale: str, context: dict | None,
    candidates: list[dict], client: CompletionClient,
) -> Selection:
    scope, suggested_mode, chosen = _decision(await client.complete(SELECT_PROMPT, {
        "question": question, "locale": locale, "context": context or {}, "candidates": candidates,
    }))
    valid = {item["record_id"]: item for item in candidates}
    invalid_ids = [identifier for identifier in chosen if identifier not in valid]
    if scope == "out_of_scope":
        return Selection("out_of_scope", [], suggested_mode, invalid_ids)
    if mode == "auto":
        mode = suggested_mode
        selected = list({x: valid[x] for x in chosen if x in valid}.values())
    else:
        selected = candidates[:5]
    if mode not in ("term", "article", "explain", "general"):
        mode = "general"
    if mode in ("term", "article"):
        matching = next((item for item in selected if item["kind"] == mode), None)
        if matching is None:
            matching = next((item for item in candidates if item["kind"] == mode), None)
        if matching is not None:
            selected = [matching]
        elif mode == "article":
            mode = "explain" if candidates else "general"
            selected = candidates[:5]
        else:
            mode = "general"
    if mode == "explain":
        selected = [matched_context(item) for item in (selected or candidates[:5])[:5]]
        if not selected:
            mode = "general"
    elif mode == "article":
        selected = [matched_context(selected[0])]
    if mode == "general":
        selected = []
    return Selection(mode, selected, suggested_mode, invalid_ids)


async def answer(question: str, mode: str, locale: str, context: dict | None, client: CompletionClient) -> dict:
    # Retrieval is synchronous (SQLite, embedding HTTP call): keep it off the event loop.
    candidates = [] if mode == "general" else await asyncio.to_thread(candidates_for, question, locale)
    selection = await select_context(question, mode, locale, context, candidates, client)
    if selection.mode == "out_of_scope":
        return {
            "mode": "out_of_scope", "grounding": "policy", "answer": OUT_OF_SCOPE[locale],
            "primary_url": None, "sources": [], "images": [], "review_status": None,
            "diagnostics": {"prompt_version": PROMPT_VERSION, "model": "scope_filter"},
        }
    mode, selected = selection.mode, selection.sources
    prompt = GENERAL_PROMPT if mode == "general" else ANSWER_PROMPT
    text, used = _answer(await client.complete(prompt, {"question": question, "locale": locale, "mode": mode, "context": context or {}, "sources": selected}))
    sources = [
        {"record_id": x["record_id"], "section_id": x["section_id"], "url": x["url"], "review_status": x["review_status"]}
        for x in selected if x["record_id"] in used
    ]
    if mode != "general" and not sources:
        mode = "general"
        text, _ = _answer(await client.complete(GENERAL_PROMPT, {"question": question, "locale": locale, "mode": mode, "context": context or {}, "sources": []}))
    return {
        "mode": mode,
        "grounding": "model" if mode == "general" else "knowledge_base",
        "answer": text,
        "primary_url": sources[0]["url"] if sources else None,
        "sources": sources,
        "images": [{"record_id": x["record_id"], **x["image"]} for x in selected if x["record_id"] in used and x.get("image")][:3],
        "review_status": "draft" if any(x["review_status"] == "draft" for x in sources) else ("reviewed" if sources else None),
        "diagnostics": {"prompt_version": PROMPT_VERSION, "model": LLMSettings.from_env().model},
    }
