"""One model call for routing and grounded answers."""

import asyncio
import re
import time
import uuid
from typing import Annotated, Literal

from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from app.assistant.client import AssistantUnavailable, CompletionClient
from app.assistant.prompts import ANSWER_PROMPT, MISSING_EVIDENCE, OUT_OF_SCOPE, PARTIAL_EVIDENCE, PROMPT_VERSION
from app.assistant.retrieval import Evidence, evidence_for, image_for
from app.infra.config import LLMSettings, assistant_deadline
from app.infra.redis_limits import RedisLimiter

EvidenceId = Annotated[str, StringConstraints(pattern=r"^e[1-6]$")]
AnswerMode = Literal["term", "article", "explain", "general", "out_of_scope"]
Coverage = Literal["complete", "partial", "insufficient"]
UNSAFE_TEXT = re.compile(r"://|www\.|<\s*/?[a-z][^>]*>|\]\s*[([]", re.IGNORECASE)


class AssistantConfigurationError(Exception):
    pass


class AssistantOverloaded(Exception):
    pass


class ModelBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2400)]
    evidence_ids: list[EvidenceId] = Field(max_length=6)


class ModelReply(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    mode: AnswerMode
    coverage: Coverage
    definition_id: EvidenceId | None
    blocks: list[ModelBlock] = Field(max_length=8)


def _select_blocks(
    reply: ModelReply, requested_mode: str, evidence: dict[str, Evidence],
) -> tuple[AnswerMode, Coverage, list[tuple[str, list[str]]]]:
    mode = reply.mode
    if mode == "out_of_scope":
        return mode, "complete", []
    if mode == "term" and requested_mode in ("auto", "term"):
        key = reply.definition_id or ""
        item = evidence.get(key)
        if item and item.kind == "term" and item.canonical_definition and reply.coverage == "complete":
            return mode, "complete", [(item.canonical_definition, [key])]
        return mode, "insufficient", []

    if requested_mode == "general" or (mode == "general" and requested_mode == "auto"):
        if mode != "general":
            return "general", "insufficient", []
    else:
        mode = "article" if mode == "article" and requested_mode in ("auto", "article") else "explain"
    grounded = mode != "general"
    blocks = []
    if reply.coverage != "insufficient":
        for block in reply.blocks:
            ids = block.evidence_ids
            if bool(ids) != grounded or any(key not in evidence for key in ids) or UNSAFE_TEXT.search(block.text):
                continue
            blocks.append((block.text, list(dict.fromkeys(ids))))
    coverage = "partial" if len(blocks) < len(reply.blocks) else reply.coverage
    return mode, coverage if blocks else "insufficient", blocks


def render(reply: ModelReply, requested_mode: str, locale: str, evidence: dict[str, Evidence]) -> dict:
    mode, coverage, passages = _select_blocks(reply, requested_mode, evidence)
    source_ids = list(dict.fromkeys(key for _, keys in passages for key in keys))
    indices = {key: index for index, key in enumerate(source_ids)}
    used = [evidence[key] for key in source_ids]
    primary = next((item for item in used if item.kind == mode), None)
    if mode == "article" and primary is None:
        mode = "explain"
    primary = primary or (used[0] if used else None)

    if mode == "out_of_scope":
        text = OUT_OF_SCOPE[locale]
    elif not passages:
        text = MISSING_EVIDENCE[locale]
    else:
        text = "\n\n".join(text for text, _ in passages)
        if coverage == "partial" and mode != "general":
            text += "\n\n" + PARTIAL_EVIDENCE[locale]

    images_by_url: dict[str, dict] = {}
    for item in used:
        image = image_for(item.record, item.section, locale)
        if image:
            images_by_url.setdefault(image["url"], {"record_id": item.record["id"], **image})
    return {
        "mode": mode,
        "grounding": {"out_of_scope": "policy", "general": "model"}.get(mode, "knowledge_base"),
        "answer": text,
        "primary_url": primary.url if primary else None,
        "sources": [{**item.reference(), "url": item.url, "review_status": item.review_status} for item in used],
        "images": list(images_by_url.values())[:3],
        "review_status": ("draft" if any(item.review_status == "draft" for item in used) else "reviewed") if used else None,
        "coverage": coverage,
        "blocks": [{"text": text, "source_indices": [indices[key] for key in keys]} for text, keys in passages],
        "diagnostics": {"prompt_version": PROMPT_VERSION, "model": LLMSettings.from_env().model},
    }


async def answer(
    question: str, mode: str, locale: str, context: dict | None, client: CompletionClient,
    *, trace: dict | None = None,
) -> dict:
    started = time.perf_counter()
    evidence = {} if mode == "general" else await asyncio.to_thread(evidence_for, question, locale)
    hint = (context or {}).get("record_id")
    payload = {
        "question": question,
        "locale": locale,
        "requested_mode": mode,
        "page_evidence_ids": [key for key, item in evidence.items() if item.record["id"] == hint],
        "evidence": [
            {"id": key, "kind": item.kind, "title": item.title, "heading": item.heading,
             "definition": item.canonical_definition is not None, "text": item.text}
            for key, item in evidence.items()
        ],
    }
    if trace is not None:
        trace["retrieval_ms"] = round((time.perf_counter() - started) * 1000, 3)
        trace["evidence"] = [
            {"id": key, **item.reference(), "url": item.url, "text": item.text}
            for key, item in evidence.items()
        ]
    started = time.perf_counter()
    try:
        raw = await client.complete(ANSWER_PROMPT, payload)
    finally:
        if trace is not None:
            trace["llm_ms"] = round((time.perf_counter() - started) * 1000, 3)
    try:
        reply = ModelReply.model_validate(raw)
    except ValidationError as exc:
        raise AssistantUnavailable("Модель вернула ответ вне схемы") from exc
    return render(reply, mode, locale, evidence)


async def run_answer(
    question: str,
    mode: str,
    locale: str,
    context: dict | None,
    client: CompletionClient,
    limiter: RedisLimiter,
) -> dict:
    settings = LLMSettings.from_env()
    if settings.provider not in ("deepseek", "runpod"):
        raise AssistantConfigurationError("Неизвестный провайдер LLM")
    if not settings.api_key:
        raise AssistantConfigurationError("Ключ LLM не настроен")

    token = str(uuid.uuid4())
    if not await run_in_threadpool(limiter.acquire_call, token):
        raise AssistantOverloaded("Помощник перегружен")
    try:
        return await asyncio.wait_for(answer(question, mode, locale, context, client), timeout=assistant_deadline())
    finally:
        await run_in_threadpool(limiter.release_call, token)
