"""One model call for routing and grounded answers."""

import asyncio
import logging
import re
import time
import uuid
from typing import Annotated, Literal

import redis
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from app.assistant.client import AssistantNotConfigured, AssistantUnavailable, CompletionClient, LLMClient
from app.assistant.prompts import (
    ANSWER_PROMPT,
    MISSING_EVIDENCE,
    MISSING_GENERAL_ANSWER,
    OUT_OF_SCOPE,
    PARTIAL_EVIDENCE,
    PROMPT_VERSION,
)
from app.assistant.retrieval import EVIDENCE_LIMIT, Evidence, evidence_for, image_for
from app.infra.config import assistant_deadline
from app.infra.redis_limits import RedisLimiter

EvidenceId = Annotated[str, StringConstraints(max_length=64)]
RequestMode = Literal["auto", "term", "article", "explain", "general"]
AnswerMode = Literal["term", "article", "explain", "general", "out_of_scope"]
Coverage = Literal["complete", "partial", "insufficient"]
UNSAFE_TEXT = re.compile(r"://|www\.|<\s*/?[a-z][^>]*>|\]\s*[([]", re.IGNORECASE)
log = logging.getLogger(__name__)


class AssistantOverloaded(Exception):
    pass


class ModelBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2400)]
    evidence_ids: list[EvidenceId] = Field(max_length=EVIDENCE_LIMIT)


class ModelReply(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    mode: AnswerMode
    coverage: Coverage
    definition_id: EvidenceId | None
    blocks: list[ModelBlock] = Field(max_length=8)


def _select_blocks(
    reply: ModelReply, requested_mode: RequestMode, evidence: dict[str, Evidence],
) -> tuple[AnswerMode, Coverage, list[tuple[str, list[str]]]]:
    mode = reply.mode
    if mode == "out_of_scope":
        return mode, "complete", []
    article_fallback = requested_mode == "article" and mode == "explain"
    if requested_mode != "auto" and requested_mode != mode and not article_fallback:
        return ("explain" if requested_mode == "article" else requested_mode), "insufficient", []
    if reply.coverage == "insufficient":
        return ("explain" if mode == "article" else mode), "insufficient", []
    if mode == "term":
        key = reply.definition_id or ""
        item = evidence.get(key)
        if item and item.kind == "term" and item.canonical_definition and reply.coverage == "complete":
            return mode, "complete", [(item.canonical_definition, [key])]
        return mode, "insufficient", []

    grounded = mode != "general"
    blocks = []
    for block in reply.blocks:
        ids = block.evidence_ids
        if bool(ids) != grounded or any(key not in evidence for key in ids) or UNSAFE_TEXT.search(block.text):
            continue
        blocks.append((block.text, list(dict.fromkeys(ids))))
    missing_article = mode == "article" and not any(evidence[key].kind == "article" for _, keys in blocks for key in keys)
    if missing_article:
        mode = "explain"
    if not blocks:
        return mode, "insufficient", []
    if article_fallback or missing_article or len(blocks) < len(reply.blocks):
        return mode, "partial", blocks
    return mode, reply.coverage, blocks


def render(
    reply: ModelReply, requested_mode: RequestMode, locale: str, evidence: dict[str, Evidence], model: str,
) -> dict:
    mode, coverage, passages = _select_blocks(reply, requested_mode, evidence)
    source_ids = list(dict.fromkeys(key for _, keys in passages for key in keys))
    indices = {key: index for index, key in enumerate(source_ids)}
    used = [evidence[key] for key in source_ids]
    primary = next((item for item in used if item.kind == mode), used[0] if used else None)

    if mode == "out_of_scope":
        text = OUT_OF_SCOPE[locale]
    elif not passages:
        text = (MISSING_GENERAL_ANSWER if mode == "general" else MISSING_EVIDENCE)[locale]
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
        "diagnostics": {"prompt_version": PROMPT_VERSION, "model": model},
    }


async def answer(
    question: str, mode: RequestMode, locale: str, context: dict | None, client: CompletionClient,
    *, trace: dict | None = None, allow_index_load: bool = True,
) -> dict:
    started = time.perf_counter()
    hint = (context or {}).get("record_id")
    if not isinstance(hint, str):
        hint = None
    evidence = {} if mode == "general" else await asyncio.to_thread(
        evidence_for, question, locale, prefer_article=mode == "article",
        context_record_id=hint, trace=trace, allow_index_load=allow_index_load,
    )
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
        completion = await client.complete(ANSWER_PROMPT, payload)
    except AssistantUnavailable as exc:
        if trace is not None:
            trace["usage"] = exc.usage
        raise
    finally:
        if trace is not None:
            trace["llm_ms"] = round((time.perf_counter() - started) * 1000, 3)
    if trace is not None:
        trace["usage"] = completion.usage
    try:
        reply = ModelReply.model_validate(completion.data)
    except ValidationError as exc:
        raise AssistantUnavailable("Модель вернула ответ вне схемы", usage=completion.usage) from exc
    return render(reply, mode, locale, evidence, completion.model)


async def run_answer(
    question: str,
    mode: RequestMode,
    locale: str,
    context: dict | None,
    client: CompletionClient,
    limiter: RedisLimiter,
) -> dict:
    if isinstance(client, LLMClient):
        client.configured_settings()
    try:
        deadline = assistant_deadline()
    except ValueError as exc:
        raise AssistantNotConfigured("Некорректный таймаут помощника") from exc

    token = str(uuid.uuid4())
    if not await run_in_threadpool(limiter.acquire_call, token):
        raise AssistantOverloaded("Помощник перегружен")
    try:
        return await asyncio.wait_for(
            answer(question, mode, locale, context, client, allow_index_load=False), timeout=deadline,
        )
    finally:
        try:
            await run_in_threadpool(limiter.release_call, token)
        except redis.RedisError:
            log.warning("Could not release the assistant slot; it will expire", exc_info=True)
