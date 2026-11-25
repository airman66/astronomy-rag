"""Run the assistant A/B in two phases with recorded evidence and blind grading."""

import argparse
import asyncio
import fcntl
import hashlib
import json
import random
import re
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any, BinaryIO, Literal, TypedDict

import httpx
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from app.assistant.client import (
    AssistantNotConfigured,
    AssistantUnavailable,
    Completion,
    CompletionClient,
    LLMClient,
)
from app.assistant.prompts import ANSWER_PROMPT, PROMPT_VERSION
from app.assistant.retrieval import PACKING_VERSION, Evidence
from app.assistant.service import ModelReply, answer_payload, render
from app.assistant.source_policy import GLOSSARY_FORBIDDEN
from app.catalog import storage
from app.infra.config import LLMSettings, assistant_deadline, load_env
from app.infra.files import write_json
from experiments.rag_ab import baseline
from experiments.rag_ab.prompts import JUDGE, PLAN, WRITE
from experiments.rag_ab.retrieval import (
    CONFIGS,
    OLD,
    OUT,
    ROOT,
    PackedContext,
    PackedEvidence,
    ms_since,
    packing_config,
    questions_with_sha,
)
from scripts.evaluation.retrieval import evaluate_response, valid_sources

Short = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=280)]
Eid = Annotated[str, StringConstraints(pattern=r"^e[1-8]$")]


class Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class Part(Strict):
    request_span: Short
    evidence_ids: list[Eid] = Field(max_length=8)


class Plan(Strict):
    intent: Literal["definition", "comparison", "explanation", "lookup", "general", "out_of_scope"]
    source_requirement: Literal["none", "glossary", "article", "named_article"]
    source_phrase: Short | None
    fallback: Literal["none", "glossary", "report_missing", "general", "default"]
    parts: list[Part] = Field(max_length=8)


class Grade(Strict):
    label: str
    request_met: bool
    correct: bool
    grounded: bool
    false_refusal: bool
    used_facts: list[bool]
    notes: str


class Grades(Strict):
    grades: list[Grade]


class ThinkingHTTP:
    def __init__(self, http: httpx.AsyncClient) -> None:
        self.http = http

    async def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: httpx.Timeout
    ) -> httpx.Response:
        json.update(thinking={"type": "enabled"}, reasoning_effort="low")
        return await self.http.post(url, headers=headers, json=json, timeout=timeout)


DEFINITION = re.compile(
    r"^(?:что такое |дай(?:те)? определение |define (?:the |an? )?|what is (?:the |an? )?)([^.!?,:;]+)[.!?]*$",
    re.I,
)


def hydrate(packed: Sequence[PackedEvidence]) -> dict[str, Evidence]:
    evidence: dict[str, Evidence] = {}
    for row in packed:
        rid = row["record_id"]
        record = storage.get_record(rid)
        if record is None:
            raise ValueError(f"Evidence record missing from catalogue: {rid}")
        sid = row["section_id"]
        section = next((s for s in record["sections"] if s["id"] == sid), None)
        if section is None:
            section = {"id": sid, "type": "caption", "media_ids": [sid]}
        evidence[row["id"]] = Evidence(
            record,
            section,
            row["kind"],
            row["title"],
            row["heading"],
            row["text"],
            row["canonical_definition"],
            row["url"],
            row["review_status"],
            row["chunk_id"],
        )
    return evidence


def checked_plan(raw: object, question: str, evidence: dict[str, Evidence]) -> dict[str, Any]:
    plan = Plan.model_validate(raw)
    question = " ".join(question.split())
    spans = [p.request_span for p in plan.parts]
    if (plan.intent == "out_of_scope") != (not spans):
        raise ValueError("empty parts for in-scope plan or parts for out-of-scope")
    if len(set(spans)) != len(spans) or any(" ".join(span.split()) not in question for span in spans):
        raise ValueError("plan includes duplicate or non-verbatim request spans")
    if plan.source_phrase and " ".join(plan.source_phrase.split()) not in question:
        raise ValueError("source phrase not found in question")
    if plan.source_requirement == "named_article" and not plan.source_phrase:
        raise ValueError("named article plan needs a source phrase")
    if any(not set(part.evidence_ids) <= evidence.keys() for part in plan.parts):
        raise ValueError("unknown evidence ID in plan")
    if plan.fallback == "glossary" and GLOSSARY_FORBIDDEN.search(question):
        raise ValueError("the question forbids a glossary fallback")
    return plan.model_dump()


def direct_catalog(case: dict, evidence: dict[str, Evidence]) -> dict | None:
    """Exact, unique canonical title with the whole question matching a definition form."""
    if case.get("requested_mode", "auto") not in {"auto", "term"}:
        return None
    match = DEFINITION.fullmatch(case["question"].strip())
    if not match:
        return None
    requested_words = storage.normalized_words(match.group(1))
    matches = [
        key
        for key, item in evidence.items()
        if item.canonical_definition
        and item.kind == "term"
        and storage.normalized_words(item.title) == requested_words
    ]
    if len(matches) != 1:
        return None
    reply = baseline.ModelReply(mode="term", coverage="complete", definition_id=matches[0], blocks=[])
    return render(
        reply, reply.blocks, case.get("requested_mode", "auto"), case["locale"], evidence, "catalog",
        prompt_version=baseline.PROMPT_VERSION,
    )


def usage_totals(usages: Iterable[dict[str, Any]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for usage in usages:
        for name in ("prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens"):
            value = usage.get(name)
            if isinstance(value, int):
                totals[name] += value
        reasoning = usage.get("completion_tokens_details", {}).get("reasoning_tokens")
        if isinstance(reasoning, int):
            totals["reasoning_tokens"] += reasoning
    return dict(totals)


class RecordedCall(TypedDict, total=False):
    role: Literal["plan", "answer"]
    usage: dict[str, Any]
    model: str
    error: str
    ms: float


async def one_call(
    client: CompletionClient,
    system: str,
    payload: dict[str, object],
    role: Literal["plan", "answer"],
    calls: list[RecordedCall],
) -> Completion:
    start = time.perf_counter()
    entry: RecordedCall = {"role": role}
    calls.append(entry)
    try:
        completion = await client.complete(system, payload)
    except AssistantUnavailable as exc:
        entry["usage"] = exc.usage
        entry["error"] = str(exc)
        raise
    else:
        entry["usage"] = completion.usage
        entry["model"] = completion.model
        return completion
    finally:
        entry["ms"] = ms_since(start)


async def trial(
    case: dict,
    snapshot: PackedContext,
    variant: str,
    client: CompletionClient,
    planner: CompletionClient,
    *,
    version: str,
) -> dict:
    start = time.perf_counter()
    calls: list[RecordedCall] = []
    row: dict[str, Any] = {
        "id": case["id"],
        "variant": version,
        "calls": calls,
        "retrieval_ms": snapshot["retrieval_ms"],
    }
    try:
        async with asyncio.timeout(assistant_deadline()):
            evidence = hydrate(snapshot["evidence"])
            if variant == "direct":
                row["result"] = direct_catalog(case, evidence)
                return row
            question = case["question"]
            mode = case.get("requested_mode", "auto")
            payload = answer_payload(question, mode, case["locale"], evidence, case.get("context_record_id"))
            system = baseline.ANSWER_PROMPT if variant == "A" else ANSWER_PROMPT
            if variant in {"C", "C-low"}:
                try:
                    completion = await one_call(planner, PLAN, payload, "plan", calls)
                    plan = checked_plan(completion.data, question, evidence)
                    row["plan"] = plan
                    payload["validated_plan"] = plan
                    system = WRITE
                except AssistantNotConfigured:
                    raise
                except (AssistantUnavailable, ValueError) as exc:
                    row["plan_error"] = "plan schema" if isinstance(exc, ValidationError) else str(exc)
            completion = await one_call(client, system, payload, "answer", calls)
            reply_schema = baseline.ModelReply if variant == "A" else ModelReply
            reply = reply_schema.model_validate(completion.data)
            row["result"] = render(
                reply, reply.blocks, mode, case["locale"], evidence, completion.model,
                question="" if variant == "A" else question,
                page=case.get("context_record_id"),
                prompt_version=baseline.PROMPT_VERSION if variant == "A" else PROMPT_VERSION,
            )
            row["model_mode"] = reply.mode
            row["model_coverage"] = reply.coverage
    except AssistantNotConfigured:
        raise
    except (AssistantUnavailable, ValidationError, TimeoutError) as exc:
        row["error"] = (
            "answer schema" if isinstance(exc, ValidationError) else (str(exc) or type(exc).__name__)
        )
    finally:
        row["answer_ms"] = ms_since(start)
        row["latency_ms"] = round(row["answer_ms"] + snapshot["retrieval_ms"], 3)
        row["usage"] = usage_totals(call.get("usage", {}) for call in calls)
    return row


def blind_label(case_id: str, variant: str, phase: str) -> str:
    return hashlib.sha256(f"2026-11-23:{phase}:{case_id}:{variant}".encode()).hexdigest()[:10]


def blind_answer(row: dict, snapshot: PackedContext, phase: str) -> dict:
    result = row["result"]
    sources = {
        (item["record_id"], item["section_id"], item["chunk_id"]): item for item in snapshot["evidence"]
    }
    linked = []
    for reference in result["sources"]:
        key = (reference["record_id"], reference["section_id"], reference["chunk_id"])
        piece = sources.get(key)
        linked.append(
            {**reference, "text": piece["text"] if piece else "", "title": piece["title"] if piece else ""}
        )
    return {
        "label": blind_label(row["id"], row["variant"], phase),
        "mode": result["mode"],
        "coverage": result["coverage"],
        "answer": result["answer"],
        "blocks": result["blocks"],
        "sources": linked,
    }


def judge_case(case: dict) -> dict:
    if "expected" in case:
        return {"question": case["question"], "expectation": case["expected"], "facts": case["facts"]}
    return {
        "question": case["question"],
        "expectation": "Answer the OAE question using a passage that supports what is stated.",
        "facts": [
            {"claim": "Accurately answer using the statement in this passage in context.", "sources": [p]}
            for p in case["passages"]
        ],
    }


async def grade(case: dict, candidates: list[dict], client: CompletionClient) -> dict[str, dict]:
    payload = {"rubric": judge_case(case), "candidates": candidates}
    expected_labels = {r["label"] for r in candidates}
    expected_facts = len(payload["rubric"]["facts"])
    for _ in range(3):
        try:
            response = await client.complete(JUDGE, payload)
            grades = Grades.model_validate(response.data).grades
            if (
                len(grades) == len(expected_labels)
                and {g.label for g in grades} == expected_labels
                and all(len(g.used_facts) == expected_facts for g in grades)
            ):
                return {g.label: g.model_dump() for g in grades}
        except AssistantNotConfigured:
            raise
        except (AssistantUnavailable, ValidationError):
            continue
    raise RuntimeError(f"Judge failed for {case['id']} (response or schema)")


def read_checkpoint(handle: BinaryIO) -> list[Any]:
    """Recover an interrupted final write while keeping complete records."""
    handle.seek(0)
    lines = handle.readlines()
    tail = lines.pop() if lines and not lines[-1].endswith(b"\n") else None
    rows = [json.loads(line) for line in lines]
    if tail is None:
        return rows
    try:
        rows.append(json.loads(tail))
    except (json.JSONDecodeError, UnicodeDecodeError):
        handle.truncate(handle.tell() - len(tail))
    else:
        handle.write(b"\n")
    return rows


async def run_batches[Key, Row](
    keys: Sequence[Key],
    worker: Callable[[Key], Awaitable[Row]],
    out: Path,
    key_of: Callable[[Row], Key],
    concurrency: int,
) -> dict[Key, Row]:
    with out.open("a+b") as handle:
        existing = read_checkpoint(handle)
        by_key: dict[Key, Row] = {key_of(row): row for row in existing}
        if len(by_key) != len(existing) or not by_key.keys() <= set(keys):
            raise ValueError(f"Unexpected or duplicate checkpoint keys: {out}")
        pending = [key for key in keys if key not in by_key]

        async def save(key: Key) -> None:
            row = await worker(key)
            assert key_of(row) == key
            by_key[key] = row
            handle.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
            handle.flush()

        for offset in range(0, len(pending), concurrency):
            results = await asyncio.gather(
                *(save(key) for key in pending[offset : offset + concurrency]), return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            progress = min(offset + concurrency, len(pending))
            if progress % 20 < concurrency or progress == len(pending):
                print(out.name, progress, "/", len(pending), flush=True)
    return by_key


def generation_order(phase: str, cases: Iterable[dict], prior: dict[str, list[str]]) -> list[tuple[str, str]]:
    jobs = []
    for case in cases:
        case_id = case["id"]
        if phase == "packing":
            versions = list(CONFIGS) if not case_id.startswith("oae-") else []
        elif phase == "llm":
            versions = ["A", "B", "C"]
            if case.get("ambiguous"):
                versions.append("C-low")
            if case.get("catalog_id"):
                versions.append("direct")
        elif phase == "oae":
            versions = ["A-base", "A", "A-wide"]
        elif phase == "legacy":
            versions = ["A", "B", "C"]
        else:
            versions = prior.get(case_id, [])
        jobs.extend((case_id, version) for version in versions)
    random.Random(23791).shuffle(jobs)
    return jobs


def open_checkpoint(phase: str, provenance: dict) -> None:
    """Validate an existing checkpoint or describe a new one before the first write."""
    manifest_path = OUT / f"{phase}-run.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_bytes()) != provenance:
            raise SystemExit(f"Checkpoint inputs differ: {manifest_path}. Use a new RAG_AB_OUTPUT_DIR.")
        return
    for suffix in ("responses", "grades"):
        path = OUT / f"{phase}-{suffix}.jsonl"
        if path.exists() and path.stat().st_size:
            raise SystemExit(f"Checkpoint manifest missing for {path}. Use a new RAG_AB_OUTPUT_DIR.")
    write_json(manifest_path, provenance)


async def run_phase(phase: str, config: str, concurrency: int, settings: LLMSettings) -> None:
    suite, path = {
        "oae": ("oae", OLD / "oae_passage_questions.json"),
        "legacy": ("legacy", OLD / "full_questions.json"),
    }.get(phase, ("new", ROOT / "questions.json"))
    questions, suite_sha = questions_with_sha(path)
    snapshot_path = OUT / ("legacy-retrieval.json" if phase == "legacy" else "retrieval.json")
    snapshot_bytes = snapshot_path.read_bytes()
    retrieval = json.loads(snapshot_bytes)
    if not isinstance(retrieval, dict) or retrieval["suite_sha256"].get(suite) != suite_sha:
        raise SystemExit(f"Retrieval snapshot suite differs: {snapshot_path}. Rebuild the snapshot.")
    if retrieval.get("packing_version") != PACKING_VERSION:
        raise SystemExit(f"Retrieval snapshot packing differs: {snapshot_path}. Rebuild the snapshot.")
    corpus_sha = retrieval["index"]["corpus_sha256"]
    if not corpus_sha or corpus_sha != storage.corpus_version():
        raise SystemExit(f"Retrieval snapshot corpus differs from the active catalogue: {snapshot_path}")
    snapshots = {row["id"]: row for row in retrieval["results"]}
    cases = {case["id"]: case for case in questions}
    is_repeat = phase in {"repeat", "repeat2"}
    prior = json.loads((OUT / "repeat-cases.json").read_bytes()) if is_repeat else {}
    if is_repeat:
        cases = {key: value for key, value in cases.items() if key in prior}
    order = generation_order(phase, cases.values(), prior)

    def snapshot_for(key: tuple[str, str]) -> PackedContext:
        case_id, version = key
        return snapshots[case_id]["configs"][packing_config(phase, version, config)]

    scheduled = {key: snapshot_for(key) for key in order}
    provenance = {
        "phase": phase,
        "context": "factorial" if phase == "packing" else config,
        "packing_version": PACKING_VERSION,
        "concurrency": concurrency,
        "model": settings.model,
        "answer_budget": settings.max_tokens or 768,
        "planner_budget": 1536,
        "thinking": "disabled except C-low planner",
        "system_sha256": {
            key: hashlib.sha256(value.encode()).hexdigest()
            for key, value in {
                "A": baseline.ANSWER_PROMPT,
                "B": ANSWER_PROMPT,
                "planner": PLAN,
                "writer": WRITE,
                "judge": JUDGE,
            }.items()
        },
        "retrieval_index_sha256": retrieval["index_manifest_sha256"],
        "retrieval_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        "suite_sha256": suite_sha,
        "schedule_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest(),
    }
    open_checkpoint(phase, provenance)
    async with httpx.AsyncClient() as http:
        planner_settings = replace(settings, max_tokens=1536)
        client = LLMClient(http, settings)
        planner = LLMClient(http, planner_settings)
        thinking = LLMClient(ThinkingHTTP(http), planner_settings)
        judge = LLMClient(http, replace(settings, max_tokens=2800))

        async def generate(key: tuple[str, str]) -> dict:
            case_id, version = key
            variant = "A" if phase == "packing" or version in {"A-base", "A-wide"} else version
            return await trial(
                cases[case_id],
                scheduled[key],
                variant,
                client,
                thinking if variant == "C-low" else planner,
                version=version,
            )

        rows = await run_batches(
            order,
            generate,
            OUT / f"{phase}-responses.jsonl",
            lambda row: (row["id"], row["variant"]),
            concurrency,
        )
        if phase == "legacy":
            for key, row in rows.items():
                if result := row.get("result"):
                    row.update(evaluate_response(cases[key[0]], result, snapshot_for(key)["evidence"]))
                else:
                    row["passed"] = False
            write_json(OUT / "legacy-scored.json", list(rows.values()))
            return

        candidates_by_case: dict[str, list[dict]] = {}
        for key, row in rows.items():
            if key[0] in cases and row.get("result"):
                candidates_by_case.setdefault(key[0], []).append(blind_answer(row, snapshot_for(key), phase))
        blind = []
        for case in cases.values():
            case_id = case["id"]
            candidates = candidates_by_case.get(case_id, [])
            candidates.sort(key=lambda item: item["label"])
            blind.append(
                {
                    "id": case_id,
                    "locale": case["locale"],
                    "question": case["question"],
                    "candidates": candidates,
                }
            )
        write_json(OUT / f"{phase}-blind.json", blind)

        async def judge_one(case_id: str) -> dict:
            return {"id": case_id, "grades": await grade(cases[case_id], candidates_by_case[case_id], judge)}

        grades = await run_batches(
            [case_id for case_id in cases if case_id in candidates_by_case],
            judge_one,
            OUT / f"{phase}-grades.jsonl",
            lambda row: row["id"],
            concurrency,
        )
        for key, row in rows.items():
            case_id, version = key
            label = blind_label(case_id, version, phase)
            row["label"] = label
            result = row.get("result")
            if not result:
                row.update(quality=False, passed=False)
                continue
            case = cases[case_id]
            judgment = grades[case_id]["grades"][label]
            row["grade"] = judgment
            row["source_valid"] = valid_sources(result, snapshot_for(key)["evidence"])
            row["mode_correct"] = result["mode"] in case.get("expected_modes", ["article", "explain"])
            row["coverage_correct"] = result["coverage"] == case.get("expected_coverage", "complete")
            row["quality"] = (
                judgment["request_met"]
                and judgment["correct"]
                and judgment["grounded"]
                and row["source_valid"]
            )
            row["passed"] = row["quality"] and row["mode_correct"] and row["coverage_correct"]
        write_json(OUT / f"{phase}-scored.json", list(rows.values()))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("packing", "llm", "oae", "legacy", "repeat", "repeat2"))
    parser.add_argument("--config", choices=CONFIGS, default="8x11000")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    load_env()
    settings = LLMSettings.from_env()
    if settings.provider != "deepseek" or settings.model != "deepseek-flash" or not settings.api_key:
        raise SystemExit("Expected deepseek-flash and a configured local key")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f".{args.phase}-run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, PermissionError):
            raise SystemExit(f"Phase {args.phase} is already running in {OUT}") from None
        await run_phase(args.phase, args.config, args.concurrency, settings)


if __name__ == "__main__":
    asyncio.run(main())
