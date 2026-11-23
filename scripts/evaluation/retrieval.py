"""Full RU/EN assistant evaluation using the production orchestration."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from app.assistant.client import AssistantNotConfigured, AssistantUnavailable, LLMClient
from app.assistant.prompts import PROMPT_VERSION
from app.assistant.retrieval import evidence_for
from app.assistant.service import answer
from app.infra.config import LLMSettings, assistant_deadline, data_directory, load_env
from app.infra.files import write_json

DATA = data_directory()
EVALUATION = Path(os.getenv("DATA_DIR", "data")) / "eval" / "assistant"


def matches_source(item: Mapping[str, object], selector: str) -> bool:
    record, _, section = selector.partition("#")
    return item["record_id"] == record and (not section or item["section_id"] == section)


def covers(groups: list[list[str]], sources: Sequence[Mapping[str, object]]) -> bool:
    return all(any(matches_source(item, selector) for item in sources for selector in group) for group in groups)


def passage_rank(passage: dict, sources: list[dict]) -> int | None:
    quote = " ".join(passage["quote"].split())
    if not quote:
        raise ValueError("Passage quote must not be empty")
    for index, source in enumerate(sources, 1):
        if matches_source(source, passage["source"]) and quote in " ".join(source.get("text", "").split()):
            return index
    return None


def evaluate_retrieval(case: dict) -> dict | None:
    mode = case.get("requested_mode", "auto")
    groups = list(case.get("evidence", []))
    for fact in case.get("facts", []):
        groups.extend(fact.get("evidence", []))
    passages = case.get("passages", [])
    has_expectations = bool(groups or passages)
    if mode == "general" and not has_expectations:
        return None
    trace: dict = {}
    evidence = {} if mode == "general" else evidence_for(
        case["question"], case["locale"], prefer_article=mode == "article",
        context_record_id=case.get("context_record_id"), trace=trace,
    )
    sources = []
    fragments = []
    for item in evidence.values():
        source = item.reference()
        sources.append(source)
        fragments.append({**source, "text": item.text})
    missing = [group for group in groups if not covers([group], sources)]
    ranks = [passage_rank(passage, fragments) for passage in passages]
    return {
        "id": case["id"], "evidence": sources, "missing_groups": missing,
        "passage_ranks": ranks, "missing_passages": [p for p, rank in zip(passages, ranks) if rank is None],
        "all_expected": (not missing and all(rank is not None for rank in ranks)) if has_expectations else None,
        **trace,
    }


def valid_sources(result: dict, evidence: Sequence[Mapping[str, object]]) -> bool:
    mode = result["mode"]
    sources = result["sources"]
    blocks = result["blocks"]
    linked = {index for block in blocks for index in block["source_indices"]}
    known = {(row["record_id"], row["section_id"], row.get("chunk_id")): row for row in evidence}
    expected_grounding = {"out_of_scope": "policy", "general": "model"}.get(mode, "knowledge_base")
    if mode == "out_of_scope" or result["coverage"] == "insufficient":
        source_shape = not (sources or result["primary_url"] or result["images"] or blocks)
    elif mode == "general":
        source_shape = not (sources or result["primary_url"] or result["images"])
    else:
        source_shape = result["primary_url"] in {source["url"] for source in sources} and all(
            block["source_indices"] for block in blocks
        )
        if mode == "article":
            source_shape = source_shape and any(
                source["record_id"].startswith("article:") and source["url"] == result["primary_url"]
                for source in sources
            )
    if not source_shape or linked != set(range(len(sources))) or result["grounding"] != expected_grounding:
        return False
    for source in sources:
        key = source["record_id"], source["section_id"], source["chunk_id"]
        if key not in known or source["url"] != known[key]["url"]:
            return False
    return True


def evaluate_response(case: dict, result: dict, evidence: Sequence[Mapping[str, object]]) -> dict:
    sources = result["sources"]
    blocks = result["blocks"]
    facts: dict[str, bool] = {}
    for fact in case.get("facts", []):
        # A fact passes when one block matches all lexical cues and covers its expected citations.
        groups = fact.get("evidence", [])
        facts[fact["id"]] = any(
            all(re.search(pattern, block["text"], re.IGNORECASE | re.DOTALL) for pattern in fact["patterns"])
            and covers(groups, [sources[i] for i in block["source_indices"] if 0 <= i < len(sources)])
            for block in blocks
        )
    forbidden = [pattern for pattern in case.get("forbidden", []) if re.search(pattern, result["answer"], re.I)]
    checks = {
        "mode_correct": result["mode"] in case["expected_modes"],
        "coverage_correct": result["coverage"] in case.get("expected_coverages", ["complete"]),
        "source_valid": valid_sources(result, evidence),
        "required_sources": covers(case.get("evidence", []), sources),
    }
    return {**checks, "facts": facts, "forbidden_hits": forbidden,
            "passed": all(checks.values()) and all(facts.values()) and not forbidden}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return round(sorted(values)[max(0, math.ceil(fraction * len(values)) - 1)], 3)


def price(usage: dict, input_rate: float, output_rate: float, cached_rate: float | None) -> float | None:
    incoming, outgoing = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if not isinstance(incoming, int) or not isinstance(outgoing, int) or min(incoming, outgoing) < 0:
        return None
    if "prompt_cache_hit_tokens" in usage:
        cached = usage["prompt_cache_hit_tokens"]
    else:
        details = usage.get("prompt_tokens_details")
        cached = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
    if not isinstance(cached, int) or not 0 <= cached <= incoming or cached_rate is None:
        cached = 0
    return ((incoming - cached) * input_rate + cached * (cached_rate or 0) + outgoing * output_rate) / 1_000_000


async def live_case(case: dict, client: LLMClient, args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    trace: dict = {}
    row: dict[str, Any] = {"id": case["id"], "question": case["question"], "locale": case["locale"],
                           "expected_modes": case["expected_modes"], "passed": False}
    try:
        result = await asyncio.wait_for(
            answer(case["question"], case.get("requested_mode", "auto"), case["locale"],
                   {"record_id": case["context_record_id"]} if "context_record_id" in case else None,
                   client, trace=trace),
            timeout=assistant_deadline(),
        )
        row.update(result=result, **evaluate_response(case, result, trace["evidence"]))
    except AssistantNotConfigured:
        raise
    except (AssistantUnavailable, TimeoutError) as exc:
        row["error"] = str(exc) or type(exc).__name__
    row["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    row["retrieval_ms"] = trace.get("retrieval_ms")
    row["llm_ms"] = trace.get("llm_ms")
    row["search_mode"] = trace.get("search_mode")
    row["vector_hits"] = trace.get("vector_hits")
    row["usage"] = trace.get("usage", {})
    row["evidence"] = trace.get("evidence", [])
    if args.input_usd is not None and args.output_usd is not None:
        row["estimated_cost_usd"] = price(row["usage"], args.input_usd, args.output_usd, args.cached_input_usd)
    return row


def summarize(rows: list[dict]) -> dict:
    good = sum(row["passed"] for row in rows)
    times = [row["latency_ms"] for row in rows]
    facts = [ok for row in rows for ok in row.get("facts", {}).values()]
    costs = [cost for row in rows if (cost := row.get("estimated_cost_usd")) is not None]
    total = sum(costs) if len(costs) == len(rows) else None
    return {
        "cases": len(rows),
        "passed": good,
        "mode_correct": sum(row.get("mode_correct", False) for row in rows),
        "coverage_correct": sum(row.get("coverage_correct", False) for row in rows),
        "sources_valid": sum(row.get("source_valid", False) for row in rows),
        "facts_verified": sum(facts),
        "facts_checked": len(facts),
        "errors": sum("error" in row for row in rows),
        "p50_ms": percentile(times, 0.5),
        "p95_ms": percentile(times, 0.95),
        "estimated_total_usd": total,
        "estimated_usd_per_1000_valid": 1000 * total / good if total is not None and good else None,
        "confusion": {
            expected: dict(Counter(row.get("result", {}).get("mode", "error")
                                   for row in rows if row["expected_modes"][0] == expected))
            for expected in sorted({row["expected_modes"][0] for row in rows})
        },
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=EVALUATION / "full_questions.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--retrieval-only", action="store_true", help="Check evidence packing without LLM calls")
    parser.add_argument("--require-search-mode", choices=("hybrid", "lexical"),
                        help="Reject retrieval executed in another mode")
    parser.add_argument("--check", action="store_true", help="Exit with an error if any selected case fails")
    parser.add_argument("--case", action="append", help="Run only these case IDs")
    parser.add_argument("--limit", type=int, help="Run the first N selected cases")
    parser.add_argument("--input-usd", type=float, help="USD per million uncached input tokens")
    parser.add_argument("--output-usd", type=float, help="USD per million output tokens")
    parser.add_argument("--cached-input-usd", type=float, help="USD per million cached input tokens")
    args = parser.parse_args()
    if (args.input_usd is None) != (args.output_usd is None) or any(
        not math.isfinite(rate) or rate < 0 for rate in (args.input_usd, args.output_usd, args.cached_input_usd)
        if rate is not None
    ):
        parser.error("Provide both non-negative finite input and output prices")
    load_env()
    source = args.suite.read_bytes()
    cases = json.loads(source)["questions"]
    if args.case:
        cases = [case for case in cases if case["id"] in args.case]
        if set(args.case) - {case["id"] for case in cases}:
            parser.error("Unknown case ID")
    if args.limit is not None:
        cases = cases[:max(0, args.limit)]
    if not cases:
        parser.error("No cases selected")

    manifest = DATA / "index" / "index-manifest.json"
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "suite_sha256": hashlib.sha256(source).hexdigest(),
        "index_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None,
    }
    rows = []
    run_metrics: dict[str, Any]
    if args.retrieval_only:
        for case in cases:
            if (row := evaluate_retrieval(case)) is not None:
                rows.append(row)
        ranks = [rank for row in rows for rank in row["passage_ranks"]]
        run_metrics = {
            "cases": len(rows), "evidence_cases": sum(row["all_expected"] is not None for row in rows),
            "with_all_evidence": sum(row["all_expected"] is True for row in rows),
            "passages_checked": len(ranks), "passages_found": sum(rank is not None for rank in ranks),
            "passage_mrr": round(sum(1 / rank for rank in ranks if rank) / len(ranks), 4) if ranks else None,
        }
    else:
        settings = LLMSettings.from_env()
        if not settings.api_key:
            parser.error(f"No API key configured for {settings.provider}")
        report.update(provider=settings.provider, model=settings.model, max_tokens=settings.max_tokens or None,
                      thinking=settings.thinking, reasoning_effort=settings.reasoning_effort or None,
                      pricing={"input": args.input_usd, "output": args.output_usd, "cached_input": args.cached_input_usd})
        async with httpx.AsyncClient() as http:
            client = LLMClient(http, settings)
            for case in cases:
                try:
                    rows.append(await live_case(case, client, args))
                except AssistantNotConfigured as exc:
                    report["aborted"] = str(exc)
                    break
                print(case["id"], "PASS" if rows[-1]["passed"] else "FAIL", round(rows[-1]["latency_ms"]), "ms")
        run_metrics = summarize(rows) if rows else {"cases": 0}
    retrieval_ids = {case["id"] for case in cases if case.get("requested_mode") != "general"}
    searched = {row["id"]: row for row in rows if row["id"] in retrieval_ids}
    search_modes = Counter(searched.get(case_id, {}).get("search_mode") or "missing" for case_id in retrieval_ids)
    run_metrics["search_modes"] = dict(search_modes)
    run_metrics["with_vector_hits"] = sum(bool(row.get("vector_hits")) for row in searched.values())
    report.update(metrics=run_metrics, results=rows)
    output = args.output or EVALUATION / ("full_retrieval_report.json" if args.retrieval_only else "full_report.json")
    write_json(output, report)
    print(json.dumps(run_metrics, ensure_ascii=False))
    if "aborted" in report:
        raise SystemExit(report["aborted"])
    if args.require_search_mode and set(search_modes) != {args.require_search_mode}:
        raise SystemExit(f"Required {args.require_search_mode} search for every retrieval case. See {output}")
    if args.check:
        outcome = "all_expected" if args.retrieval_only else "passed"
        failures = [row for row in rows if row[outcome] is False]
        for row in failures:
            if args.retrieval_only:
                print(row["id"], "missing", row["missing_groups"], row["missing_passages"])
        if not rows or failures:
            raise SystemExit(f"Evaluation check failed. See {output}")
