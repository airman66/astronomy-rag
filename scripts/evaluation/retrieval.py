"""Full RU/EN assistant evaluation using the production orchestration."""

import argparse
import asyncio
import hashlib
import json
import math
import re
import time
from collections import Counter
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
EVALUATION = DATA / "eval" / "assistant"


def matches_source(item: dict, selector: str) -> bool:
    record, _, section = selector.partition("#")
    return item["record_id"] == record and (not section or item["section_id"] == section)


def covers(groups: list, sources: list[dict]) -> bool:
    return all(any(matches_source(item, selector) for item in sources for selector in group) for group in groups)


def evaluate_response(case: dict, result: dict, evidence: list[dict]) -> dict:
    mode = result["mode"]
    sources = result["sources"]
    blocks = result["blocks"]
    linked = {index for block in blocks for index in block["source_indices"]}
    known = {(row["record_id"], row["section_id"], row.get("chunk_id")): row for row in evidence}
    expected_grounding = {"out_of_scope": "policy", "general": "model"}.get(mode, "knowledge_base")
    if mode in ("general", "out_of_scope") or result["coverage"] == "insufficient":
        source_shape = not (sources or result["primary_url"] or result["images"])
    else:
        source_shape = result["primary_url"] in {source["url"] for source in sources} and all(
            block["source_indices"] for block in blocks
        )
    source_valid = (
        source_shape
        and linked == set(range(len(sources)))
        and result["grounding"] == expected_grounding
        and all(
            (key := (source["record_id"], source["section_id"], source["chunk_id"])) in known
            and source["url"] == known[key]["url"]
            for source in sources
        )
    )
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
        "mode_correct": mode in case["expected_modes"],
        "coverage_correct": result["coverage"] in case.get("expected_coverages", ["complete"]),
        "source_valid": source_valid,
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
    details = usage.get("prompt_tokens_details") or {}
    cached = usage.get("prompt_cache_hit_tokens", details.get("cached_tokens", 0))
    if not isinstance(cached, int) or not 0 <= cached <= incoming or cached_rate is None:
        cached = 0
    return ((incoming - cached) * input_rate + cached * (cached_rate or 0) + outgoing * output_rate) / 1_000_000


async def live_case(case: dict, http: httpx.AsyncClient, settings: LLMSettings, args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    trace: dict = {}
    client = LLMClient(http, settings)
    row: dict[str, Any] = {"id": case["id"], "question": case["question"], "locale": case["locale"],
                           "expected_modes": case["expected_modes"], "passed": False}
    try:
        result = await asyncio.wait_for(
            answer(case["question"], case.get("requested_mode", "auto"), case["locale"], None, client, trace=trace),
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
    row["usage"] = client.usage
    row["evidence"] = trace.get("evidence", [])
    if args.input_usd is not None and args.output_usd is not None:
        row["estimated_cost_usd"] = price(client.usage, args.input_usd, args.output_usd, args.cached_input_usd)
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
    if args.retrieval_only:
        for case in cases:
            groups = case.get("evidence", [])
            if groups:
                sources = [item.reference() for item in evidence_for(case["question"], case["locale"]).values()]
                rows.append({"id": case["id"], "evidence": sources, "all_expected": covers(groups, sources)})
        report["metrics"] = {"cases": len(rows), "with_all_evidence": sum(row["all_expected"] for row in rows)}
    else:
        settings = LLMSettings.from_env()
        if not settings.api_key:
            parser.error(f"No API key configured for {settings.provider}")
        report.update(provider=settings.provider, model=settings.model, max_tokens=settings.max_tokens or None,
                      thinking=settings.thinking, reasoning_effort=settings.reasoning_effort or None,
                      pricing={"input": args.input_usd, "output": args.output_usd, "cached_input": args.cached_input_usd})
        async with httpx.AsyncClient() as http:
            for case in cases:
                try:
                    rows.append(await live_case(case, http, settings, args))
                except AssistantNotConfigured as exc:
                    report["aborted"] = str(exc)
                    break
                print(case["id"], "PASS" if rows[-1]["passed"] else "FAIL", round(rows[-1]["latency_ms"]), "ms")
        report["metrics"] = summarize(rows) if rows else {"cases": 0}
    report["results"] = rows
    output = args.output or EVALUATION / ("full_retrieval_report.json" if args.retrieval_only else "full_report.json")
    write_json(output, report)
    print(json.dumps(report["metrics"], ensure_ascii=False))
    if "aborted" in report:
        raise SystemExit(report["aborted"])
