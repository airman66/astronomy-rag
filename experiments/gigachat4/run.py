"""Freeze production RAG inputs and run a resumable, paired model comparison."""

import argparse
import asyncio
import fcntl
import hashlib
import json
import time
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from app.assistant.prompts import ANSWER_PROMPT, PROMPT_VERSION
from app.assistant.retrieval import PACKING_VERSION
from app.assistant.service import ModelReply, RequestMode, prepare_answer, render
from app.assistant.source_policy import ArticleScope
from app.catalog import storage
from app.infra.config import load_env
from app.infra.files import checksum, write_json
from experiments.gigachat4.providers import DEEPSEEK, GIGACHAT, MODELS, Candidate
from experiments.rag_ab.live import hydrate, run_batches
from experiments.rag_ab.retrieval import PackedEvidence, evidence_row

EVAL = Path(__file__).resolve().parents[2] / "data/eval/assistant"
QUESTIONS = EVAL / "astronomy_v1_questions.json"
PROBE_SYSTEM = 'Return exactly the JSON format {"answer": 0}. Replace 0 with a number; omit all units.'
PROBE_PAYLOAD = {
    "question": "A 90-minute period repeats five times. How many seconds elapsed?"
}


def digest(data: Any) -> str:
    return hashlib.sha256(
        json.dumps(data, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def input_provenance() -> dict:
    return {
        "questions_sha256": checksum(QUESTIONS),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": digest(ANSWER_PROMPT),
        "packing_version": PACKING_VERSION,
        "corpus_sha256": storage.corpus_version(),
        "index_sha256": checksum(storage.DATA / "index" / "index-manifest.json"),
    }


async def freeze(out: Path) -> None:
    path = out / "inputs.json"
    provenance = input_provenance()
    if path.exists():
        if read_json(path)["configuration"] != provenance:
            raise SystemExit("Inputs changed; select a new output directory")
        print(f"Reusing {path}")
        return
    rows = []
    for case in read_json(QUESTIONS)["questions"]:
        mode: RequestMode = case.get("requested_mode", "auto")
        context = (
            {"record_id": case["context_record_id"]}
            if "context_record_id" in case
            else None
        )
        trace: dict = {}
        payload, evidence, scope = await prepare_answer(
            case["question"], mode, case["locale"], context, trace=trace
        )
        if mode != "general" and trace.get("search_mode") != "hybrid":
            raise SystemExit(
                f"{case['id']}: hybrid search not available; check EMBEDDING_URL"
            )
        rows.append(
            {
                "id": case["id"],
                "payload": payload,
                "scope": {
                    "record_ids": scope.record_ids,
                    "unresolved": scope.unresolved,
                }
                if scope
                else None,
                "evidence": [evidence_row(key, item) for key, item in evidence.items()],
                "retrieval_ms": trace["retrieval_ms"],
                "search_mode": trace.get("search_mode", "skipped"),
                "vector_hits": trace.get("vector_hits", 0),
            }
        )
        if len(rows) % 20 == 0:
            print("freeze", len(rows), flush=True)
    if provenance != input_provenance():
        raise SystemExit("Corpus, index or question set changed during retrieval")
    write_json(
        path, {"configuration": provenance, "system": ANSWER_PROMPT, "inputs": rows}
    )


async def preflight(candidate: Candidate, http: httpx.AsyncClient) -> dict:
    outcome = await candidate.call(http, PROBE_SYSTEM, PROBE_PAYLOAD)
    observed_model = outcome.get("reported_model")
    if (
        "error" in outcome
        or outcome.get("finish_reason") != "stop"
        or not outcome["reasoning_observed"]
        or not observed_model
    ):
        raise SystemExit(
            f"{candidate.name}: preflight failed: {outcome.get('error') or 'reasoning/model not confirmed'}"
        )
    try:
        correct = float(json.loads(outcome["content"])["answer"]) == 27000
    except (TypeError, ValueError, KeyError):
        correct = False
    if not correct:
        raise SystemExit(
            f"{candidate.name}: preflight did not return the expected JSON"
        )
    return {key: value for key, value in outcome.items() if key != "content"}


async def generate(
    system: str,
    snapshot: dict,
    candidate: Candidate,
    http: httpx.AsyncClient,
    timeout: int,
) -> dict:
    row = {
        "id": snapshot["id"],
        "provider": candidate.name,
        "request_sha256": digest([system, snapshot["payload"]]),
    }
    started = time.perf_counter()
    try:
        async with asyncio.timeout(timeout):
            row.update(await candidate.call(http, system, snapshot["payload"]))
    except TimeoutError:
        row["error"] = "deadline"
    row["llm_ms"] = round((time.perf_counter() - started) * 1000, 3)
    if row.get("http_status") in (401, 402, 403):
        raise SystemExit(f"{candidate.name}: {row['error']}")
    if "error" in row:
        return row
    if row["finish_reason"] != "stop":
        row["error"] = f"finish_reason={row['finish_reason']}"
        return row
    try:
        reply = ModelReply.model_validate_json(row.pop("content") or "")
    except ValidationError:
        row["error"] = "answer schema"
        return row
    known: Sequence[PackedEvidence] = snapshot["evidence"]
    scope = snapshot["scope"]
    payload = snapshot["payload"]
    row["proposed"] = reply.model_dump()
    row["result"] = render(
        reply,
        reply.blocks,
        payload["requested_mode"],
        payload["locale"],
        hydrate(known),
        row["reported_model"],
        question=payload["question"],
        article_scope=ArticleScope(tuple(scope["record_ids"]), scope["unresolved"])
        if scope
        else None,
    )
    return row


async def run(
    out: Path, names: list[str], max_tokens: int, timeout: int, concurrency: int
) -> None:
    inputs = read_json(out / "inputs.json")
    if (
        inputs["configuration"] != input_provenance()
        or inputs["system"] != ANSWER_PROMPT
    ):
        raise SystemExit(
            "Frozen inputs do not match the current catalogue, index or prompt"
        )
    inputs_sha = checksum(out / "inputs.json")
    snapshots = {row["id"]: row for row in inputs["inputs"]}
    candidates = {name: Candidate.from_env(name, max_tokens) for name in names}
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout, connect=10),
        transport=httpx.AsyncHTTPTransport(retries=0),
    ) as http:
        for name, candidate in candidates.items():
            path = out / f"{name}-configuration.json"
            configuration = {
                "request": candidate.configuration(),
                "inputs_sha256": inputs_sha,
                "provider": name,
                "timeout_s": timeout,
                "concurrency_per_provider": concurrency,
            }
            if path.exists():
                if read_json(path)["configuration"] != configuration:
                    raise SystemExit(
                        f"{name}: existing responses use a different configuration"
                    )
            else:
                proof = await preflight(candidate, http)
                write_json(path, {"configuration": configuration, "preflight": proof})
                print(name, "preflight OK", proof["reported_model"], flush=True)

        async def run_candidate(candidate: Candidate) -> None:
            async def worker(case_id: str) -> dict:
                return await generate(
                    inputs["system"], snapshots[case_id], candidate, http, timeout
                )

            await run_batches(
                list(snapshots),
                worker,
                out / f"{candidate.name}-answers.jsonl",
                lambda row: row["id"],
                concurrency,
            )

        await asyncio.gather(
            *(run_candidate(candidate) for candidate in candidates.values())
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "probe", "run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provider", action="append", choices=tuple(MODELS))
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    if min(args.max_tokens, args.timeout, args.concurrency) <= 0:
        parser.error("token budget, timeout and concurrency must be positive")
    providers = list(dict.fromkeys(args.provider or [DEEPSEEK, GIGACHAT]))
    load_env()
    args.out.mkdir(parents=True, exist_ok=True)

    async def execute() -> None:
        if args.action == "freeze":
            await freeze(args.out)
        elif args.action == "run":
            await run(
                args.out, providers, args.max_tokens, args.timeout, args.concurrency
            )
        else:
            async with httpx.AsyncClient(timeout=args.timeout) as http:
                for name in providers:
                    print(
                        name,
                        await preflight(
                            Candidate.from_env(name, args.max_tokens), http
                        ),
                    )

    lock_names = (
        sorted(providers)
        if args.action == "run"
        else [args.action]
        if args.action == "freeze"
        else []
    )
    with ExitStack() as locks:
        for name in lock_names:
            lock = locks.enter_context((args.out / f".{name}.lock").open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit(
                    f"{name}: an experiment is already writing to {args.out}"
                ) from None
        asyncio.run(execute())


if __name__ == "__main__":
    main()
