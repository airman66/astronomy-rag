"""Compare DeepSeek and GigaChat on the same production RAG inputs."""

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
from collections import Counter

import httpx

from app.assistant.client import LLMClient
from app.assistant.prompts import ANSWER_PROMPT, PROMPT_VERSION
from app.assistant.retrieval import CONTEXT_LIMIT, EVIDENCE_LIMIT
from app.assistant.service import ModelReply
from app.catalog import storage
from app.infra.config import LLMSettings, assistant_deadline, load_env
from app.infra.files import checksum, write_json
from experiments.rag_ab import baseline
from experiments.rag_ab.gigachat import MODEL, RESPONSE_FORMAT, GigaChatClient, tls_context
from experiments.rag_ab.live import open_checkpoint, run_batches, trial, usage_totals
from experiments.rag_ab.retrieval import OLD, OUT, questions_with_sha
from scripts.evaluation.retrieval import evaluate_response, percentile, summarize

PHASE = "providers"


async def compare(variant: str) -> None:
    settings = LLMSettings.from_env()
    key = os.getenv("GIGACHAT_API_KEY")
    if settings.provider != "deepseek" or not settings.api_key or not key:
        raise SystemExit("Set ASSISTANT_PROVIDER=deepseek, DEEPSEEK_API_KEY and GIGACHAT_API_KEY")

    questions, suite_sha = questions_with_sha(OLD / "full_questions.json")
    case_by_id = {case["id"]: case for case in questions}
    raw = (OUT / "legacy-retrieval.json").read_bytes()
    retrieval = json.loads(raw)
    index_sha = checksum(storage.DATA / "index" / "index-manifest.json")
    if (
        not isinstance(retrieval, dict)
        or retrieval["suite_sha256"]["legacy"] != suite_sha
        or retrieval["index"]["corpus_sha256"] != storage.corpus_version()
        or retrieval["index_manifest_sha256"] != index_sha
    ):
        raise SystemExit("Rebuild legacy-retrieval.json against the active catalogue and index")
    if variant == "A":
        packing = f"{baseline.EVIDENCE_LIMIT}x{baseline.CONTEXT_LIMIT}"
        system, prompt_version = baseline.ANSWER_PROMPT, baseline.PROMPT_VERSION
        response_format = RESPONSE_FORMAT
    else:
        packing = f"{EVIDENCE_LIMIT}x{CONTEXT_LIMIT}"
        system, prompt_version = ANSWER_PROMPT, PROMPT_VERSION
        response_format = {"type": "json_schema", "schema": ModelReply.model_json_schema(), "strict": True}
    snapshots = {row["id"]: row["configs"][packing] for row in retrieval["results"]}
    if snapshots.keys() != case_by_id.keys():
        raise SystemExit("Retrieval snapshot doesn't contain the full question suite")

    budget = settings.max_tokens or 768
    provenance = {
        "models": {"deepseek": settings.model, "gigachat": MODEL},
        "answer_budget": budget,
        "deadline_s": assistant_deadline(),
        "prompt_version": prompt_version,
        "prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "gigachat_schema_sha256": hashlib.sha256(json.dumps(response_format, sort_keys=True).encode()).hexdigest(),
        "packing": packing,
        "index_sha256": index_sha,
        "retrieval_sha256": hashlib.sha256(raw).hexdigest(),
        "suite_sha256": suite_sha,
    }
    open_checkpoint(PHASE, provenance)
    providers = ("deepseek", "gigachat")
    order = [(case_id, provider) for case_id in case_by_id for provider in providers]

    timeout = httpx.Timeout(assistant_deadline(), connect=5.0)
    async with (
        httpx.AsyncClient() as deepseek_http,
        httpx.AsyncClient(verify=tls_context(os.getenv("GIGACHAT_CA_FILE", "")), timeout=timeout) as giga_http,
    ):
        gigachat = GigaChatClient(giga_http, key, budget, response_format=response_format)
        await gigachat.access_token()
        clients = {"deepseek": LLMClient(deepseek_http, settings), "gigachat": gigachat}
        # Resuming can leave only one response of a pair in the checkpoint.
        slots = {provider: asyncio.Lock() for provider in providers}

        async def generate(job: tuple[str, str]) -> dict:
            case_id, provider = job
            client = clients[provider]
            async with slots[provider]:
                return await trial(case_by_id[case_id], snapshots[case_id], variant, client, client, version=provider)

        rows = await run_batches(
            order, generate, OUT / "providers-responses.jsonl", lambda row: (row["id"], row["variant"]), 2,
        )
    for (case_id, _), row in rows.items():
        case = case_by_id[case_id]
        row.update(question=case["question"], locale=case["locale"], expected_modes=case["expected_modes"])
        if result := row.get("result"):
            row.update(evaluate_response(case, result, snapshots[case_id]["evidence"]))
        else:
            row["passed"] = False

    metrics = {}
    for provider in providers:
        selected = [rows[case_id, provider] for case_id in case_by_id]
        successful_calls = [row["calls"][0]["ms"] for row in selected if "result" in row]
        metrics[provider] = {
            **summarize(selected),
            "errors_by_type": dict(Counter(row["error"] for row in selected if "error" in row)),
            "reported_models": dict(Counter(
                call["model"] for row in selected for call in row["calls"] if "model" in call
            )),
            "llm_p50_ms": percentile(successful_calls, 0.5),
            "llm_p95_ms": percentile(successful_calls, 0.95),
            "usage": usage_totals(row["usage"] for row in selected),
            "passed_by_locale": {
                locale: sum(row["passed"] for row in selected if row["locale"] == locale)
                for locale in ("ru", "en")
            },
        }
    discordant = {
        provider: [
            case_id for case_id in case_by_id
            if rows[case_id, provider]["passed"] and not rows[case_id, other]["passed"]
        ]
        for provider, other in (providers, providers[::-1])
    }
    write_json(
        OUT / "providers-scored.json",
        {"configuration": provenance, "metrics": metrics, "discordant": discordant, "results": list(rows.values())},
    )
    print(json.dumps({"metrics": metrics, "discordant": discordant}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("A", "B"), default="A")
    args = parser.parse_args()
    load_env()
    with (OUT / ".providers-run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Provider comparison already running in {OUT}") from None
        asyncio.run(compare(args.variant))


if __name__ == "__main__":
    main()
