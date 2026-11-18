"""Run the 50-question retrieval and DeepSeek mode regression set."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone

from app.assistant.client import LLMClient
from app.assistant.prompts import PROMPT_VERSION
from app.assistant.retrieval import candidates_for
from app.assistant.service import answer, select_context
from app.catalog.search import search_with_mode
from app.infra.config import LLMSettings, data_directory
from app.infra.files import existing_path, write_json

DATA = data_directory()
EVALUATION = DATA / "eval" / "assistant"


def score_candidates(candidates: list[dict], expected: list[str]) -> tuple[list[str], bool, float]:
    ids = [candidate["record_id"] for candidate in candidates]
    ranks = [ids.index(identifier) + 1 for identifier in expected if identifier in ids]
    best = min(ranks, default=0)
    return ids, 0 < best <= 5, 1 / best if best else 0


async def evaluate_model(
    question: dict, candidates: list[dict], client: LLMClient, semaphore: asyncio.Semaphore
) -> dict:
    async with semaphore:
        selection = await select_context(
            question["question"], "auto", question["locale"], None, candidates, client
        )
        return {
            "mode": selection.mode,
            "selected_ids": [item["record_id"] for item in selection.sources],
            "raw_mode": selection.suggested_mode,
            "invalid_selected_ids": selection.invalid_ids,
        }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--answers",
        type=int,
        default=10,
        help="Number of full answers to inspect in addition to 50 mode choices",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Recall@5 and MRR without DeepSeek; writes data/eval/assistant/retrieval_report.json",
    )
    args = parser.parse_args()
    questions = existing_path(EVALUATION / "questions.json", DATA / "eval" / "questions.json")
    rows = json.loads(questions.read_text(encoding="utf-8"))["questions"]
    assert len(rows) >= 50
    candidates = [candidates_for(row["question"], row["locale"]) for row in rows]
    if args.retrieval_only:
        results = []
        for row, choice in zip(rows, candidates):
            ids, recall, reciprocal_rank = score_candidates(choice, row["expected_record_ids"])
            results.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "expected": sorted(set(row["expected_record_ids"])),
                    "candidate_ids": ids,
                    "recall_at_5": recall,
                    "mrr": reciprocal_rank,
                }
            )
        with_gold = [row for row in results if row["expected"]]
        _, mode = search_with_mode(rows[0]["question"], rows[0]["locale"], 5)
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "search_mode": mode,
            "question_count": len(results),
            "retrieval_recall_at_5": sum(row["recall_at_5"] for row in with_gold) / len(with_gold),
            "retrieval_mrr": sum(row["mrr"] for row in with_gold) / len(with_gold),
            "results": results,
        }
        write_json(EVALUATION / "retrieval_report.json", report)
        print(
            json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False)
        )
        print(
            "misses:",
            [
                (row["id"], row["expected"], row["candidate_ids"][:5])
                for row in with_gold
                if not row["recall_at_5"]
            ],
        )
        return
    client = LLMClient()
    semaphore = asyncio.Semaphore(4)
    decisions = await asyncio.gather(
        *(evaluate_model(row, choice, client, semaphore) for row, choice in zip(rows, candidates))
    )
    checked = []
    for row, choice, decision in zip(rows, candidates, decisions):
        ids, recall, reciprocal_rank = score_candidates(choice, row["expected_record_ids"])
        checked.append(
            {
                **row,
                "candidate_ids": ids,
                "retrieval_recall_at_5": recall,
                "retrieval_mrr": reciprocal_rank,
                "mode": decision["mode"],
                "mode_correct": decision["mode"] == row["expected_mode"],
                "selected_ids": decision["selected_ids"],
                "invalid_selected_ids": decision["invalid_selected_ids"],
            }
        )
    for index in range(min(args.answers, len(rows))):
        try:
            result = await answer(rows[index]["question"], "auto", rows[index]["locale"], None, client)
            checked[index]["answer"] = result["answer"]
            checked[index]["sources"] = result["sources"]
            checked[index]["source_ids_valid"] = all(
                source["record_id"] in checked[index]["candidate_ids"] for source in result["sources"]
            )
        except Exception as exc:
            checked[index]["answer_error"] = type(exc).__name__
    with_gold = [row for row in checked if row["expected_record_ids"]]
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "model": LLMSettings.from_env().model,
        "index_manifest": json.loads((DATA / "index" / "index-manifest.json").read_text()),
        "question_count": len(checked),
        "mode_accuracy": sum(row["mode_correct"] for row in checked) / len(checked),
        "retrieval_recall_at_5": sum(row["retrieval_recall_at_5"] for row in with_gold) / len(with_gold),
        "retrieval_mrr": sum(row["retrieval_mrr"] for row in with_gold) / len(with_gold),
        "invalid_selected_count": sum(len(row["invalid_selected_ids"]) for row in checked),
        "invalid_answer_source_count": sum(row.get("source_ids_valid") is False for row in checked),
        "results": checked,
    }
    write_json(EVALUATION / "report.json", report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in ("results", "index_manifest")},
            ensure_ascii=False,
        )
    )
    print(
        "mode errors:",
        [(row["id"], row["expected_mode"], row["mode"]) for row in checked if not row["mode_correct"]],
    )
