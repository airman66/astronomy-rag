"""Run the 50-question retrieval and DeepSeek mode regression set."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from app.assistant import DeepSeekClient, SELECT_PROMPT, PROMPT_VERSION, answer, candidates_for
from app.storage import DATA


async def evaluate_model(question: dict, candidates: list[dict], client: DeepSeekClient, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        if not candidates:
            decision = {"mode": "general", "record_ids": []}
        else:
            decision = await client.complete(SELECT_PROMPT, {"question": question["question"], "locale": question["locale"], "context": {}, "candidates": candidates})
        selected = [rid for rid in decision.get("record_ids", []) if rid in {item["record_id"] for item in candidates}]
        actual = decision.get("mode", "general")
        if actual == "article" and not any(item["kind"] == "article" and item["record_id"] in selected for item in candidates):
            actual = "explain" if selected else "general"
        if actual in ("term", "explain") and not selected:
            actual = "general"
        return {"mode": actual, "selected_ids": selected, "raw_mode": decision.get("mode"), "invalid_selected_ids": [rid for rid in decision.get("record_ids", []) if rid not in {item["record_id"] for item in candidates}]}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=int, default=10, help="Number of full answers to inspect in addition to 50 mode choices")
    parser.add_argument("--retrieval-only", action="store_true", help="Recall@5 and MRR without DeepSeek; writes data/eval/retrieval_report.json")
    args = parser.parse_args()
    rows = json.loads((DATA / "eval" / "questions.json").read_text(encoding="utf-8"))["questions"]
    assert len(rows) >= 50
    if args.retrieval_only:
        from app import storage

        results = []
        for row in rows:
            ids = [candidate["record_id"] for candidate in candidates_for(row["question"], row["locale"])]
            gold = set(row["expected_record_ids"])
            ranks = [ids.index(item) + 1 for item in gold if item in ids]
            results.append({"id": row["id"], "question": row["question"], "expected": sorted(gold), "candidate_ids": ids, "recall_at_5": bool(gold and any(rank <= 5 for rank in ranks)), "mrr": (1 / min(ranks)) if ranks else 0})
        with_gold = [row for row in results if row["expected"]]
        _, mode = storage.search_with_mode(rows[0]["question"], rows[0]["locale"], 5)
        report = {"timestamp": datetime.now(timezone.utc).isoformat(), "search_mode": mode, "question_count": len(results),
                  "retrieval_recall_at_5": sum(row["recall_at_5"] for row in with_gold) / len(with_gold),
                  "retrieval_mrr": sum(row["mrr"] for row in with_gold) / len(with_gold), "results": results}
        (DATA / "eval" / "retrieval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False))
        print("misses:", [(row["id"], row["expected"], row["candidate_ids"][:5]) for row in with_gold if not row["recall_at_5"]])
        return
    client = DeepSeekClient()
    semaphore = asyncio.Semaphore(4)
    candidates = [candidates_for(row["question"], row["locale"]) for row in rows]
    decisions = await asyncio.gather(*(evaluate_model(row, choice, client, semaphore) for row, choice in zip(rows, candidates)))
    checked = []
    for row, choice, decision in zip(rows, candidates, decisions):
        ids = [candidate["record_id"] for candidate in choice]
        gold = set(row["expected_record_ids"])
        ranks = [ids.index(item) + 1 for item in gold if item in ids]
        checked.append({**row, "candidate_ids": ids, "retrieval_recall_at_5": bool(gold and any(rank <= 5 for rank in ranks)), "retrieval_mrr": (1 / min(ranks)) if ranks else 0, "mode": decision["mode"], "mode_correct": decision["mode"] == row["expected_mode"], "selected_ids": decision["selected_ids"], "invalid_selected_ids": decision["invalid_selected_ids"]})
    for index in range(min(args.answers, len(rows))):
        try:
            result = await answer(rows[index]["question"], "auto", rows[index]["locale"], None, client)
            checked[index]["answer"] = result["answer"]
            checked[index]["sources"] = result["sources"]
            checked[index]["source_ids_valid"] = all(source["record_id"] in checked[index]["candidate_ids"] for source in result["sources"])
        except Exception as exc:
            checked[index]["answer_error"] = type(exc).__name__
    with_gold = [row for row in checked if row["expected_record_ids"]]
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "model": "deepseek-flash",
        "index_manifest": json.loads((DATA / "index" / "index-manifest.json").read_text()),
        "question_count": len(checked),
        "mode_accuracy": sum(row["mode_correct"] for row in checked) / len(checked),
        "retrieval_recall_at_5": sum(row["retrieval_recall_at_5"] for row in with_gold) / len(with_gold),
        "retrieval_mrr": sum(row["retrieval_mrr"] for row in with_gold) / len(with_gold),
        "invalid_selected_count": sum(len(row["invalid_selected_ids"]) for row in checked),
        "invalid_answer_source_count": sum(row.get("source_ids_valid") is False for row in checked),
        "results": checked,
    }
    destination = DATA / "eval" / "report.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("results", "index_manifest")}, ensure_ascii=False))
    print("mode errors:", [(row["id"], row["expected_mode"], row["mode"]) for row in checked if not row["mode_correct"]])


if __name__ == "__main__":
    asyncio.run(main())
