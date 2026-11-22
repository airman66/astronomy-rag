import json
import unittest
from unittest.mock import AsyncMock

from pydantic import ValidationError

from app.assistant.client import Completion
from app.assistant.service import ModelReply
from experiments.rag_ab.live import ExperimentalReply, run_batches, trial
from experiments.rag_ab.prompts import REVISED
from experiments.rag_ab.retrieval import PackedContext
from tests.support.catalog import temporary_dir


class ExperimentTests(unittest.IsolatedAsyncioTestCase):
    def test_experimental_reply_accepts_eight_passages(self) -> None:
        reply = {
            "mode": "explain",
            "coverage": "complete",
            "definition_id": None,
            "blocks": [{"text": "Answer", "evidence_ids": [f"e{i}" for i in range(1, 9)]}],
        }
        with self.assertRaises(ValidationError):
            ModelReply.model_validate(reply)
        self.assertEqual(ExperimentalReply.model_validate(reply).blocks[0].evidence_ids[-1], "e8")

    async def test_invalid_plan_uses_the_single_call_prompt(self) -> None:
        case = {"id": "planet", "question": "Why do planets orbit stars?", "locale": "en"}
        snapshot: PackedContext = {"evidence": [], "ranks": [], "retrieval_ms": 0.0, "chars": 0}
        plan = {
            "intent": "explanation",
            "source_requirement": "none",
            "source_phrase": None,
            "fallback": "default",
            "parts": [{"request_span": case["question"], "evidence_ids": ["e1"]}],
        }
        response = {
            "mode": "general",
            "coverage": "complete",
            "definition_id": None,
            "blocks": [{"text": "Gravity keeps planets in orbit.", "evidence_ids": []}],
        }
        planner = AsyncMock()
        planner.complete.return_value = Completion(plan, "planner", {"prompt_tokens": 10})
        writer = AsyncMock()
        writer.complete.return_value = Completion(response, "writer", {"prompt_tokens": 20})

        result = await trial(case, snapshot, "C", writer, planner, version="C")

        self.assertEqual(result["plan_error"], "unknown evidence ID in plan")
        self.assertEqual(result["result"]["answer"], "Gravity keeps planets in orbit.")
        self.assertEqual(result["usage"], {"prompt_tokens": 30})
        system, payload = writer.complete.call_args.args
        self.assertEqual(system, REVISED)
        self.assertNotIn("validated_plan", payload)

    async def test_resume_only_runs_missing_keys(self) -> None:
        checkpoint = temporary_dir(self) / "responses.jsonl"
        first = {"id": "a", "answer": "saved"}
        second = {"id": "b", "answer": "new"}
        checkpoint.write_text(json.dumps(first) + "\n", encoding="utf-8")
        worker = AsyncMock(return_value=second)

        rows = await run_batches(["a", "b"], worker, checkpoint, lambda row: row["id"], 2)

        worker.assert_awaited_once_with("b")
        self.assertEqual(rows, {"a": first, "b": second})
        self.assertEqual([json.loads(line) for line in checkpoint.read_text().splitlines()], [first, second])
