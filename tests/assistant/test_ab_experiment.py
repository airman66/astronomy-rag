import hashlib
import json
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from app.assistant.client import Completion
from app.assistant.service import ModelReply
from app.infra.config import LLMSettings
from app.infra.files import write_json
from experiments.rag_ab import live, report
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

    async def test_resume_recovers_a_final_write(self) -> None:
        checkpoint = temporary_dir(self) / "responses.jsonl"
        first = b'{"id":"a"}'
        second = {"id": "b"}
        for content in (first, first + b'\n{"id":"b","answer":"\xd0'):
            with self.subTest(content=content):
                checkpoint.write_bytes(content)
                worker = AsyncMock(return_value=second)

                rows = await run_batches(["a", "b"], worker, checkpoint, lambda row: row["id"], 2)

                worker.assert_awaited_once_with("b")
                self.assertEqual(list(rows), ["a", "b"])
                self.assertEqual([json.loads(line) for line in checkpoint.read_bytes().splitlines()], list(rows.values()))

    async def test_resume_rejects_corrupt_complete_lines(self) -> None:
        checkpoint = temporary_dir(self) / "grades.jsonl"
        content = b'{"id":"a"}\n{"id":\n'
        checkpoint.write_bytes(content)
        worker = AsyncMock()

        with self.assertRaises(json.JSONDecodeError):
            await run_batches(["a", "b"], worker, checkpoint, lambda row: row["id"], 2)

        self.assertEqual(checkpoint.read_bytes(), content)
        worker.assert_not_awaited()

    async def test_phase_rejects_stale_responses_and_grades(self) -> None:
        root = temporary_dir(self)
        self.enterContext(patch.multiple(live, ROOT=root, OUT=root))
        self.enterContext(patch.object(live.storage, "corpus_version", return_value="corpus"))
        case = {"id": "a", "question": "First question?", "locale": "en", "expected_modes": ["general"]}
        suite_path = root / "questions.json"
        snapshot_path = root / "retrieval.json"
        schedule_path = root / "repeat-cases.json"
        write_json(suite_path, {"questions": [case]})
        snapshot = {"evidence": [], "ranks": [], "retrieval_ms": 1.0, "chars": 0}
        retrieval = {
            "index": {"corpus_sha256": "corpus"},
            "index_manifest_sha256": "same-index",
            "suite_sha256": {"new": hashlib.sha256(suite_path.read_bytes()).hexdigest()},
            "results": [{"id": "a", "configs": {"8x11000": snapshot}}],
        }
        write_json(snapshot_path, retrieval)
        write_json(schedule_path, {"a": ["A"]})
        trial_mock = self.enterContext(patch.object(live, "trial", return_value={
            "id": "a", "variant": "A", "result": {
                "answer": "First answer", "mode": "general", "coverage": "complete", "grounding": "model",
                "sources": [], "images": [], "blocks": [], "primary_url": None,
            },
        }))
        label = live.blind_label("a", "A", "repeat")
        grade_mock = self.enterContext(patch.object(live, "grade", return_value={
            label: {"request_met": True, "correct": True, "grounded": True},
        }))
        settings = LLMSettings("deepseek", "deepseek-flash", "unused")

        await live.run_phase("repeat", "8x11000", 1, settings)
        await live.run_phase("repeat", "8x11000", 1, settings)

        for path, data in (
            (suite_path, {"questions": [{**case, "question": "Updated question?"}]}),
            (snapshot_path, {**retrieval, "results": [{"id": "a", "configs": {"8x11000": {**snapshot, "chars": 1}}}]}),
            (schedule_path, {"a": ["B"]}),
        ):
            with self.subTest(file=path.name):
                original = path.read_bytes()
                write_json(path, data)
                with self.assertRaisesRegex(SystemExit, "snapshot suite differs|Checkpoint inputs differ"):
                    await live.run_phase("repeat", "8x11000", 1, settings)
                path.write_bytes(original)

        (root / "repeat-run.json").unlink()
        with self.assertRaisesRegex(SystemExit, "Checkpoint manifest missing"):
            await live.run_phase("repeat", "8x11000", 1, settings)
        trial_mock.assert_awaited_once()
        grade_mock.assert_awaited_once()

    def test_report_uses_recorded_context_for_passage_fate(self) -> None:
        root = temporary_dir(self)
        self.enterContext(patch.multiple(report, ROOT=root, OLD=root, OUT=root))
        case = {"id": "a", "question": "Question?", "locale": "en"}
        write_json(root / "questions.json", {"questions": [case]})
        for name in ("oae_passage_questions", "full_questions"):
            write_json(root / f"{name}.json", {"questions": []})
        write_json(root / "manual-review.json", {})
        write_json(root / "legacy-live.json", {"results": []})
        write_json(root / "retrieval.json", {"metrics": {}, "results": [{
            "id": "a", "candidates": {"lexical_sections": [1], "semantic_40": [1]},
            "configs": {"8x8500": {"ranks": [None]}, "8x11000": {"ranks": [7]}},
        }]})
        row = {
            "id": "a", "label": "blind", "variant": "A", "usage": {}, "latency_ms": 1, "calls": [],
            "result": {"mode": "explain", "coverage": "complete", "answer": "Answer", "sources": []},
            "passed": True, "quality": True, "mode_correct": True, "coverage_correct": True, "source_valid": True,
            "grade": {"used_facts": [False]},
        }
        for phase in report.PHASES:
            context = {"packing": "factorial", "llm": "8x11000"}.get(phase, "8x8500")
            write_json(root / f"{phase}-run.json", {"phase": phase, "context": context})
            write_json(root / f"{phase}-scored.json", [row] if phase in {"llm", "repeat"} else [])

        report.build(include_repeats=True)

        phases = report.read("per-question.json")["new"]["a"]["phases"]
        self.assertEqual(phases["llm"][0]["parts"], ["передан, в ответе не подтверждён"])
        self.assertEqual(phases["repeat"][0]["parts"], ["найден и отсеян"])
