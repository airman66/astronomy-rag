import json
import unittest

from app.assistant.prompts import PROMPT_VERSION
from app.infra.config import ROOT


class AssistantEvaluationReportTests(unittest.TestCase):
    def test_report_matches_current_prompt_and_questions(self):
        directory = ROOT / "data/eval/assistant"
        questions = json.loads((directory / "questions.json").read_text(encoding="utf-8"))["questions"]
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["prompt_version"], PROMPT_VERSION)
        self.assertEqual(report["question_count"], len(questions))
        results = [{key: result[key] for key in questions[0]} for result in report["results"]]
        self.assertEqual(results, questions)
