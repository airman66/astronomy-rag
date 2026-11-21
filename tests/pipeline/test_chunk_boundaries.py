"""Chunk sizing must preserve science blocks and bound long prose."""

import unittest
from unittest.mock import patch

from app.pipeline import chunks


class ChunkBoundaryTests(unittest.TestCase):
    def setUp(self):
        # Deterministic stand-in avoids downloading an external tokenizer.
        self.enterContext(patch.object(chunks, "token_count", side_effect=lambda text: len(text.split())))

    def test_long_unbroken_prose_is_split_within_token_budget(self):
        source = " ".join(f"word{i}" for i in range(65))
        result = chunks.paragraphs(source, budget=12, overlap=0.1)
        self.assertGreater(len(result), 3)
        self.assertTrue(all(chunks.token_count(part) <= 12 for part in result), result)
        self.assertEqual(" ".join(result).split(), source.split())

    def test_structured_formula_is_not_sliced_into_invalid_fragments(self):
        formula = "\\begin{align} E = mc^2 \\end{align} " + ("\u03b1 " * 20)
        self.assertEqual(chunks.paragraphs(formula, budget=5, overlap=0), [formula.strip()])

    def test_invalid_configuration_rejected(self):
        for budget, overlap in ((0, 0.1), (10, -0.1), (10, 1)):
            with self.assertRaises(ValueError):
                chunks.paragraphs("A simple sentence.", budget, overlap)

    def test_overlap_does_not_push_a_short_new_group_over_budget(self):
        body = "a b c d e f g h\n\ni j k l m n o p"
        result = chunks.paragraphs(body, budget=10, overlap=0.9)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(chunks.token_count(x) <= 10 for x in result))

    def test_wrapped_ocr_prose_is_split_and_sentences_preferred(self):
        body = "One two three four.\nFive six seven eight. Nine ten eleven twelve."
        result = chunks.paragraphs(body, budget=5, overlap=0)
        self.assertGreater(len(result), 1)
        self.assertTrue(all(chunks.token_count(fragment) <= 5 for fragment in result))
        self.assertEqual([part.strip().endswith(".") for part in result], [True] * len(result))

    def test_split_sentence_strips_each_piece(self):
        result = chunks._split_prose("  One two. Three four five six.", budget=3)
        self.assertEqual(result, ["One two.", "Three four five", "six."])
