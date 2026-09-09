import json
import tempfile
import unittest
from pathlib import Path

from path_consistency.math_eval import (
    aime_equal,
    canonicalize_vote_answer,
    extract_aime_integer,
    extract_gpqa_letter,
    gpqa_equal,
    grade_answer,
)
from path_consistency.math_protocol import format_gpqa_stem, instruction_for_task
from run_math import load_eval_records


class AimeGpqaEvalTest(unittest.TestCase):
    def test_aime_integer_canonical(self):
        self.assertEqual(extract_aime_integer("042"), "42")
        self.assertEqual(extract_aime_integer("0"), "0")
        self.assertEqual(extract_aime_integer(r"\boxed{117}"), "117")
        self.assertIsNone(extract_aime_integer("1000"))
        self.assertTrue(aime_equal("042", "42"))
        self.assertFalse(aime_equal("16", "17"))

    def test_gpqa_letter_canonical(self):
        self.assertEqual(extract_gpqa_letter("B"), "B")
        self.assertEqual(extract_gpqa_letter("(A)"), "A")
        self.assertEqual(extract_gpqa_letter("The answer is C."), "C")
        self.assertEqual(extract_gpqa_letter("d"), "D")
        self.assertIsNone(extract_gpqa_letter("maybe A or B"))
        self.assertTrue(gpqa_equal("(B)", "B"))
        self.assertFalse(gpqa_equal("A", "C"))

    def test_vote_keys_and_grader(self):
        self.assertEqual(canonicalize_vote_answer("aime", "070"), "70")
        self.assertEqual(canonicalize_vote_answer("gpqa", "(D)"), "D")
        self.assertTrue(grade_answer("aime", "7", "007"))
        self.assertTrue(grade_answer("gpqa", "A", "A"))

    def test_gpqa_prompt_shape(self):
        stem = format_gpqa_stem("What is 2+2?", {"A": "3", "B": "4", "C": "5", "D": "6"})
        self.assertIn("Question:", stem)
        self.assertIn("(B) 4", stem)
        self.assertIn("(A, B, C, or D)", instruction_for_task("gpqa"))

    def test_loaders(self):
        with tempfile.TemporaryDirectory() as tmp:
            aime_path = Path(tmp) / "aime.jsonl"
            aime_path.write_text(
                json.dumps(
                    {
                        "task": "aime",
                        "dataset": "aime_2024",
                        "year": 2024,
                        "unique_id": "aime_2024/00",
                        "problem": "Compute 1+1.",
                        "answer": "2",
                    }
                )
                + "\n"
            )
            rows, task = load_eval_records(aime_path, None, 2027)
            self.assertEqual(task, "aime")
            self.assertEqual(rows[0]["answer"], "2")
            self.assertEqual(rows[0]["year"], 2024)

            gpqa_path = Path(tmp) / "gpqa.jsonl"
            gpqa_path.write_text(
                json.dumps(
                    {
                        "task": "gpqa",
                        "dataset": "gpqa_diamond",
                        "unique_id": "recTEST",
                        "question": "Pick the even number.",
                        "choices": {"A": "1", "B": "2", "C": "3", "D": "5"},
                        "answer": "B",
                    }
                )
                + "\n"
            )
            rows, task = load_eval_records(gpqa_path, None, 2027)
            self.assertEqual(task, "gpqa")
            self.assertEqual(rows[0]["answer"], "B")
            self.assertIn("(B) 2", rows[0]["problem"])
            self.assertIn("Please reason step by step", instruction_for_task("gpqa"))


if __name__ == "__main__":
    unittest.main()
