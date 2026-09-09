import json
import tempfile
import unittest
from pathlib import Path

from path_consistency.math_eval import (
    extract_gsm8k_numeric,
    extract_last_boxed,
    format_gsm8k_gold,
    math_equal,
)
from run_math import load_eval_records


class Gsm8kDatasetTest(unittest.TestCase):
    def test_gold_canonicalization(self):
        self.assertEqual(format_gsm8k_gold(18.0), "18")
        self.assertEqual(format_gsm8k_gold(3), "3")
        self.assertEqual(format_gsm8k_gold("18.0"), "18")

    def test_numeric_grader(self):
        self.assertTrue(math_equal("18", "18.0"))
        self.assertTrue(math_equal("18", format_gsm8k_gold(18.0)))

    def test_extract_prefers_boxed_then_answer_is(self):
        self.assertEqual(extract_last_boxed(r"work \boxed{18}"), "18")
        self.assertEqual(
            extract_gsm8k_numeric("3 + 5 = 8. The answer is 18."), "18"
        )

    def test_loader_and_round_robin_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gsm8k.jsonl"
            rows = [
                {"question": f"q{index}", "target": float(index)}
                for index in range(8)
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            all_rows, task = load_eval_records(path, None, 2027)
            self.assertEqual(task, "gsm8k")
            self.assertEqual(len(all_rows), 8)
            self.assertEqual(all_rows[0]["question_id"], "gsm8k_0000")
            self.assertEqual(all_rows[0]["answer"], "0")
            shard0, _ = load_eval_records(
                path, None, 2027, num_shards=4, shard_index=0
            )
            shard1, _ = load_eval_records(
                path, None, 2027, num_shards=4, shard_index=1
            )
            self.assertEqual([row["source_row"] for row in shard0], [0, 4])
            self.assertEqual([row["source_row"] for row in shard1], [1, 5])


if __name__ == "__main__":
    unittest.main()
