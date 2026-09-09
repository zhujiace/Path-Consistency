import unittest

from path_consistency.confidence import beta_confidence, confidence_criteria
from path_consistency.deepconf_offline import (
    aggregate_deepconf_offline,
    token_confidence,
)
from path_consistency.math_eval import extract_last_boxed, math_equal, normalize_answer
from path_consistency.math_inference import MathInference, lower_tail_local_score


class MathProtocolTest(unittest.TestCase):
    def test_nested_last_boxed(self):
        text = r"first \boxed{1}, final \boxed{\frac{1+\sqrt{5}}{2}}"
        self.assertEqual(extract_last_boxed(text), r"\frac{1+\sqrt{5}}{2}")

    def test_incomplete_last_box_falls_back(self):
        text = r"valid \boxed{7}; incomplete \boxed{8"
        self.assertEqual(extract_last_boxed(text), "7")

    def test_normalization(self):
        self.assertEqual(normalize_answer(r" $\left( 1, 2 \right).$ "), "(1,2)")

    def test_symbolic_grader(self):
        self.assertTrue(math_equal(r"\frac{1}{2}", "0.5"))
        self.assertFalse(math_equal("2", "3"))

    def test_beta_uses_all_alternatives(self):
        # 3 modal votes vs. 2 total alternatives -> Beta(4,3), not Beta(4,2).
        self.assertAlmostEqual(beta_confidence(["a", "a", "a", "b", "c"]), 0.65625)
        self.assertEqual(beta_confidence([]), 0.0)
        self.assertIsNone(confidence_criteria([], 0.8)["most_common"])

    def test_lower_tail_score(self):
        values = [-0.1] * 128 + [-2.0] * 128
        self.assertAlmostEqual(lower_tail_local_score(values, 0, 256), -2.0)

    def test_deepconf_retention(self):
        traces = [
            {"answer": "a", "branch_id": "0", "lgc_2048": 1.0},
            {"answer": "b", "branch_id": "1", "lgc_2048": 4.0},
            {"answer": "b", "branch_id": "2", "lgc_2048": 3.0},
            {"answer": "a", "branch_id": "3", "lgc_2048": 2.0},
        ]
        result = aggregate_deepconf_offline(traces)
        self.assertEqual(result["offline_lgc_10"]["answer"], "b")
        self.assertEqual(result["offline_lgc_10"]["retained"], 1)
        self.assertEqual(result["offline_lgc_90"]["retained"], 3)

    def test_deepconf_uses_exact_top20(self):
        values = {index: -1.0 for index in range(20)}
        values[20] = -100.0
        self.assertEqual(token_confidence(values), 1.0)

    def test_pc_reuses_identical_sc_root_windows(self):
        engine = MathInference.__new__(MathInference)
        engine.calls = 0
        engine.build_prompt = lambda problem: (problem, [99])

        def generate(*args, **kwargs):
            engine.calls += 1
            traces = []
            for branch in range(4):
                answer = f"answer-{engine.calls}-{branch}"
                traces.append(
                    {
                        "full_token_ids": [branch + 1],
                        "new_token_ids": [branch + 1],
                        "sampled_logprobs": [-0.1],
                        "token_confidence": [1.0],
                        "full_text": answer,
                        "suffix_text": answer,
                        "raw_answer": answer,
                        "answer": answer,
                        "reasoning_tokens": 1,
                        "finish_reason": "stop",
                        "stop_reason": None,
                        "generated_tokens": 1,
                        "prompt_tokens": 1,
                        "prefix_tokens": 0,
                        "truncated": False,
                        "lgc_2048": 1.0,
                    }
                )
            return traces, float(engine.calls)

        engine._generate_window = generate
        sc = engine.run_question(
            "qid",
            "problem",
            "sc",
            max_branches=8,
            window_size=4,
            beta_threshold=1.0,
            return_internal_windows=True,
        )
        cached = sc.pop("_internal_root_windows")
        self.assertEqual(engine.calls, 2)

        engine.calls = 0
        pc = engine.run_question(
            "qid",
            "problem",
            "pc",
            max_branches=8,
            window_size=4,
            beta_threshold=1.0,
            paired_root_windows=cached,
        )
        self.assertEqual(engine.calls, 0)
        self.assertEqual(pc["reused_root_windows"], 2)
        self.assertEqual(pc["actual_generation_seconds"], 0.0)
        self.assertEqual(pc["generation_seconds"], sc["generation_seconds"])
        self.assertEqual(
            [trace["answer"] for trace in pc["traces"]],
            [trace["answer"] for trace in sc["traces"]],
        )


if __name__ == "__main__":
    unittest.main()
