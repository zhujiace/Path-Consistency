import unittest

from path_consistency.confidence import beta_confidence, confidence_criteria
from path_consistency.deepconf_offline import (
    aggregate_deepconf_offline,
    token_confidence,
)
from path_consistency.generation_methods import (
    ESC_STOP_TWO_WINDOW_OR_8,
    parse_generation_methods,
    should_early_stop,
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

    def test_parse_methods_orders_roots_first(self):
        self.assertEqual(parse_generation_methods("pc_ac,ac"), ["ac", "pc_ac"])
        self.assertEqual(parse_generation_methods("esc+pc,esc"), ["esc", "pc_esc"])
        self.assertEqual(parse_generation_methods("pc,sc"), ["sc", "pc"])

    def test_esc_requires_a_fully_unanimous_window(self):
        self.assertTrue(should_early_stop("esc", ["7", "7", "7", "7"], ["7"] * 4))
        self.assertFalse(should_early_stop("esc", ["7", "7", "7", None], ["7"] * 3))
        self.assertFalse(should_early_stop("esc", ["7", "7", "8", "7"], ["7", "7", "8", "7"]))

    def test_esc_delayed_rule_needs_two_windows_or_eight(self):
        self.assertFalse(
            should_early_stop(
                "esc",
                ["7"] * 4,
                ["7"] * 4,
                esc_stop_rule=ESC_STOP_TWO_WINDOW_OR_8,
                window_size=4,
            )
        )
        self.assertTrue(
            should_early_stop(
                "esc",
                ["7"] * 4,
                ["7"] * 8,
                esc_stop_rule=ESC_STOP_TWO_WINDOW_OR_8,
                window_size=4,
            )
        )
        self.assertFalse(
            should_early_stop(
                "esc",
                ["8"] * 4,
                ["7"] * 4 + ["8"] * 4,
                esc_stop_rule=ESC_STOP_TWO_WINDOW_OR_8,
                window_size=4,
            )
        )
        self.assertTrue(
            should_early_stop(
                "esc",
                ["8"] * 4,
                ["7"] * 4 + ["8"] * 8,
                esc_stop_rule=ESC_STOP_TWO_WINDOW_OR_8,
                window_size=4,
            )
        )

    def test_ac_tau099_does_not_stop_on_four_unanimous(self):
        engine = _dummy_engine(lambda call, branch: "7")
        ac = engine.run_question(
            "qid",
            "problem",
            "ac",
            max_branches=8,
            window_size=4,
            early_stop_threshold=0.99,
        )
        self.assertTrue(ac["early_stopped"])
        self.assertEqual(ac["launched_windows"], 2)
        self.assertEqual(len(ac["traces"]), 8)

        engine = _dummy_engine(lambda call, branch: "7")
        ac95 = engine.run_question(
            "qid",
            "problem",
            "ac",
            max_branches=8,
            window_size=4,
            early_stop_threshold=0.95,
        )
        self.assertEqual(ac95["launched_windows"], 1)
        self.assertEqual(len(ac95["traces"]), 4)

    def test_esc_delayed_does_not_stop_after_one_window(self):
        engine = _dummy_engine(lambda call, branch: "7")
        esc = engine.run_question(
            "qid",
            "problem",
            "esc",
            max_branches=8,
            window_size=4,
            esc_stop_rule=ESC_STOP_TWO_WINDOW_OR_8,
        )
        self.assertTrue(esc["early_stopped"])
        self.assertEqual(esc["launched_windows"], 2)
        self.assertEqual(len(esc["traces"]), 8)

    def test_ac_stops_on_unanimous_window_and_esc_matches(self):
        engine = _dummy_engine(lambda call, branch: "7")
        ac = engine.run_question(
            "qid",
            "problem",
            "ac",
            max_branches=8,
            window_size=4,
            beta_threshold=0.8,
            early_stop_threshold=0.95,
        )
        self.assertTrue(ac["early_stopped"])
        self.assertEqual(ac["launched_windows"], 1)
        self.assertEqual(len(ac["traces"]), 4)
        self.assertEqual(engine.calls, 1)

        engine.calls = 0
        esc = engine.run_question(
            "qid",
            "problem",
            "esc",
            max_branches=8,
            window_size=4,
        )
        self.assertTrue(esc["early_stopped"])
        self.assertEqual(len(esc["traces"]), 4)

    def test_ac_keeps_sampling_when_confidence_is_low(self):
        engine = _dummy_engine(
            lambda call, branch: "a" if (call + branch) % 2 == 0 else "b"
        )
        ac = engine.run_question(
            "qid",
            "problem",
            "ac",
            max_branches=8,
            window_size=4,
            early_stop_threshold=0.95,
        )
        self.assertFalse(ac["early_stopped"])
        self.assertEqual(len(ac["traces"]), 8)
        self.assertEqual(engine.calls, 2)

    def test_pc_ac_reuses_ac_root_windows(self):
        engine = _dummy_engine(lambda call, branch: f"ans-{call}-{branch}")
        engine._select_prefix = lambda *args, **kwargs: None
        ac = engine.run_question(
            "qid",
            "problem",
            "ac",
            max_branches=8,
            window_size=4,
            early_stop_threshold=0.99,
            return_internal_windows=True,
        )
        cached = ac.pop("_internal_root_windows")
        self.assertEqual(engine.calls, 2)
        engine.calls = 0
        combo = engine.run_question(
            "qid",
            "problem",
            "pc_ac",
            max_branches=8,
            window_size=4,
            early_stop_threshold=0.99,
            paired_root_windows=cached,
        )
        self.assertEqual(engine.calls, 0)
        self.assertEqual(combo["reused_root_windows"], 2)
        self.assertEqual(
            [trace["answer"] for trace in combo["traces"]],
            [trace["answer"] for trace in ac["traces"]],
        )

    def test_recovery_off_keeps_failed_trial_prefix(self):
        proposed = {
            "source_id": "dummy",
            "token_ids": ["PREFIX"],
            "endpoint": 1,
            "source_reasoning_tokens": 1,
            "actual_ratio": 0.2,
            "blc_score": 0.0,
            "source_generated_tokens": 1,
            "selector": "blc",
        }

        def answers(call, branch):
            return "7" if call == 1 else "8"

        engine = _dummy_engine(answers)
        engine._select_prefix = lambda *args, **kwargs: dict(proposed)
        kept = engine.run_question(
            "qid",
            "problem",
            "pc",
            max_branches=12,
            window_size=4,
            beta_threshold=0.8,
            recovery="off",
        )
        self.assertEqual(
            [event["used_prefix_tokens"] for event in kept["windows"]],
            [0, 1, 1],
        )
        self.assertEqual(kept["windows"][1]["decision"], "retain_failed_trial")
        self.assertFalse(kept["prefixing_disabled"])

        engine = _dummy_engine(answers)
        engine._select_prefix = lambda *args, **kwargs: dict(proposed)
        reverted = engine.run_question(
            "qid",
            "problem",
            "pc",
            max_branches=12,
            window_size=4,
            beta_threshold=0.8,
            recovery="on",
        )
        self.assertEqual(reverted["windows"][1]["decision"], "rollback_trial")
        self.assertEqual(reverted["windows"][2]["used_prefix_tokens"], 0)


def _dummy_engine(answer_fn):
    engine = MathInference.__new__(MathInference)
    engine.calls = 0
    engine.build_prompt = lambda problem: (problem, [99])
    engine._select_prefix = lambda *args, **kwargs: None

    def generate(*args, **kwargs):
        engine.calls += 1
        traces = []
        for branch in range(4):
            answer = answer_fn(engine.calls, branch)
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
    return engine


if __name__ == "__main__":
    unittest.main()
