import unittest
from unittest.mock import Mock

from path_consistency.deepseek_api import assistant_prefix_message, continuation_text
from path_consistency.deepseek_inference import (
    DeepSeekMathInference,
    parse_logprob_channel,
    reasoning_span_length,
)
from path_consistency.math_protocol import (
    decode_token_pieces,
    select_shared_prefix,
    token_sequence_distance,
)


class DeepSeekProtocolTest(unittest.TestCase):
    def test_continuation_strips_repeated_prefix(self):
        self.assertEqual(continuation_text("abc", "abcdef"), "def")
        self.assertEqual(continuation_text("abc", "xyz"), "xyz")
        self.assertEqual(continuation_text("", "xyz"), "xyz")

    def test_prefix_message_variants(self):
        message = assistant_prefix_message("think", "reasoning_content")
        self.assertEqual(message["role"], "assistant")
        self.assertTrue(message["prefix"])
        self.assertEqual(message["reasoning_content"], "think")

    def test_parse_reasoning_logprobs(self):
        items = [
            {
                "token": "Hello",
                "logprob": -0.1,
                "top_logprobs": [
                    {"token": "Hello", "logprob": -0.1},
                    {"token": "Hi", "logprob": -1.2},
                ],
            },
            {"token": " world", "logprob": -0.2, "top_logprobs": []},
        ]
        tokens, sampled, confidences = parse_logprob_channel(items)
        self.assertEqual(tokens, ["Hello", " world"])
        self.assertEqual(sampled, [-0.1, -0.2])
        self.assertEqual(len(confidences), 2)
        self.assertEqual(decode_token_pieces(tokens), "Hello world")

    def test_reasoning_span_cuts_before_boxed(self):
        tokens = ["step", " one", ".\n", "\\boxed", "{2}"]
        text = decode_token_pieces(tokens)
        self.assertEqual(reasoning_span_length(tokens, text, ""), 3)

    def test_blc_beats_weaker_logprobs(self):
        traces = [
            {
                "branch_id": "a",
                "answer": "2",
                "full_token_ids": ["aaaa"] * 20,
                "sampled_logprobs": [-0.1] * 20,
                "reasoning_tokens": 20,
                "generated_tokens": 20,
            },
            {
                "branch_id": "b",
                "answer": "2",
                "full_token_ids": ["bbbb"] * 20,
                "sampled_logprobs": [-2.0] * 20,
                "reasoning_tokens": 20,
                "generated_tokens": 10,
            },
        ]
        selected = select_shared_prefix(
            traces, "2", [], 0.5, decode_token_pieces, selector="blc"
        )
        self.assertEqual(selected["source_id"], "a")

    def test_uniform_is_deterministic_and_stays_in_eligible_pool(self):
        traces = [
            {
                "branch_id": "a",
                "answer": "2",
                "full_token_ids": ["aaaa"] * 20,
                "sampled_logprobs": [-0.1] * 20,
                "reasoning_tokens": 20,
                "generated_tokens": 20,
            },
            {
                "branch_id": "b",
                "answer": "2",
                "full_token_ids": ["bbbb"] * 20,
                "sampled_logprobs": [-2.0] * 20,
                "reasoning_tokens": 20,
                "generated_tokens": 10,
            },
        ]
        first = select_shared_prefix(
            traces, "2", [], 0.5, decode_token_pieces, selector="uniform", selection_seed=0
        )
        second = select_shared_prefix(
            traces, "2", [], 0.5, decode_token_pieces, selector="uniform", selection_seed=0
        )
        self.assertEqual(first["source_id"], second["source_id"])
        self.assertIn(first["source_id"], {"a", "b"})
        self.assertEqual(first["selector"], "uniform")
        blc = select_shared_prefix(
            traces, "2", [], 0.5, decode_token_pieces, selector="blc"
        )
        self.assertEqual(blc["source_id"], "a")

    def test_medoid_selects_central_prefix(self):
        traces = [
            {
                "branch_id": "a",
                "answer": "2",
                "full_token_ids": ["x"] * 10 + ["a"] * 10,
                "sampled_logprobs": [None] * 20,
                "reasoning_tokens": 20,
                "generated_tokens": 20,
            },
            {
                "branch_id": "b",
                "answer": "2",
                "full_token_ids": ["x"] * 10 + ["b"] * 10,
                "sampled_logprobs": [None] * 20,
                "reasoning_tokens": 20,
                "generated_tokens": 20,
            },
            {
                "branch_id": "c",
                "answer": "2",
                "full_token_ids": ["x"] * 10 + ["a"] * 10,
                "sampled_logprobs": [None] * 20,
                "reasoning_tokens": 20,
                "generated_tokens": 20,
            },
        ]
        selected = select_shared_prefix(
            traces, "2", [], 0.8, decode_token_pieces, selector="token-medoid"
        )
        self.assertEqual(selected["source_id"], "a")
        self.assertEqual(token_sequence_distance(["a", "b"], ["a", "c"]), 1)

    def test_api_backend_reuses_sc_root_windows(self):
        client = Mock()
        engine = DeepSeekMathInference(client, selector="token-medoid")
        engine.calls = 0

        def generate(*args, **kwargs):
            engine.calls += 1
            traces = []
            for branch in range(4):
                answer = f"ans-{engine.calls}-{branch}"
                tokens = [answer]
                traces.append(
                    {
                        "full_token_ids": tokens,
                        "new_token_ids": tokens,
                        "sampled_logprobs": [None],
                        "token_confidence": [None],
                        "full_text": answer,
                        "suffix_text": answer,
                        "raw_answer": answer,
                        "answer": answer,
                        "reasoning_tokens": 1,
                        "finish_reason": "stop",
                        "stop_reason": None,
                        "generated_tokens": 1,
                        "prompt_tokens": 8,
                        "prefix_tokens": 0,
                        "truncated": False,
                        "lgc_2048": 0.0,
                    }
                )
            return traces, float(engine.calls)

        engine._generate_window = generate
        sc = engine.run_question(
            question_id="qid",
            problem="2+2",
            method="sc",
            max_branches=8,
            window_size=4,
            beta_threshold=1.0,
            return_internal_windows=True,
        )
        cached = sc.pop("_internal_root_windows")
        self.assertEqual(engine.calls, 2)
        engine.calls = 0
        pc = engine.run_question(
            question_id="qid",
            problem="2+2",
            method="pc",
            max_branches=8,
            window_size=4,
            beta_threshold=1.0,
            paired_root_windows=cached,
        )
        self.assertEqual(engine.calls, 0)
        self.assertEqual(pc["reused_root_windows"], 2)
        self.assertEqual(
            [trace["answer"] for trace in pc["traces"]],
            [trace["answer"] for trace in sc["traces"]],
        )


if __name__ == "__main__":
    unittest.main()
