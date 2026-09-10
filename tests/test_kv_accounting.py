import unittest
from types import SimpleNamespace

from path_consistency.kv_accounting import (
    aggregate_question_kv,
    kv_bytes_per_token,
    summarize_window_kv,
    window_kv_tokens,
)
from path_consistency.math_inference import MathInference


class KvAccountingTest(unittest.TestCase):
    def test_shared_prompt_unique_vs_naive(self):
        stats = window_kv_tokens(10, [4, 5, 6, 7])
        self.assertEqual(stats["decode_kv_tokens"], 22)
        self.assertEqual(stats["unique_kv_tokens"], 32)
        self.assertEqual(stats["naive_kv_tokens"], 62)
        self.assertEqual(stats["sequences"], 4)

    def test_llama31_kv_bytes_per_token(self):
        config = SimpleNamespace(
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=4096,
            head_dim=128,
        )
        # 2 * 32 layers * 8 kv heads * 128 dim * 2 bytes
        self.assertEqual(kv_bytes_per_token(config, dtype_bytes=2, tensor_parallel_size=1), 131072)

    def test_reused_window_charges_zero_gpu_kv(self):
        traces = [
            {"prompt_tokens": 8, "generated_tokens": 3, "num_cached_tokens": 8},
            {"prompt_tokens": 8, "generated_tokens": 4, "num_cached_tokens": 8},
        ]
        live = summarize_window_kv(traces, reused=False, kv_bytes_per_token_value=100)
        reused = summarize_window_kv(traces, reused=True, kv_bytes_per_token_value=100)
        self.assertEqual(live["unique_kv_tokens"], 15)
        self.assertEqual(live["logical_unique_kv_tokens"], 15)
        self.assertEqual(reused["unique_kv_tokens"], 0)
        self.assertEqual(reused["logical_unique_kv_tokens"], 15)
        self.assertEqual(reused["naive_kv_tokens"], 0)
        self.assertIsNone(reused["num_cached_tokens"])
        self.assertEqual(live["unique_kv_bytes"], 1500)

    def test_question_aggregate_peaks(self):
        events = [
            summarize_window_kv(
                [{"prompt_tokens": 2, "generated_tokens": 1, "gpu_cache_usage_after": 0.2}],
                reused=False,
            ),
            summarize_window_kv(
                [{"prompt_tokens": 2, "generated_tokens": 3, "gpu_cache_usage_after": 0.5}],
                reused=True,
            ),
        ]
        total = aggregate_question_kv(events)
        self.assertEqual(total["total_unique_kv_tokens"], 3)
        self.assertEqual(total["total_logical_unique_kv_tokens"], 8)
        self.assertEqual(total["peak_gpu_cache_usage"], 0.2)

    def test_pc_reuse_records_zero_gpu_unique_kv(self):
        engine = MathInference.__new__(MathInference)
        engine.calls = 0
        engine.kv_bytes_per_token = 2
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
        pc = engine.run_question(
            "qid",
            "problem",
            "pc",
            max_branches=8,
            window_size=4,
            beta_threshold=1.0,
            paired_root_windows=cached,
        )
        self.assertEqual(sc["total_unique_kv_tokens"], 10)
        self.assertEqual(pc["total_unique_kv_tokens"], 0)
        self.assertEqual(pc["total_logical_unique_kv_tokens"], sc["total_logical_unique_kv_tokens"])
        self.assertEqual(pc["reused_root_windows"], 2)


if __name__ == "__main__":
    unittest.main()
