import unittest

from analysis.sc_prefix_k import prefix_k_from_row, summarize_prefix_k


class PrefixKTest(unittest.TestCase):
    def _row(self):
        traces = []
        windows = []
        for window in range(8):
            windows.append(
                {"window": window, "generation_seconds": 10.0 + window}
            )
            for branch in range(4):
                answer = "18" if window < 5 else "7"
                traces.append(
                    {
                        "answer": answer,
                        "generated_tokens": 100 + window,
                        "window": window,
                    }
                )
        return {
            "question_id": "gsm8k_0000",
            "gold": "18",
            "dataset_task": "gsm8k",
            "traces": traces,
            "windows": windows,
            "predictions": {"SC": {"answer": "18", "correct": True}},
        }

    def test_k20_uses_first_five_windows(self):
        stats = prefix_k_from_row(self._row(), 20)
        self.assertEqual(stats["n_traces"], 20)
        self.assertEqual(stats["n_windows"], 5)
        self.assertTrue(stats["correct"])
        self.assertEqual(stats["voted"], "18")
        self.assertEqual(stats["generated_tokens"], sum(100 + w for w in range(5) for _ in range(4)))
        self.assertEqual(stats["generation_seconds"], sum(10.0 + w for w in range(5)))

    def test_k32_recovers_full_pool(self):
        row = self._row()
        summary = summarize_prefix_k([row], [20, 32])
        self.assertEqual(summary["n"], 1)
        self.assertEqual(summary["official_sc_correct"], 1)
        self.assertEqual(summary["by_k"]["20"]["correct"], 1)
        self.assertEqual(summary["by_k"]["32"]["correct"], 1)


if __name__ == "__main__":
    unittest.main()
