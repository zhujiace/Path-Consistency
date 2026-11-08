import math
import unittest
from collections import Counter

from analysis.sc_prefix_similarity import (
    analyze_rows,
    branch_sort_key,
    bootstrap_mean_ci,
    cosine_counts,
    fractional_prefixes,
    grouped_pairwise_similarity,
    mean_between_similarity,
    normalize_reasoning_tokens,
    render_svg,
    render_tex_rows,
    word_ngram_counts,
)


class _WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        self.tokens = text.split()
        return list(range(len(self.tokens)))

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return " ".join(self.tokens[index] for index in ids)


class ScPrefixSimilarityTest(unittest.TestCase):
    def test_branch_sort_key_accepts_real_window_branch_identifiers(self):
        traces = [
            {"window": 1, "branch_id": "gsm8k_0000:w1:b0"},
            {"window": 0, "branch_id": "gsm8k_0000:w0:b2"},
            {"window": 0, "branch_id": "gsm8k_0000:w0:b1"},
        ]
        ordered = sorted(traces, key=branch_sort_key)
        self.assertEqual(
            [trace["branch_id"] for trace in ordered],
            ["gsm8k_0000:w0:b1", "gsm8k_0000:w0:b2", "gsm8k_0000:w1:b0"],
        )

    def test_fractional_prefixes_use_cumulative_ceil_boundaries(self):
        prefixes = fractional_prefixes(
            "one two three four five six",
            (0.2, 0.4, 0.6, 0.8),
            _WhitespaceTokenizer(),
        )
        self.assertEqual(prefixes[0.2], "one two")
        self.assertEqual(prefixes[0.4], "one two three")
        self.assertEqual(prefixes[0.6], "one two three four")
        self.assertEqual(prefixes[0.8], "one two three four five")

    def test_normalization_ignores_formatting_but_preserves_arithmetic(self):
        tokens = normalize_reasoning_tokens("We need $3 \\times 4$.\nThen 12 = 12!")
        self.assertEqual(tokens, ("we", "need", "3", "times", "4", "then", "12", "=", "12"))
        self.assertEqual(
            word_ngram_counts(tokens),
            Counter(
                {
                    ("we", "need"): 1,
                    ("need", "3"): 1,
                    ("3", "times"): 1,
                    ("times", "4"): 1,
                    ("4", "then"): 1,
                    ("then", "12"): 1,
                    ("12", "="): 1,
                    ("=", "12"): 1,
                }
            ),
        )

    def test_cosine_counts_has_hand_checked_values(self):
        left = Counter({("a", "b"): 1, ("b", "c"): 1})
        same = Counter(left)
        partial = Counter({("a", "b"): 1})
        disjoint = Counter({("x", "y"): 2})
        self.assertAlmostEqual(cosine_counts(left, same), 1.0)
        self.assertAlmostEqual(cosine_counts(left, partial), 1 / math.sqrt(2))
        self.assertEqual(cosine_counts(left, disjoint), 0.0)
        self.assertEqual(cosine_counts(Counter(), disjoint), 0.0)

    def test_grouped_pairwise_separates_same_and_different_answers(self):
        vectors = [
            Counter({("a", "b"): 1}),
            Counter({("a", "b"): 1}),
            Counter({("x", "y"): 1}),
        ]
        result = grouped_pairwise_similarity(vectors, ("1", "1", "2"))
        self.assertEqual(result["overall_n"], 3)
        self.assertAlmostEqual(result["overall"], 1 / 3)
        self.assertEqual(result["same_n"], 1)
        self.assertEqual(result["same"], 1.0)
        self.assertEqual(result["different_n"], 2)
        self.assertEqual(result["different"], 0.0)

    def test_between_similarity_uses_every_cross_pair(self):
        left = [Counter({("a", "b"): 1}), Counter({("x", "y"): 1})]
        right = [Counter({("a", "b"): 1}), Counter({("a", "b"): 1})]
        self.assertAlmostEqual(mean_between_similarity(left, right), 0.5)

    def test_bootstrap_constant_sample_has_degenerate_interval(self):
        result = bootstrap_mean_ci([0.25, 0.25, 0.25], repetitions=200, seed=7)
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["mean"], 0.25)
        self.assertEqual(result["ci95"], [0.25, 0.25])

    def test_analysis_macro_averages_questions_and_builds_cross_question_baseline(self):
        def row(source_row, text, answer, truncated=False):
            return {
                "source_row": source_row,
                "question_id": f"q{source_row}",
                "traces": [
                    {
                        "branch_id": branch,
                        "full_text": text,
                        "answer": answer,
                        "raw_answer": answer,
                        "truncated": truncated and branch == 0,
                    }
                    for branch in range(3)
                ],
            }

        result = analyze_rows(
            [
                row(0, "alpha beta gamma delta epsilon", "1", truncated=True),
                row(1, "omega psi chi phi tau", "2"),
            ],
            tokenizer=_WhitespaceTokenizer(),
            ratios=(0.2, 0.8),
            expected_traces=3,
            bootstrap_repetitions=100,
            seed=11,
        )
        self.assertEqual(result["data_quality"]["questions"], 2)
        self.assertEqual(result["data_quality"]["traces"], 6)
        self.assertEqual(result["data_quality"]["truncated_traces"], 1)
        for ratio in ("0.2", "0.8"):
            self.assertAlmostEqual(result["by_ratio"][ratio]["within_question"]["mean"], 1.0)
            self.assertAlmostEqual(result["by_ratio"][ratio]["cross_question"]["mean"], 0.0)
            self.assertAlmostEqual(result["by_ratio"][ratio]["difference"]["mean"], 1.0)
            self.assertAlmostEqual(result["by_ratio"][ratio]["same_answer"]["mean"], 1.0)
            self.assertIsNone(result["by_ratio"][ratio]["different_answer"])

    def test_cross_question_baseline_uses_seeded_random_cycle(self):
        rows = []
        words = ("alpha", "beta", "gamma", "delta")
        for source_row, word in enumerate(words):
            rows.append(
                {
                    "source_row": source_row,
                    "question_id": f"q{source_row}",
                    "traces": [
                        {
                            "branch_id": branch,
                            "full_text": f"{word} {word} {word} {word} {word}",
                            "answer": str(source_row),
                            "raw_answer": str(source_row),
                            "truncated": False,
                        }
                        for branch in range(2)
                    ],
                }
            )
        result = analyze_rows(
            rows,
            tokenizer=_WhitespaceTokenizer(),
            ratios=(0.2,),
            expected_traces=2,
            bootstrap_repetitions=20,
            seed=7,
        )
        partners = {
            record["source_row"]: record["baseline_partner_source_row"]
            for record in result["per_question"]
        }
        self.assertEqual(partners, {0: 2, 1: 0, 2: 3, 3: 1})

    def test_renderers_include_all_ratio_rows_and_valid_svg(self):
        summary = {
            "by_ratio": {
                key: {
                    "within_question": {"mean": value, "ci95": [value - 0.01, value + 0.01]},
                    "cross_question": {"mean": 0.1, "ci95": [0.09, 0.11]},
                    "difference": {"mean": value - 0.1, "ci95": [value - 0.12, value - 0.08]},
                    "same_answer": {"mean": value + 0.05, "ci95": [value + 0.04, value + 0.06]},
                    "different_answer": {"mean": value - 0.05, "ci95": [value - 0.06, value - 0.04]},
                    "untruncated": {"mean": value, "ci95": [value - 0.01, value + 0.01]},
                }
                for key, value in (("0.2", 0.4), ("0.4", 0.5), ("0.6", 0.6), ("0.8", 0.7))
            }
        }
        tex = render_tex_rows(summary)
        self.assertEqual(tex.count(r"\\"), 4)
        self.assertIn("80\\%", tex)
        svg = render_svg(summary)
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("Within-question SC pairs", svg)
        self.assertIn("Cross-question baseline", svg)
        self.assertTrue(svg.rstrip().endswith("</svg>"))


if __name__ == "__main__":
    unittest.main()
