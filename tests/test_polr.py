import unittest

from path_consistency.polr import (
    assemble_polr_result,
    cluster_prefixes,
    indices_to_expand,
    prefix_is_complete,
    summarize_polr_rows,
)


class PolrClusteringTest(unittest.TestCase):
    def test_identical_prefixes_are_one_cluster(self):
        texts = ["Let x = 3. Then the answer is 6."] * 8
        clustering = cluster_prefixes(texts)
        self.assertEqual(clustering.n_clusters, 1)
        self.assertEqual(clustering.dominant_indices, tuple(range(8)))
        self.assertEqual(clustering.kappa, 1.0)

    def test_dominant_cluster_is_the_larger_group(self):
        group_a = [
            "Convert the fraction 3/4 into a decimal by dividing three by four."
        ] * 5
        group_b = [
            "The garden has twelve roses and five tulips so count the flowers."
        ] * 3
        clustering = cluster_prefixes(group_a + group_b)
        self.assertGreaterEqual(clustering.n_clusters, 2)
        self.assertEqual(clustering.dominant_indices, (0, 1, 2, 3, 4))
        self.assertAlmostEqual(clustering.kappa, 5 / 8)

    def test_empty_prefixes_do_not_crash(self):
        clustering = cluster_prefixes(["", "", "alpha beta gamma"])
        self.assertEqual(len(clustering.labels), 3)
        self.assertGreaterEqual(clustering.n_clusters, 1)

    def test_single_prefix(self):
        clustering = cluster_prefixes(["one document"])
        self.assertEqual(clustering.dominant_indices, (0,))
        self.assertEqual(clustering.kappa, 1.0)

    def test_full_dominant_cluster_matches_self_consistency_pool(self):
        texts = [f"path {i % 2} uses the same leftover tokens here {i % 2}" for i in range(6)]
        clustering = cluster_prefixes(texts)
        self.assertEqual(len(clustering.dominant_indices), max(
            clustering.cluster_sizes, key=lambda item: item[1]
        )[1])


class PolrAssemblyTest(unittest.TestCase):
    def _prefix(self, answer, generated, finish="length"):
        return {
            "answer": answer,
            "raw_answer": answer,
            "generated_tokens": generated,
            "prefix_tokens": 0,
            "finish_reason": finish,
            "truncated": finish == "length",
            "full_text": f"prefix {answer}",
        }

    def test_expand_only_incomplete_dominant_members(self):
        prefixes = [
            self._prefix("18", 256, "length"),
            self._prefix("18", 80, "stop"),
            self._prefix("7", 256, "length"),
            self._prefix("7", 256, "length"),
        ]
        clustering = cluster_prefixes(
            [
                "add the apples and get eighteen boxed",
                "add the apples and get eighteen boxed again",
                "subtract the oranges and get seven boxed",
                "subtract the oranges and get seven boxed too",
            ]
        )
        to_expand = indices_to_expand(prefixes, clustering, prefix_len=256)
        for index in to_expand:
            self.assertIn(index, clustering.dominant_indices)
            self.assertFalse(prefix_is_complete(prefixes[index], 256))

    def test_vote_uses_only_dominant_cluster(self):
        prefixes = [
            self._prefix("18", 40, "stop"),
            self._prefix("18", 40, "stop"),
            self._prefix("18", 40, "stop"),
            self._prefix("7", 40, "stop"),
        ]
        clustering = cluster_prefixes(
            [
                "apples eighteen solution boxed",
                "apples eighteen solution boxed",
                "apples eighteen solution boxed",
                "oranges seven different boxed",
            ]
        )
        result = assemble_polr_result(
            prefixes,
            [],
            clustering,
            prefix_seconds=1.0,
            expand_seconds=0.0,
            prefix_len=256,
            gold="18",
            task="gsm8k",
        )
        self.assertEqual(result["answer"], "18")
        self.assertTrue(result["correct"])
        self.assertEqual(result["n_expanded"], 0)
        self.assertEqual(result["total_generated_tokens"], 160)
        self.assertAlmostEqual(result["generation_seconds"], 1.0)

    def test_token_formula_counts_prefix_plus_expansion_only(self):
        prefixes = [
            self._prefix("18", 256, "length"),
            self._prefix("18", 256, "length"),
            self._prefix("7", 256, "length"),
        ]
        clustering = cluster_prefixes(
            [
                "apples eighteen solution boxed",
                "apples eighteen solution boxed",
                "oranges seven different boxed",
            ]
        )
        expanded = [
            {
                "prefix_index": index,
                "answer": "18",
                "raw_answer": "18",
                "generated_tokens": 20,
                "prefix_tokens": 256,
                "finish_reason": "stop",
                "truncated": False,
                "full_text": "expanded 18",
            }
            for index in clustering.dominant_indices
        ]
        result = assemble_polr_result(
            prefixes,
            expanded,
            clustering,
            prefix_seconds=2.0,
            expand_seconds=0.5,
            prefix_len=256,
            gold="18",
            task="gsm8k",
        )
        self.assertEqual(result["prefix_generated_tokens"], 768)
        self.assertEqual(
            result["expand_generated_tokens"], 20 * len(clustering.dominant_indices)
        )
        self.assertEqual(
            result["total_generated_tokens"],
            768 + 20 * len(clustering.dominant_indices),
        )
        self.assertTrue(all(trace["expanded"] for trace in result["traces"] if trace["in_dominant"]))

    def test_summary_accuracy(self):
        summary = summarize_polr_rows(
            [
                {"correct": True, "total_generated_tokens": 100, "generation_seconds": 1, "kappa": 0.5, "dominant_size": 2, "traces": []},
                {"correct": False, "total_generated_tokens": 50, "generation_seconds": 1, "kappa": 1.0, "dominant_size": 4, "traces": []},
            ]
        )
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["questions"], 2)
        self.assertAlmostEqual(summary["accuracy"], 0.5)
        self.assertAlmostEqual(summary["avg_generated_tokens"], 75)


if __name__ == "__main__":
    unittest.main()
