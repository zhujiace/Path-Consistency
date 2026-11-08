import unittest

from analysis.sc_repetition_matrix import (
    repetition_masks, extract_operations, assign_motifs, bin_fractions, reference_labels,
)


class RepetitionMatrixTests(unittest.TestCase):
    def test_within_path_repeat_is_not_cross_path_repeat(self):
        raw, novel = repetition_masks([[1, 2, 1, 2], [3, 4]], n=2)
        self.assertEqual(raw, [[False] * 4, [False] * 2])

    def test_prompt_filter_and_overlap_union(self):
        raw, novel = repetition_masks([[1, 2, 3, 4], [1, 2, 3, 5]], n=2, prompt=[1, 2])
        self.assertEqual(raw, [[True, True, True, False]] * 2)
        self.assertEqual(novel, [[False, True, True, False]] * 2)

    def test_operation_preserves_operand_order_and_source_span(self):
        text = r'Compute $1,200 \\times 2 = 2400; 7 - 3 = 4; 3 - 7 = -4.'
        text = text.replace('\\\\', '\\')
        found = extract_operations(text)
        self.assertEqual([x['key'] for x in found], ['1200*2', '7-3', '3-7'])
        self.assertEqual(text[found[0]['start']:found[0]['end']], r'1,200 \times 2')

    def test_number_mentions_alone_do_not_count_as_computation(self):
        self.assertEqual(extract_operations('Step 1. There are 16 eggs. Answer 18.'), [])

    def test_negative_operands_are_not_merged_with_positive_ones(self):
        self.assertEqual([x['key'] for x in extract_operations('-3 + 7; 3 * -7; −3 + 7')],
                         ['-3+7', '3*-7', '-3+7'])

    def test_same_color_means_exact_sequence_even_when_shifted(self):
        motifs, labels = assign_motifs([[1, 2, 3, 4], [9, 1, 2, 3, 8]], n=3)
        self.assertEqual(motifs[0]['tokens'], [1, 2, 3])
        self.assertEqual(labels, [[1, 1, 1, 0], [0, 1, 1, 1, 0]])

    def test_bins_do_not_color_unmatched_tokens_at_boundaries(self):
        self.assertEqual(bin_fractions([1, 0, 2], bins=2), [{1: 2/3, 0: 1/3}, {0: 1/3, 2: 2/3}])

    def test_pc_reference_colors_preserved_and_prefix_overrides(self):
        motifs = [{'tokens': [1, 2]}, {'tokens': [3, 4]}]
        self.assertEqual(reference_labels([[3, 4, 1, 2], [1, 2, 3, 4]], motifs, [1, 0]),
                         [[-1, 2, 1, 1], [1, 1, 2, 2]])

    def test_pc_novel_suffix_not_assigned_a_reference_color(self):
        self.assertEqual(reference_labels([[1, 9, 2]], [{'tokens': [1, 2]}], [0]), [[0, 0, 0]])


if __name__ == '__main__':
    unittest.main()
