import json
import tempfile
import unittest
from pathlib import Path
from path_consistency.math_eval import extract_gsm8k_numeric, extract_last_boxed, format_gsm8k_gold, grade_answer, gsm8k_numeric_core, gsm8k_numeric_equal, math_equal
from evaluation_utils import load_eval_records

class Gsm8kDatasetTest(unittest.TestCase):

    def test_gold_canonicalization(self):
        self.assertEqual(format_gsm8k_gold(18.0), '18')
        self.assertEqual(format_gsm8k_gold(3), '3')
        self.assertEqual(format_gsm8k_gold('18.0'), '18')

    def test_numeric_grader(self):
        self.assertTrue(math_equal('18', '18.0'))
        self.assertTrue(math_equal('18', format_gsm8k_gold(18.0)))

    def test_numeric_core_strips_units_and_currency(self):
        self.assertEqual(gsm8k_numeric_core('\\$70,000'), '70000')
        self.assertEqual(gsm8k_numeric_core('160\\text{ minutes}'), '160')
        self.assertEqual(gsm8k_numeric_core('\\$26.00'), '26')
        self.assertEqual(gsm8k_numeric_core('8{,}000'), '8000')
        self.assertEqual(gsm8k_numeric_core('60\\%'), '60')
        self.assertIsNone(gsm8k_numeric_core('\\frac{400}{11}\\text{seconds}'))
        self.assertTrue(gsm8k_numeric_equal('\\$125', '125'))
        self.assertTrue(grade_answer('gsm8k', '45\\text{miles}', '45'))
        self.assertFalse(gsm8k_numeric_equal('12', '8'))
        self.assertTrue(gsm8k_numeric_equal('1\\text{hour}35\\text{minutes}', '95'))
        self.assertTrue(gsm8k_numeric_equal('3\\text{minutes}36\\text{seconds}', '216'))

    def test_vote_keys_match_numeric_grader(self):
        from path_consistency.math_eval import canonicalize_vote_answer
        self.assertEqual(canonicalize_vote_answer('gsm8k', '\\$70,000'), '70000')
        self.assertEqual(canonicalize_vote_answer('gsm8k', '160\\text{ minutes}'), '160')
        self.assertEqual(canonicalize_vote_answer('gsm8k', '1\\text{hour}35\\text{minutes}'), '95')
        traces = [{'raw_answer': '\\$70,000', 'answer': '\\$70,000'}]
        from path_consistency.math_eval import recanonicalize_trace_answers
        recanonicalize_trace_answers(traces, 'gsm8k')
        self.assertEqual(traces[0]['answer'], '70000')

    def test_extract_prefers_boxed_then_answer_is(self):
        self.assertEqual(extract_last_boxed('work \\boxed{18}'), '18')
        self.assertEqual(extract_gsm8k_numeric('3 + 5 = 8. The answer is 18.'), '18')

    def test_loader_and_round_robin_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'gsm8k.jsonl'
            rows = [{'question': f'q{index}', 'target': float(index)} for index in range(8)]
            path.write_text(''.join((json.dumps(row) + '\n' for row in rows)))
            (all_rows, task) = load_eval_records(path, None, 2027)
            self.assertEqual(task, 'gsm8k')
            self.assertEqual(len(all_rows), 8)
            self.assertEqual(all_rows[0]['question_id'], 'gsm8k_0000')
            self.assertEqual(all_rows[0]['answer'], '0')
            (shard0, _) = load_eval_records(path, None, 2027, num_shards=4, shard_index=0)
            (shard1, _) = load_eval_records(path, None, 2027, num_shards=4, shard_index=1)
            self.assertEqual([row['source_row'] for row in shard0], [0, 4])
            self.assertEqual([row['source_row'] for row in shard1], [1, 5])
if __name__ == '__main__':
    unittest.main()
