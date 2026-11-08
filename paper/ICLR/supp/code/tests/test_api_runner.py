import contextlib
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import run_math_deepseek as runner


class OfflineClient:
    def __init__(self, **kwargs):
        self.usage_totals = {'requests': 0}
        self.lock = threading.Lock()

    def complete(self, messages, **kwargs):
        with self.lock:
            self.usage_totals['requests'] += 1
        reasoning = 'Two plus two equals four. We add the two quantities. The resulting total is four.'
        if len(messages) > 1:
            prefix = messages[-1].get('reasoning_content', '')
            reasoning = reasoning[len(prefix):] if reasoning.startswith(prefix) else reasoning
        return {
            'reasoning_content': reasoning,
            'content': r'\boxed{4}',
            'reasoning_logprobs': [{'token': char, 'logprob': -0.1} for char in reasoning],
            'usage': {'completion_tokens': len(reasoning), 'prompt_tokens': 20},
            'finish_reason': 'stop',
        }

    def close(self):
        pass


class ApiRunnerTest(unittest.TestCase):
    def test_paired_run_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data.jsonl'
            data.write_text(json.dumps({'question': 'What is 2 + 2?', 'target': 4}) + '\n')
            probe = root / 'probe.json'
            probe.write_text(json.dumps({'protocol': {'ready_for_math500_pc': True}}))
            output = root / 'results'
            argv = [
                'run_math_deepseek.py', '--dataset', str(data), '--outdir', str(output),
                '--smoke-path', str(probe), '--selector', 'blc', '--max-branches', '8',
                '--window-size', '4', '--question-concurrency', '1', '--tag', 'offline',
            ]
            config = {'api_key': 'offline-test', 'base_url': 'https://invalid.example/beta', 'model': 'offline'}
            with patch.object(runner, 'DeepSeekClient', OfflineClient), patch.object(runner, 'load_deepseek_env', return_value=config), patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
                runner.main()
                runner.main()
            manifest = json.loads(next(output.glob('*.run.json')).read_text())
            self.assertEqual(manifest['status'], 'complete')
            for method in ('sc', 'pc'):
                rows = next(output.glob(f'*_{method}.jsonl')).read_text().splitlines()
                self.assertEqual(len(rows), 1)
                row = json.loads(rows[0])
                self.assertTrue(any(item['correct'] for item in row['predictions'].values()))
                if method == 'pc':
                    self.assertGreaterEqual(row['reused_root_windows'], 1)


if __name__ == '__main__':
    unittest.main()
