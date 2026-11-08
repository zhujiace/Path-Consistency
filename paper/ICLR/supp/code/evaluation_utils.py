from __future__ import annotations
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
DEEPCONF_LABELS = {'offline_mean': 'Offline-Mean', 'offline_bottom10_90': 'Offline-Bottom10@90%', 'offline_bottom10_10': 'Offline-Bottom10@10%', 'offline_tail_90': 'Offline-Tail@90%', 'offline_tail_10': 'Offline-Tail@10%', 'offline_lgc_90': 'Offline-LGC@90%', 'offline_lgc_10': 'Offline-LGC@10%'}

def _record_task(item: dict) -> str:
    dataset = str(item.get('dataset') or item.get('task') or '').lower()
    if dataset == 'gpqa' or dataset.startswith('gpqa') or 'choices' in item:
        return 'gpqa'
    if dataset.startswith('aime') or item.get('task') == 'aime':
        return 'aime'
    if 'problem' in item and 'answer' in item:
        return 'math'
    if 'question' in item and 'target' in item:
        return 'gsm8k'
    return 'unknown'

def load_eval_records(path: Path, max_questions: int | None, sample_seed: int, source_rows: list[int] | None=None, num_shards: int=1, shard_index: int=0) -> tuple[list[dict], str]:
    from path_consistency.math_eval import extract_aime_integer, extract_gpqa_letter, format_gsm8k_gold
    from path_consistency.math_protocol import format_gpqa_stem
    records = []
    task = None
    with path.open() as handle:
        for (row_number, line) in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            record_task = _record_task(item)
            if record_task == 'gpqa':
                choices = item['choices']
                gold = extract_gpqa_letter(item.get('answer'))
                if gold is None:
                    raise ValueError(f'{path} row {row_number} has no A/B/C/D gold')
                records.append({'question_id': str(item.get('unique_id') or item.get('Record ID') or f'gpqa_{row_number:03d}'), 'problem': format_gpqa_stem(item['question'], choices), 'answer': gold, 'subject': item.get('domain') or item.get('High-level domain'), 'level': item.get('subdomain') or item.get('Subdomain'), 'source_row': row_number, 'year': None, 'task': 'gpqa'})
            elif record_task == 'aime':
                gold = extract_aime_integer(item.get('answer'))
                if gold is None:
                    raise ValueError(f'{path} row {row_number} is not an AIME integer')
                year = item.get('year')
                records.append({'question_id': str(item.get('unique_id', f'aime_{year}_{row_number:02d}')), 'problem': item['problem'], 'answer': gold, 'subject': f'AIME {year}' if year else 'AIME', 'level': year, 'source_row': row_number, 'year': year, 'task': 'aime'})
            elif record_task == 'math':
                records.append({'question_id': str(item.get('unique_id', f'math_{row_number}')), 'problem': item['problem'], 'answer': str(item['answer']), 'subject': item.get('subject'), 'level': item.get('level'), 'source_row': row_number, 'year': None, 'task': 'math'})
            elif record_task == 'gsm8k':
                records.append({'question_id': f'gsm8k_{row_number:04d}', 'problem': item['question'], 'answer': format_gsm8k_gold(item['target']), 'subject': 'gsm8k', 'level': None, 'source_row': row_number, 'year': None, 'task': 'gsm8k'})
            else:
                raise ValueError(f'{path} row {row_number} is not a supported eval JSONL record')
            if task is None:
                task = record_task
            elif task != record_task:
                raise ValueError(f'{path} mixes {task} and {record_task} records')
    if not records:
        raise ValueError(f'{path} is empty')
    if source_rows is not None:
        invalid = [index for index in source_rows if index < 0 or index >= len(records)]
        if invalid:
            raise ValueError(f'Invalid --source-rows values: {invalid}')
        records = [records[index] for index in source_rows]
    elif max_questions is not None and max_questions < len(records):
        rng = random.Random(sample_seed)
        indices = list(range(len(records)))
        rng.shuffle(indices)
        records = [records[index] for index in indices[:max_questions]]
    if num_shards < 1:
        raise ValueError('--num-shards must be >= 1')
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError('--shard-index must be in [0, num-shards)')
    if num_shards > 1:
        records = [record for (index, record) in enumerate(records) if index % num_shards == shard_index]
    return (records, task or 'math')

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def combined_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.name).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()

def stable_question_seed(seed: int, question_id: str) -> int:
    payload = f'{seed}:{question_id}'.encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2 ** 31 - 1)

def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]

def _present(rows: list[dict], key: str) -> list:
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None:
            values.append(value)
    return values

def _mean(rows: list[dict], key: str) -> float | None:
    values = _present(rows, key)
    if not values:
        return None
    return sum(values) / len(values)

def _sum_present(rows: list[dict], key: str) -> float | None:
    values = _present(rows, key)
    if not values:
        return None
    return sum(values)

def _max_present(rows: list[dict], key: str) -> float | None:
    values = _present(rows, key)
    if not values:
        return None
    return max(values)

def summarize(rows: list[dict], config: dict, engine_times: dict) -> dict:
    prediction_names = sorted({name for row in rows for name in row.get('predictions', {})})
    methods = {}
    for name in prediction_names:
        eligible = [row['predictions'][name] for row in rows if name in row['predictions']]
        methods[name] = {'questions': len(eligible), 'correct': sum((bool(item['correct']) for item in eligible)), 'accuracy': sum((bool(item['correct']) for item in eligible)) / len(eligible) if eligible else None}
    parent_costs = {}
    for method in sorted({row['method'] for row in rows}):
        selected = [row for row in rows if row['method'] == method]
        if not selected:
            continue
        parent_costs[method] = {'questions': len(selected), 'total_generated_tokens': sum((row['total_generated_tokens'] for row in selected)), 'avg_generated_tokens': sum((row['total_generated_tokens'] for row in selected)) / len(selected), 'avg_prompt_tokens': sum((row['total_prompt_tokens'] for row in selected)) / len(selected), 'avg_prefix_tokens': sum((row['total_prefix_tokens'] for row in selected)) / len(selected), 'avg_request_prompt_tokens': sum((row.get('total_request_prompt_tokens', row['total_prompt_tokens']) for row in selected)) / len(selected), 'total_generation_seconds': sum((row['generation_seconds'] for row in selected)), 'avg_generation_seconds': sum((row['generation_seconds'] for row in selected)) / len(selected), 'total_actual_generation_seconds': sum((row.get('actual_generation_seconds', row['generation_seconds']) for row in selected)), 'reused_root_windows': sum((row.get('reused_root_windows', 0) for row in selected)), 'parser_failures': sum((row['parser_failures'] for row in selected)), 'truncations': sum((row['truncations'] for row in selected)), 'avg_unique_kv_tokens': _mean(selected, 'total_unique_kv_tokens'), 'avg_logical_unique_kv_tokens': _mean(selected, 'total_logical_unique_kv_tokens'), 'avg_naive_kv_tokens': _mean(selected, 'total_naive_kv_tokens'), 'total_unique_kv_tokens': _sum_present(selected, 'total_unique_kv_tokens'), 'total_naive_kv_tokens': _sum_present(selected, 'total_naive_kv_tokens'), 'avg_unique_kv_bytes': _mean(selected, 'total_unique_kv_bytes'), 'peak_gpu_cache_usage': _max_present(selected, 'peak_gpu_cache_usage'), 'avg_prefix_cache_hit_tokens': _mean(selected, 'total_prefix_cache_hit_tokens')}
    return {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config': config, 'engine_times': engine_times, 'completed_rows': len(rows), 'methods': methods, 'parent_costs': parent_costs}
