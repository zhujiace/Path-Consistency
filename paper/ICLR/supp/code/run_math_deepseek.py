from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from path_consistency.deepseek_api import DeepSeekAPIError, DeepSeekClient, assistant_prefix_message, continuation_text, load_deepseek_env
from path_consistency.deepseek_inference import DeepSeekMathInference, parse_logprob_channel
from path_consistency.math_eval import grade_answer, recanonicalize_trace_answers
from path_consistency.math_protocol import decode_token_pieces, instruction_for_task
from path_consistency.generation_methods import is_root_for_any, method_label, method_spec, pair_root_method, parse_generation_methods
from evaluation_utils import DEEPCONF_LABELS, combined_sha256, file_sha256, load_eval_records, load_rows, stable_question_seed, summarize
ENGINEERING_ROWS = [303, 457, 359, 315, 316, 415, 407, 370, 151, 332]
SMOKE_PROBLEM = 'Compute 2+2. Please reason step by step, and put your final answer within \\boxed{}.'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dataset', default='data.jsonl')
    parser.add_argument('--outdir', default='results')
    parser.add_argument('--tag', default='math500_deepseek_v4_flash_b32_c08_rep0')
    parser.add_argument('--methods', default='sc,pc')
    parser.add_argument('--max-questions', type=int, default=None)
    parser.add_argument('--source-rows', default=None)
    parser.add_argument('--exclude-engineering-split', action='store_true', default=True)
    parser.add_argument('--include-engineering-split', action='store_true', help='Include the 10 Llama engineering IDs in this DeepSeek run')
    parser.add_argument('--sample-seed', type=int, default=2027)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-branches', type=int, default=32)
    parser.add_argument('--window-size', type=int, default=4)
    parser.add_argument('--beta-threshold', type=float, default=0.8)
    parser.add_argument('--early-stop-threshold', type=float, default=0.95, help='AC stop threshold (Adaptive-Consistency default 0.95; not PC C=0.8)')
    parser.add_argument('--ratios', default='0,0.2,0.4,0.6,0.8')
    parser.add_argument('--max-tokens', type=int, default=32768)
    parser.add_argument('--logprobs', type=int, default=20)
    parser.add_argument('--reasoning-effort', default='high')
    parser.add_argument('--question-concurrency', type=int, default=8)
    parser.add_argument('--max-inflight', type=int, default=32)
    parser.add_argument('--user-id', default=None)
    parser.add_argument('--store-text', action='store_true')
    parser.add_argument('--store-confidence', action='store_true')
    parser.add_argument('--prefix-variant', default='reasoning_content')
    parser.add_argument('--selector', default='auto', help='auto (from smoke), blc, token-medoid, or uniform')
    parser.add_argument('--recovery', choices=['on', 'off'], default='on', help='on = trial/commit + one-level rollback + two-strike; off = keep failed trial/committed prefix, no stack pop or two-strike')
    parser.add_argument('--selection-seed', type=int, default=0, help='Recorded seed for uniform source selection; ignored by BLC')
    parser.add_argument('--smoke-path', default='results/reports/deepseek_smoke.json', help='Reuse the frozen prefix/logprob smoke instead of requiring one in --outdir')
    parser.add_argument('--reuse-root-cache', action='append', default=[], help='Existing *_root_cache directories (repeatable). PC can reuse SC no-prefix windows without regenerating SC. Answers are recanonicalized.')
    return parser.parse_args()

def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

def append_jsonl(path: Path, row: dict, lock: threading.Lock) -> None:
    line = json.dumps(row, ensure_ascii=False) + '\n'
    with lock:
        with path.open('a') as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

def save_root_cache(path: Path, windows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        json.dump(windows, handle)

def root_cache_path(cache_dir: Path, qid: str, root_method: str) -> Path:
    stem = qid.replace('/', '_')
    if root_method == 'sc':
        return cache_dir / f'{stem}.json.gz'
    return cache_dir / f'{stem}.{root_method}.json.gz'

def load_root_cache(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)

def recanonicalize_root_windows(windows: list[dict] | None, task: str) -> list[dict] | None:
    if not windows:
        return windows
    for window in windows:
        recanonicalize_trace_answers(window.get('traces') or [], task)
    return windows

def index_root_caches(directories: list[str]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for raw in directories:
        directory = Path(raw).expanduser().resolve()
        if not directory.is_dir():
            raise SystemExit(f'--reuse-root-cache not a directory: {directory}')
        for path in directory.glob('*.json.gz'):
            index[path.name.replace('.json.gz', '')] = path
    return index

def _preview(text: str, limit: int=240) -> str:
    compact = ' '.join((text or '').split())
    return compact[:limit]

def run_smoke(args: argparse.Namespace, output_dir: Path) -> dict:
    env = load_deepseek_env()
    events = []
    client = DeepSeekClient(env=env, max_inflight=4, user_id='pc-math500-smoke', on_event=events.append, timeout_s=180.0)
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'model': client.model, 'base_url': client.base_url, 'checks': {}}
    prompt = [{'role': 'user', 'content': SMOKE_PROBLEM}]
    try:
        first = client.complete(prompt, max_tokens=256, reasoning_effort=args.reasoning_effort, logprobs=True, top_logprobs=20, request_tag='smoke-logprob')
    except DeepSeekAPIError as error:
        report['checks']['basic_completion'] = {'ok': False, 'status_code': error.status_code, 'message': str(error), 'body': error.body}
        dump_json(output_dir / 'reports' / 'deepseek_smoke.json', report)
        client.close()
        return report
    (reasoning_tokens, sampled, _) = parse_logprob_channel(first.get('reasoning_logprobs') or [])
    (content_tokens, _, _) = parse_logprob_channel(first.get('content_logprobs') or [])
    report['checks']['basic_completion'] = {'ok': True, 'response_id': first.get('id'), 'returned_model': first.get('model'), 'system_fingerprint': first.get('system_fingerprint'), 'finish_reason': first.get('finish_reason'), 'usage': first.get('usage'), 'reasoning_chars': len(first.get('reasoning_content') or ''), 'content_chars': len(first.get('content') or ''), 'content_preview': _preview(first.get('content') or ''), 'reasoning_preview': _preview(first.get('reasoning_content') or ''), 'reasoning_logprob_tokens': len(reasoning_tokens), 'content_logprob_tokens': len(content_tokens), 'sampled_logprob_finite': sum((1 for value in sampled if value is not None and value > -9000))}
    concurrent = []
    errors = []

    def _one(index: int) -> dict:
        return client.complete(prompt, max_tokens=128, reasoning_effort=args.reasoning_effort, logprobs=True, request_tag=f'smoke-conc-{index}')
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_one, index) for index in range(4)]
        for future in as_completed(futures):
            try:
                item = future.result()
                concurrent.append({'id': item.get('id'), 'fingerprint': item.get('system_fingerprint'), 'usage': item.get('usage'), 'attempt': item.get('attempt')})
            except Exception as error:
                errors.append(error.__class__.__name__)
    ids = [row['id'] for row in concurrent]
    report['checks']['concurrency4'] = {'ok': len(concurrent) == 4 and len(set(ids)) == 4 and (not errors), 'n_ok': len(concurrent), 'unique_ids': len(set(ids)), 'errors': errors, 'retries': [row.get('attempt') for row in concurrent]}
    prefix_source = first.get('reasoning_content') or ''
    if reasoning_tokens:
        cut = max(8, len(reasoning_tokens) // 5)
        prefix_text = decode_token_pieces(reasoning_tokens[:cut])
        prefix_kind = 'logprob_tokens'
    else:
        cut = max(40, len(prefix_source) // 5)
        prefix_text = prefix_source[:cut]
        prefix_kind = 'character_fallback'
    report['prefix_probe'] = {'kind': prefix_kind, 'prefix_chars': len(prefix_text), 'prefix_preview': _preview(prefix_text, 160)}
    variant_results = {}
    winning_variant = None
    for variant in ('reasoning_content', 'reasoning_only', 'both', 'content'):
        messages = [{'role': 'user', 'content': SMOKE_PROBLEM}, assistant_prefix_message(prefix_text, variant)]
        try:
            continued = client.complete(messages, max_tokens=256, reasoning_effort=args.reasoning_effort, logprobs=True, request_tag=f'smoke-prefix-{variant}')
        except DeepSeekAPIError as error:
            variant_results[variant] = {'ok': False, 'status_code': error.status_code, 'message': str(error), 'body': error.body}
            continue
        returned = continued.get('reasoning_content') or ''
        suffix = continuation_text(prefix_text, returned)
        continued_thinking = bool(returned) and (returned.startswith(prefix_text) or (suffix and suffix != returned) or (returned and (not returned.startswith(prefix_source[:40] if prefix_source else ' '))))
        looks_like_restart = bool(prefix_source) and returned.startswith(prefix_source[:80])
        ok = bool(suffix.strip()) and (not looks_like_restart)
        variant_results[variant] = {'ok': ok, 'response_id': continued.get('id'), 'finish_reason': continued.get('finish_reason'), 'usage': continued.get('usage'), 'returned_reasoning_chars': len(returned), 'suffix_chars': len(suffix), 'starts_with_prefix': returned.startswith(prefix_text) if prefix_text else False, 'looks_like_restart': looks_like_restart, 'reasoning_logprob_tokens': len(continued.get('reasoning_logprobs') or []), 'content_preview': _preview(continued.get('content') or ''), 'suffix_preview': _preview(suffix)}
        if ok and winning_variant is None:
            winning_variant = variant
    logprobs_ok = report['checks']['basic_completion'].get('reasoning_logprob_tokens', 0) > 0
    prefix_ok = winning_variant is not None
    selector = 'blc' if logprobs_ok else 'token-medoid'
    report['checks']['prefix_completion'] = {'ok': prefix_ok, 'winning_variant': winning_variant, 'variants': variant_results}
    report['protocol'] = {'selector': selector, 'prefix_variant': winning_variant or args.prefix_variant, 'logprobs_ok': logprobs_ok, 'concurrency_ok': report['checks']['concurrency4']['ok'], 'ready_for_math500_pc': prefix_ok, 'label': 'PC' if logprobs_ok else 'PC-medoid'}
    report['usage_totals'] = dict(client.usage_totals)
    report['http_events'] = events
    dump_json(output_dir / 'reports' / 'deepseek_smoke.json', report)
    client.close()
    return report

def process_question(item: dict, methods: list[str], engine: DeepSeekMathInference, args: argparse.Namespace, ratios: tuple[float, ...], result_paths: dict[str, Path], cache_dir: Path, done: dict[str, set[str]], write_lock: threading.Lock, done_lock: threading.Lock, dataset_task: str, reused_root_index: dict[str, Path] | None=None) -> dict:
    qid = item['question_id']
    stem = qid.replace('/', '_')
    cached_roots: dict[str, list] = {}
    outputs = {}
    for method in methods:
        spec = method_spec(method)
        root = spec['root']
        cache_this = is_root_for_any(method, methods)
        with done_lock:
            already = qid in done[method]
        if already:
            if cache_this and any((qid not in done[child] for child in methods if pair_root_method(child) == method)):
                cached_roots[method] = recanonicalize_root_windows(load_root_cache(root_cache_path(cache_dir, qid, method)), dataset_task)
            continue
        paired_root = None
        if root:
            paired_root = cached_roots.get(root)
            if paired_root is None:
                source = root_cache_path(cache_dir, qid, root)
                if reused_root_index and root == 'sc' and (stem in reused_root_index):
                    source = reused_root_index[stem]
                paired_root = recanonicalize_root_windows(load_root_cache(source), dataset_task)
                if paired_root is None and root not in methods:
                    raise RuntimeError(f'{qid}: {method} needs a reused {root} root cache, missing {source}')
        output = engine.run_question(question_id=qid, problem=item['problem'], method=method, max_branches=args.max_branches, window_size=args.window_size, beta_threshold=args.beta_threshold, ratios=ratios, temperature=0.6, top_p=0.9, top_k=0, max_tokens=args.max_tokens, logprobs=args.logprobs, presence_penalty=0.0, seed=stable_question_seed(args.seed, qid), store_text=args.store_text, store_confidence=args.store_confidence, paired_root_windows=paired_root, return_internal_windows=cache_this, early_stop_threshold=args.early_stop_threshold, recovery=args.recovery)
        if cache_this and '_internal_root_windows' in output:
            cached_roots[method] = output.pop('_internal_root_windows')
            save_root_cache(root_cache_path(cache_dir, qid, method), cached_roots[method])
        gold = item['answer']
        first_answer = output['traces'][0]['answer'] if output['traces'] else None
        parent_name = method_label(method)
        predictions = {parent_name: {'answer': output['answer'], 'correct': grade_answer(dataset_task, output['answer'], gold)}, f'Pass@1({parent_name})': {'answer': first_answer, 'correct': grade_answer(dataset_task, first_answer, gold)}}
        if output.get('deepconf') and args.logprobs >= 20:
            for (policy_name, policy) in output['deepconf'].items():
                suffix = '+PC' if spec['share_prefix'] else ''
                display_name = DEEPCONF_LABELS.get(policy_name, policy_name) + suffix
                predictions[display_name] = {'answer': policy['answer'], 'correct': grade_answer(dataset_task, policy['answer'], gold), 'retained': policy.get('retained'), 'valid_traces': policy.get('valid_traces')}
        row = {**output, 'gold': gold, 'subject': item.get('subject'), 'level': item.get('level'), 'source_row': item.get('source_row'), 'year': item.get('year'), 'dataset_task': dataset_task, 'predictions': predictions}
        append_jsonl(result_paths[method], row, write_lock)
        with done_lock:
            done[method].add(qid)
        outputs[method] = {'question_id': qid, 'method': method, 'predictions': predictions, 'generated_tokens': output['total_generated_tokens'], 'generation_seconds': round(output['generation_seconds'], 3), 'actual_generation_seconds': round(output.get('actual_generation_seconds', output['generation_seconds']), 3), 'reused_root_windows': output.get('reused_root_windows', 0), 'parser_failures': output['parser_failures'], 'truncations': output['truncations'], 'final_prefix_ratio': output['final_prefix_ratio'], 'selector': output.get('selector')}
    return {'question_id': qid, 'methods': outputs}

def main() -> None:
    args = parse_args()
    output_dir = Path(args.outdir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'reports').mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)
    if args.smoke:
        report = run_smoke(args, output_dir)
        print(json.dumps({k: v for (k, v) in report.items() if k != 'http_events'}, indent=2, ensure_ascii=False))
        protocol = report.get('protocol') or {}
        if not protocol.get('ready_for_math500_pc'):
            raise SystemExit('DeepSeek prefix-completion smoke failed; not launching MATH-500 PC.')
        return
    smoke_path = Path(args.smoke_path).expanduser().resolve()
    if not smoke_path.exists():
        alt = output_dir / 'reports' / 'deepseek_smoke.json'
        if alt.exists():
            smoke_path = alt
        else:
            raise SystemExit('Run with --smoke first, or pass --smoke-path to a passing prefix/logprob probe.')
    smoke = json.loads(smoke_path.read_text())
    protocol = smoke.get('protocol') or {}
    if not protocol.get('ready_for_math500_pc'):
        raise SystemExit('Stored DeepSeek smoke did not pass prefix continuation.')
    selector = args.selector
    if selector == 'auto':
        selector = protocol.get('selector') or 'token-medoid'
    if selector == 'random':
        selector = 'uniform'
    if selector not in {'blc', 'token-medoid', 'pc-medoid', 'uniform'}:
        raise SystemExit(f'unknown --selector {selector}')
    prefix_variant = args.prefix_variant
    if prefix_variant == 'reasoning_content' and protocol.get('prefix_variant'):
        prefix_variant = protocol['prefix_variant']
    try:
        methods = parse_generation_methods(args.methods)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    ratios = tuple((float(value) for value in args.ratios.split(',')))
    dataset_path = Path(args.dataset).expanduser().resolve()
    requested_rows = [int(value) for value in args.source_rows.split(',')] if args.source_rows else None
    (data, dataset_task) = load_eval_records(dataset_path, args.max_questions, args.sample_seed, requested_rows)
    excluded_engineering = []
    if dataset_task == 'math' and (not args.include_engineering_split):
        data = [item for item in data if item['source_row'] not in ENGINEERING_ROWS]
        excluded_engineering = list(ENGINEERING_ROWS)
        if args.max_questions is not None and requested_rows is None:
            data = data[:args.max_questions]
    user_id = args.user_id or f'pc-{args.tag}'[:64]
    implementation_paths = [Path(__file__).resolve(), Path(__file__).resolve().parent / 'path_consistency' / 'deepseek_api.py', Path(__file__).resolve().parent / 'path_consistency' / 'deepseek_inference.py', Path(__file__).resolve().parent / 'path_consistency' / 'math_protocol.py', Path(__file__).resolve().parent / 'path_consistency' / 'math_eval.py', Path(__file__).resolve().parent / 'path_consistency' / 'confidence.py', Path(__file__).resolve().parent / 'path_consistency' / 'generation_methods.py']
    env = load_deepseek_env()
    config = {'protocol_version': 'math-pc-paired-root-v2-deepseek-api', 'implementation_sha256': combined_sha256(implementation_paths), 'model': env['model'], 'base_url': env['base_url'], 'dataset': str(dataset_path), 'dataset_sha256': file_sha256(dataset_path), 'dataset_task': dataset_task, 'selected_question_ids': [item['question_id'] for item in data], 'excluded_engineering_source_rows': excluded_engineering, 'methods': methods, 'max_questions': args.max_questions, 'requested_source_rows': requested_rows, 'sample_seed': args.sample_seed, 'seed': args.seed, 'max_branches': args.max_branches, 'window_size': args.window_size, 'beta_threshold': args.beta_threshold, 'early_stop_threshold': args.early_stop_threshold, 'ratios': ratios, 'max_tokens': args.max_tokens, 'logprobs': args.logprobs, 'reasoning_effort': args.reasoning_effort, 'thinking': True, 'selector': selector, 'recovery': args.recovery, 'selection_seed': args.selection_seed, 'prefix_variant': prefix_variant, 'question_concurrency': args.question_concurrency, 'max_inflight': args.max_inflight, 'user_id': user_id, 'gpus': 'api', 'prompt': '{problem}\\n\\n' + instruction_for_task(dataset_task).replace('\n', '\\n'), 'pc_length_policy': 'provisional_fixed_plus_20_percent', 'paired_common_root_windows': any((pair_root_method(method) in methods for method in methods)) or bool(args.reuse_root_cache), 'reuse_root_cache': [str(Path(p).expanduser().resolve()) for p in args.reuse_root_cache], 'gsm8k_vote_canonicalization': 'numeric_core_duration', 'api_has_no_seed': True, 'repetition_index': args.seed, 'note': 'API has no sampling seed; --seed only labels this repetition. C=0.8 is not Dev-Mix C*.'}
    reused_root_index = index_root_caches(args.reuse_root_cache)
    if methods == ['pc'] and (not reused_root_index):
        raise SystemExit('PC-only DeepSeek run requires --reuse-root-cache')
    if methods == ['pc']:
        missing = [item['question_id'] for item in data if item['question_id'].replace('/', '_') not in reused_root_index]
        if missing:
            raise SystemExit(f'PC-only run missing {len(missing)} SC root caches, e.g. {missing[:5]}')
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True, default=list).encode()).hexdigest()[:12]
    base_name = f'{args.tag}_{fingerprint}'
    manifest_path = output_dir / f'{base_name}.run.json'
    result_paths = {method: output_dir / f'{base_name}_{method}.jsonl' for method in methods}
    summary_path = output_dir / f'{base_name}.summary.json'
    cache_dir = output_dir / f'{base_name}_root_cache'
    api_log_path = output_dir / f'{base_name}_api_calls.jsonl'
    cache_dir.mkdir(parents=True, exist_ok=True)
    run_record = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(), 'config': config, 'fingerprint': fingerprint, 'result_paths': {key: str(value) for (key, value) in result_paths.items()}, 'smoke_path': str(smoke_path), 'api_log_path': str(api_log_path)}
    dump_json(manifest_path, run_record)
    api_lock = threading.Lock()

    def on_event(event: dict) -> None:
        event = {**event, 'ts_utc': datetime.now(timezone.utc).isoformat()}
        append_jsonl(api_log_path, event, api_lock)
    client = DeepSeekClient(env=env, max_inflight=args.max_inflight, user_id=user_id, on_event=on_event)
    engine = DeepSeekMathInference(client, selector=selector, prefix_variant=prefix_variant, reasoning_effort=args.reasoning_effort, want_logprobs=args.logprobs > 0, logprob_top_k=args.logprobs, task=dataset_task, selection_seed=args.selection_seed)
    engine_times = {'model_type': engine.model_type, 'tokenizer_init_seconds': 0.0, 'engine_init_seconds': 0.0, 'selector': selector, 'recovery': args.recovery, 'selection_seed': args.selection_seed, 'prefix_variant': prefix_variant, 'label': protocol.get('label')}
    run_record['engine_times'] = engine_times
    dump_json(manifest_path, run_record)
    done = {method: {row['question_id'] for row in load_rows(path)} for (method, path) in result_paths.items()}
    write_lock = threading.Lock()
    done_lock = threading.Lock()
    summary_lock = threading.Lock()
    total_start = time.perf_counter()

    def after_question(progress_index: int, qid: str) -> None:
        rows = [row for path in result_paths.values() for row in load_rows(path)]
        summary = summarize(rows, config, engine_times)
        summary['run_elapsed_seconds'] = time.perf_counter() - total_start
        summary['api_usage'] = dict(client.usage_totals)
        with summary_lock:
            dump_json(summary_path, summary)
        print(json.dumps({'progress': f'{progress_index}/{len(data)}', 'question_id': qid, 'sc_done': len(done.get('sc', [])), 'pc_done': len(done.get('pc', [])), 'api_usage': client.usage_totals}, ensure_ascii=False), flush=True)
    pending = []
    for item in data:
        qid = item['question_id']
        if all((qid in done[method] for method in methods)):
            continue
        pending.append(item)
    try:
        if args.question_concurrency <= 1:
            for (index, item) in enumerate(pending, start=1):
                result = process_question(item, methods, engine, args, ratios, result_paths, cache_dir, done, write_lock, done_lock, dataset_task, reused_root_index)
                after_question(index, result['question_id'])
                print(json.dumps(result, ensure_ascii=False), flush=True)
        else:
            with ThreadPoolExecutor(max_workers=args.question_concurrency) as pool:
                futures = {pool.submit(process_question, item, methods, engine, args, ratios, result_paths, cache_dir, done, write_lock, done_lock, dataset_task, reused_root_index): item for item in pending}
                for (index, future) in enumerate(as_completed(futures), start=1):
                    result = future.result()
                    after_question(index, result['question_id'])
                    print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as error:
        run_record.update({'status': 'failed', 'failed_at_utc': datetime.now(timezone.utc).isoformat(), 'error_type': error.__class__.__name__, 'error_message': str(error)})
        dump_json(manifest_path, run_record)
        client.close()
        raise
    rows = [row for path in result_paths.values() for row in load_rows(path)]
    summary = summarize(rows, config, engine_times)
    summary['run_elapsed_seconds'] = time.perf_counter() - total_start
    summary['api_usage'] = dict(client.usage_totals)
    dump_json(summary_path, summary)
    run_record.update({'status': 'complete', 'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'run_elapsed_seconds': summary['run_elapsed_seconds'], 'summary_path': str(summary_path), 'api_usage': dict(client.usage_totals)})
    dump_json(manifest_path, run_record)
    client.close()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
if __name__ == '__main__':
    main()
