from __future__ import annotations
from typing import Any, Optional, Sequence

def window_kv_tokens(prompt_len: int, generated_lengths: Sequence[int]) -> dict[str, int]:
    prompt_len = int(prompt_len or 0)
    generated = [int(value or 0) for value in generated_lengths]
    decode = sum(generated)
    sequences = len(generated)
    return {'prompt_kv_tokens': prompt_len, 'decode_kv_tokens': decode, 'unique_kv_tokens': prompt_len + decode, 'naive_kv_tokens': sequences * prompt_len + decode, 'sequences': sequences}

def kv_bytes_per_token(model_config: Any, dtype_bytes: int=2, tensor_parallel_size: int=1) -> Optional[int]:
    try:
        n_layers = int(model_config.num_hidden_layers)
        n_heads = int(model_config.num_attention_heads)
        n_kv = int(getattr(model_config, 'num_key_value_heads', n_heads) or n_heads)
        hidden = int(model_config.hidden_size)
        head_dim = int(getattr(model_config, 'head_dim', None) or hidden // n_heads)
        tp = max(1, int(tensor_parallel_size))
        if n_kv % tp:
            n_kv_rank = n_kv
        else:
            n_kv_rank = n_kv // tp
        return 2 * n_layers * n_kv_rank * head_dim * int(dtype_bytes)
    except (TypeError, ValueError, AttributeError, ZeroDivisionError):
        return None

def _optional_max(values: Sequence[Any]) -> Optional[float]:
    present = [float(value) for value in values if value is not None]
    return max(present) if present else None

def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def summarize_window_kv(traces: Sequence[dict], reused: bool=False, kv_bytes_per_token_value: Optional[int]=None) -> dict[str, Any]:
    prompt_len = 0
    if traces:
        prompt_len = int(traces[0].get('prompt_tokens') or 0)
    stats = window_kv_tokens(prompt_len, [trace.get('generated_tokens') or 0 for trace in traces])
    logical_unique = stats['unique_kv_tokens']
    unique = 0 if reused else logical_unique
    naive = 0 if reused else stats['naive_kv_tokens']
    cached = _optional_max((trace.get('num_cached_tokens') for trace in traces))
    usage = _optional_max((trace.get('gpu_cache_usage_after') if trace.get('gpu_cache_usage_after') is not None else trace.get('gpu_cache_usage') for trace in traces))
    blocks_used = _optional_max((trace.get('gpu_kv_blocks_used') for trace in traces))
    blocks_total = None
    for trace in traces:
        if trace.get('gpu_kv_blocks_total') is not None:
            blocks_total = _optional_int(trace.get('gpu_kv_blocks_total'))
            break
    bytes_each = kv_bytes_per_token_value
    if bytes_each is None and traces:
        bytes_each = traces[0].get('kv_bytes_per_token')
    unique_bytes = unique * bytes_each if bytes_each is not None else None
    naive_bytes = naive * bytes_each if bytes_each is not None else None
    logical_bytes = logical_unique * bytes_each if bytes_each is not None else None
    return {'unique_kv_tokens': unique, 'logical_unique_kv_tokens': logical_unique, 'naive_kv_tokens': naive, 'decode_kv_tokens': stats['decode_kv_tokens'], 'prompt_kv_tokens': stats['prompt_kv_tokens'], 'num_cached_tokens': None if reused else cached, 'gpu_cache_usage': None if reused else usage, 'gpu_kv_blocks_used': None if reused else blocks_used, 'gpu_kv_blocks_total': blocks_total, 'kv_bytes_per_token': bytes_each, 'unique_kv_bytes': unique_bytes, 'logical_unique_kv_bytes': logical_bytes, 'naive_kv_bytes': naive_bytes}

def aggregate_question_kv(window_events: Sequence[dict]) -> dict[str, Any]:
    unique = sum((int(event.get('unique_kv_tokens') or 0) for event in window_events))
    logical = sum((int(event.get('logical_unique_kv_tokens') if event.get('logical_unique_kv_tokens') is not None else event.get('unique_kv_tokens') or 0) for event in window_events))
    naive = sum((int(event.get('naive_kv_tokens') or 0) for event in window_events))
    decode = sum((int(event.get('decode_kv_tokens') or 0) for event in window_events))
    cached_vals = [event.get('num_cached_tokens') for event in window_events if event.get('num_cached_tokens') is not None]
    usage = _optional_max((event.get('gpu_cache_usage') for event in window_events))
    blocks_used = _optional_max((event.get('gpu_kv_blocks_used') for event in window_events))
    bytes_each = None
    for event in window_events:
        if event.get('kv_bytes_per_token') is not None:
            bytes_each = int(event['kv_bytes_per_token'])
            break
    return {'total_unique_kv_tokens': unique, 'total_logical_unique_kv_tokens': logical, 'total_naive_kv_tokens': naive, 'total_decode_kv_tokens': decode, 'total_prefix_cache_hit_tokens': int(sum(cached_vals)) if cached_vals else None, 'peak_gpu_cache_usage': usage, 'peak_gpu_kv_blocks': blocks_used, 'kv_bytes_per_token': bytes_each, 'total_unique_kv_bytes': unique * bytes_each if bytes_each is not None else None, 'total_logical_unique_kv_bytes': logical * bytes_each if bytes_each is not None else None, 'total_naive_kv_bytes': naive * bytes_each if bytes_each is not None else None}

def snapshot_vllm_gpu_kv(llm: Any) -> dict[str, Any]:
    empty = {'gpu_cache_usage': None, 'gpu_kv_blocks_used': None, 'gpu_kv_blocks_total': None, 'gpu_kv_block_size': None}
    engine = getattr(llm, 'llm_engine', None)
    if engine is None:
        return empty
    try:
        schedulers = getattr(engine, 'scheduler', None)
        cache = getattr(engine, 'cache_config', None)
        if schedulers and cache is not None and getattr(cache, 'num_gpu_blocks', None):
            total = int(cache.num_gpu_blocks)
            free = sum((int(scheduler.block_manager.get_num_free_gpu_blocks()) for scheduler in schedulers))
            used = max(total - free, 0)
            block_size = getattr(cache, 'block_size', None)
            return {'gpu_cache_usage': used / total if total else None, 'gpu_kv_blocks_used': used, 'gpu_kv_blocks_total': total, 'gpu_kv_block_size': int(block_size) if block_size else None}
    except Exception:
        pass
    try:
        core = getattr(engine, 'engine_core', None)
        inner = getattr(core, 'engine_core', core)
        scheduler = getattr(inner, 'scheduler', None)
        manager = getattr(scheduler, 'kv_cache_manager', None)
        if manager is None:
            return empty
        usage = float(manager.usage)
        total = int(manager.num_gpu_blocks)
        used = int(round(usage * total)) if total else None
        return {'gpu_cache_usage': usage, 'gpu_kv_blocks_used': used, 'gpu_kv_blocks_total': total, 'gpu_kv_block_size': int(getattr(manager, 'block_size', 0) or 0) or None}
    except Exception:
        return empty
