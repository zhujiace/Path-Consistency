from __future__ import annotations
import copy
import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional, Sequence
import numpy as np
from .confidence import beta_confidence
from .deepconf_offline import aggregate_deepconf_offline
from .generation_methods import AC_STOP_THRESHOLD, ESC_STOP_WINDOW, method_spec, should_early_stop
from .kv_accounting import aggregate_question_kv, summarize_window_kv
from .math_eval import majority_vote
MATH_INSTRUCTION = 'Please reason step by step, and put your final answer within \\boxed{}.'
GPQA_INSTRUCTION = 'Please reason step by step, and put the single correct option letter (A, B, C, or D) within \\boxed{}.'

def instruction_for_task(task: str) -> str:
    if task == 'gpqa':
        return GPQA_INSTRUCTION
    return MATH_INSTRUCTION

def format_gpqa_stem(question: str, choices: dict) -> str:
    return '\n'.join(['Answer the following multiple-choice question.', '', 'Question:', question.strip(), '', 'Choices:', f"(A) {choices['A']}", f"(B) {choices['B']}", f"(C) {choices['C']}", f"(D) {choices['D']}"])
DecodeFn = Callable[[Sequence], str]

def gate_snapshot(answers: list[Optional[str]], threshold: float) -> dict:
    valid = [answer for answer in answers if answer is not None]
    counts = Counter(valid)
    modal = counts.most_common(1)[0][0] if counts else None
    votes = counts[modal] if modal is not None else 0
    confidence = beta_confidence(valid)
    return {'modal_answer': modal, 'modal_votes': votes, 'valid_answers': len(valid), 'confidence': confidence, 'threshold': threshold, 'passed': bool(valid) and confidence >= threshold}

def balanced_prefix(text: str) -> bool:
    if text.count('```') % 2:
        return False
    depth = 0
    escaped = False
    for char in text:
        if char == '\\' and (not escaped):
            escaped = True
            continue
        if char == '{' and (not escaped):
            depth += 1
        elif char == '}' and (not escaped):
            depth -= 1
            if depth < 0:
                return False
        escaped = False
    return depth == 0

def decode_token_pieces(pieces: Sequence) -> str:
    return ''.join((str(piece) for piece in pieces))

def safe_ratio_endpoint(full_reasoning_ids: list, ratio: float, minimum: int, decode: DecodeFn, max_backtrack: int=32) -> Optional[int]:
    if not full_reasoning_ids:
        return None
    target = min(len(full_reasoning_ids), int(math.floor(ratio * len(full_reasoning_ids))))
    if target <= minimum:
        return None
    lower = max(minimum + 1, target - max_backtrack)
    fallback = None
    for endpoint in range(target, lower - 1, -1):
        text = decode(full_reasoning_ids[:endpoint])
        if '\\boxed' in text or 'Final Answer' in text or 'final answer' in text:
            continue
        if not balanced_prefix(text):
            continue
        fallback = fallback or endpoint
        if text.endswith(('\n', '.', ';', ':', ',', ')', ']', '}')):
            return endpoint
    return fallback

def lower_tail_local_score(sampled_logprobs: list[Optional[float]], start: int, end: int, window_size: int=128, stride: int=64) -> float:
    values = [float(value) for value in sampled_logprobs[start:end] if value is not None and math.isfinite(value)]
    if not values:
        return float('-inf')
    if len(values) <= window_size:
        return float(np.mean(values))
    means = [float(np.mean(values[offset:offset + window_size])) for offset in range(0, len(values) - window_size + 1, stride)]
    last_start = len(values) - window_size
    if last_start % stride:
        means.append(float(np.mean(values[last_start:])))
    return min(means)

def token_sequence_distance(left: Sequence, right: Sequence) -> int:
    shared = min(len(left), len(right))
    mismatches = sum((a != b for (a, b) in zip(left[:shared], right[:shared])))
    return mismatches + abs(len(left) - len(right))

def select_shared_prefix(traces: list[dict], modal_answer: str, current_prefix_ids: list, next_ratio: float, decode: DecodeFn, selector: str='blc', selection_seed: int=0) -> Optional[dict]:
    candidates = []
    current_length = len(current_prefix_ids)
    for trace in traces:
        if trace['answer'] != modal_answer:
            continue
        reasoning_end = trace['reasoning_tokens']
        reasoning_ids = trace['full_token_ids'][:reasoning_end]
        endpoint = safe_ratio_endpoint(reasoning_ids, next_ratio, current_length, decode)
        if endpoint is None:
            continue
        added = endpoint - current_length
        blc_score = lower_tail_local_score(trace['sampled_logprobs'], 0, added, window_size=128, stride=64)
        candidates.append({'source_id': trace['branch_id'], 'token_ids': reasoning_ids[:endpoint], 'endpoint': endpoint, 'source_reasoning_tokens': reasoning_end, 'actual_ratio': endpoint / reasoning_end if reasoning_end else None, 'blc_score': blc_score, 'source_generated_tokens': trace['generated_tokens'], 'selector': selector})
    if not candidates:
        return None
    if selector == 'blc':
        return min(candidates, key=lambda item: (-item['blc_score'], item['source_generated_tokens'], item['source_id']))
    if selector in {'token-medoid', 'pc-medoid'}:
        return min(candidates, key=lambda item: (sum((token_sequence_distance(item['token_ids'], other['token_ids']) for other in candidates)), item['source_generated_tokens'], item['source_id']))
    if selector in {'uniform', 'random'}:
        ordered = sorted(candidates, key=lambda item: (item['source_id'], item['endpoint']))
        payload = {'selection_seed': int(selection_seed), 'next_ratio': next_ratio, 'candidates': [(item['source_id'], item['endpoint']) for item in ordered]}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).digest()
        index = int.from_bytes(digest[:8], 'big') % len(ordered)
        chosen = dict(ordered[index])
        chosen['selector'] = 'uniform'
        chosen['selection_seed'] = int(selection_seed)
        return chosen
    raise ValueError(f'Unknown prefix selector: {selector}')

@dataclass
class PrefixState:
    token_ids: list
    ratio: float
    modal_answer: Optional[str]
    beta_confidence: float
    source_id: Optional[str]

def run_sc_pc_question(backend, question_id: str, problem: str, method: str, max_branches: int=32, window_size: int=4, beta_threshold: float=0.8, ratios: tuple[float, ...]=(0.0, 0.2, 0.4, 0.6, 0.8), temperature: float=0.6, top_p: float=0.95, top_k: int=20, max_tokens: int=32768, logprobs: int=20, presence_penalty: float=0.0, seed: int=0, store_text: bool=False, store_confidence: bool=False, paired_root_windows: Optional[list[dict]]=None, return_internal_windows: bool=False, early_stop_threshold: float=AC_STOP_THRESHOLD, esc_stop_rule: str=ESC_STOP_WINDOW, recovery: str='on') -> dict:
    spec = method_spec(method)
    share_prefix = bool(spec['share_prefix'])
    early_stop = spec['early_stop']
    recover = (recovery or 'on') != 'off'
    if recovery not in {'on', 'off', None}:
        raise ValueError(f'unknown recovery: {recovery}')
    if max_branches % window_size:
        raise ValueError('max_branches must be divisible by window_size')
    (prompt_text, base_prompt_ids) = backend.build_prompt(problem)
    all_traces = []
    window_events = []
    validated_answers: list[str] = []
    state_stack = [PrefixState([], 0.0, None, 0.0, None)]
    pending_trial = None
    no_more_prefixing = not share_prefix
    early_stopped = False
    consecutive_failures = 0
    generation_seconds = 0.0
    actual_generation_seconds = 0.0
    reused_root_generation_seconds = 0.0
    reused_root_windows = 0
    internal_windows = []
    question_start = time.perf_counter()
    num_windows = max_branches // window_size
    for window_index in range(num_windows):
        safe_state = state_stack[-1]
        active_prefix_ids = pending_trial['token_ids'] if pending_trial else safe_state.token_ids
        active_ratio = pending_trial['target_ratio'] if pending_trial else safe_state.ratio
        sampling_seed = int((seed * 1000003 + window_index * 10007 + 17) % (2 ** 31 - 1))
        reused_sc_root = False
        if window_index == 0 and not active_prefix_ids and paired_root_windows is not None and (window_index < len(paired_root_windows)) and (paired_root_windows[window_index].get('sampling_seed') == sampling_seed):
            cached = paired_root_windows[window_index]
            traces = copy.deepcopy(cached['traces'])
            elapsed = float(cached['generation_seconds'])
            reused_sc_root = True
            reused_root_windows += 1
            reused_root_generation_seconds += elapsed
        else:
            (traces, elapsed) = backend._generate_window(base_prompt_ids, active_prefix_ids, window_size, sampling_seed, temperature, top_p, top_k, max_tokens, logprobs, presence_penalty)
            actual_generation_seconds += elapsed
        generation_seconds += elapsed
        for (branch_offset, trace) in enumerate(traces):
            trace['branch_id'] = f'{question_id}:w{window_index}:b{branch_offset}'
            trace['window'] = window_index
            trace['prefix_ratio'] = active_ratio
            trace['window_seconds'] = elapsed
            trace['reused_sc_root'] = reused_sc_root
        all_traces.extend(traces)
        if not active_prefix_ids:
            internal_windows.append({'window': window_index, 'sampling_seed': sampling_seed, 'generation_seconds': elapsed, 'traces': copy.deepcopy(traces)})
        window_answers = [trace['answer'] for trace in traces]
        window_valid = [answer for answer in window_answers if answer is not None]
        event = {'window': window_index, 'used_ratio': active_ratio, 'used_prefix_tokens': len(active_prefix_ids), 'used_prefix_sha256': hashlib.sha256(repr(tuple(active_prefix_ids)).encode()).hexdigest()[:12], 'generation_seconds': elapsed, 'sampling_seed': sampling_seed, 'reused_sc_root': reused_sc_root, 'valid_answers': len(window_valid), 'trial': pending_trial is not None, 'prompt_tokens': traces[0]['prompt_tokens'] if traces else 0}
        event.update(summarize_window_kv(traces, reused=reused_sc_root, kv_bytes_per_token_value=getattr(backend, 'kv_bytes_per_token', None)))
        if store_text:
            event['used_prefix_token_ids'] = list(active_prefix_ids)
        accepted_for_source = True
        if not share_prefix:
            validated_answers.extend(window_valid)
            event['decision'] = 'sc'
        elif pending_trial is not None:
            trial_snapshot = gate_snapshot(validated_answers + window_valid, beta_threshold)
            event['gate_after'] = trial_snapshot
            if trial_snapshot['passed'] and trial_snapshot['modal_answer'] == pending_trial['parent_modal']:
                validated_answers.extend(window_valid)
                committed = PrefixState(token_ids=list(pending_trial['token_ids']), ratio=pending_trial['target_ratio'], modal_answer=trial_snapshot['modal_answer'], beta_confidence=trial_snapshot['confidence'], source_id=pending_trial['source_id'])
                state_stack.append(committed)
                consecutive_failures = 0
                event['decision'] = 'commit'
                pending_trial = None
            else:
                accepted_for_source = False
                event['rollback_reason'] = 'mode_flip' if trial_snapshot['modal_answer'] != pending_trial['parent_modal'] else 'beta_below_threshold'
                if recover:
                    consecutive_failures += 1
                    event['decision'] = 'rollback_trial'
                    pending_trial = None
                else:
                    event['decision'] = 'retain_failed_trial'
        elif safe_state.ratio > 0:
            monitor_snapshot = gate_snapshot(validated_answers + window_valid, beta_threshold)
            event['gate_after'] = monitor_snapshot
            if monitor_snapshot['passed'] and monitor_snapshot['modal_answer'] == safe_state.modal_answer:
                validated_answers.extend(window_valid)
                safe_state.modal_answer = monitor_snapshot['modal_answer']
                safe_state.beta_confidence = monitor_snapshot['confidence']
                consecutive_failures = 0
                event['decision'] = 'keep'
            else:
                accepted_for_source = False
                event['rollback_reason'] = 'mode_flip' if monitor_snapshot['modal_answer'] != safe_state.modal_answer else 'beta_below_threshold'
                if recover:
                    failed_state = state_stack.pop()
                    consecutive_failures += 1
                    event['decision'] = 'rollback_committed'
                    event['rolled_back_source'] = failed_state.source_id
                else:
                    event['decision'] = 'retain_failed_committed'
        else:
            validated_answers.extend(window_valid)
            event['decision'] = 'root_observe'
        if recover and consecutive_failures >= 2:
            state_stack = [PrefixState([], 0.0, None, 0.0, None)]
            pending_trial = None
            no_more_prefixing = True
            event['decision'] += '+disable_prefixing'
        current_gate = gate_snapshot(validated_answers, beta_threshold)
        event['validated_gate'] = current_gate
        event['validated_answer_count'] = len(validated_answers)
        if share_prefix and (not no_more_prefixing) and accepted_for_source and (window_index < num_windows - 1) and current_gate['passed']:
            current_state = state_stack[-1]
            try:
                ratio_index = ratios.index(current_state.ratio)
            except ValueError:
                ratio_index = max((index for (index, ratio) in enumerate(ratios) if ratio <= current_state.ratio))
            if ratio_index + 1 < len(ratios):
                next_ratio = ratios[ratio_index + 1]
                candidate = backend._select_prefix(traces, current_gate['modal_answer'], current_state.token_ids, next_ratio)
                if candidate is not None:
                    pending_trial = {**candidate, 'target_ratio': next_ratio, 'parent_modal': current_gate['modal_answer'], 'parent_confidence': current_gate['confidence']}
                    event['proposed_prefix'] = {key: value for (key, value) in pending_trial.items() if key != 'token_ids'}
                    event['proposed_prefix']['token_ids_sha256'] = hashlib.sha256(repr(tuple(pending_trial['token_ids'])).encode()).hexdigest()[:12]
                    if store_text:
                        event['proposed_prefix']['token_ids'] = list(pending_trial['token_ids'])
                else:
                    event['proposal'] = 'no_safe_candidate'
        all_answers_so_far = [trace['answer'] for trace in all_traces]
        if should_early_stop(early_stop, window_answers, all_answers_so_far, early_stop_threshold, esc_stop_rule=esc_stop_rule, window_size=window_size):
            early_stopped = True
            event['early_stopped'] = True
            event['decision'] += '+early_stop'
        window_events.append(event)
        if early_stopped:
            break
    final_answer = majority_vote((trace['answer'] for trace in all_traces))
    deepconf = aggregate_deepconf_offline(all_traces, window_size=2048)
    public_traces = []
    for trace in all_traces:
        public = {key: value for (key, value) in trace.items() if key not in {'full_token_ids', 'new_token_ids', 'sampled_logprobs', 'token_confidence', 'full_text', 'suffix_text', 'reasoning_content', 'content'}}
        if store_text:
            public['full_text'] = trace.get('full_text')
            public['suffix_text'] = trace.get('suffix_text')
            if 'reasoning_content' in trace:
                public['reasoning_content'] = trace['reasoning_content']
            if 'content' in trace:
                public['content'] = trace['content']
        if store_confidence:
            public['token_confidence'] = trace.get('token_confidence')
            public['sampled_logprobs'] = trace.get('sampled_logprobs')
        public_traces.append(public)
    prompt_tokens = 0
    if base_prompt_ids and (not isinstance(base_prompt_ids[0], str)):
        prompt_tokens = len(base_prompt_ids)
    elif all_traces:
        prompt_tokens = int(all_traces[0].get('prompt_tokens') or 0)
    result = {'question_id': question_id, 'method': method, 'prompt': prompt_text, 'prompt_tokens': prompt_tokens, 'answer': final_answer, 'deepconf': deepconf, 'traces': public_traces, 'windows': window_events, 'total_generated_tokens': sum((trace['generated_tokens'] for trace in all_traces)), 'total_prompt_tokens': sum((trace['prompt_tokens'] for trace in all_traces)), 'total_prefix_tokens': sum((trace['prefix_tokens'] for trace in all_traces)), 'total_request_prompt_tokens': sum((event.get('prompt_tokens') or len(base_prompt_ids) + event['used_prefix_tokens'] for event in window_events)), 'total_request_prefix_tokens': sum((event['used_prefix_tokens'] for event in window_events)), **aggregate_question_kv(window_events), 'generation_seconds': generation_seconds, 'actual_generation_seconds': actual_generation_seconds, 'reused_root_generation_seconds': reused_root_generation_seconds, 'reused_root_windows': reused_root_windows, 'wall_seconds': time.perf_counter() - question_start, 'parser_failures': sum((trace['answer'] is None for trace in all_traces)), 'truncations': sum((trace['truncated'] for trace in all_traces)), 'final_prefix_ratio': state_stack[-1].ratio, 'prefixing_disabled': no_more_prefixing and share_prefix, 'recovery': 'on' if recover else 'off', 'early_stop': early_stop, 'early_stop_threshold': early_stop_threshold if early_stop else None, 'esc_stop_rule': esc_stop_rule if early_stop == 'esc' else None, 'early_stopped': early_stopped, 'launched_windows': len(window_events), 'config_fingerprint': hashlib.sha256(repr((method, max_branches, window_size, beta_threshold, ratios, temperature, top_p, top_k, max_tokens, logprobs, presence_penalty, seed, early_stop, early_stop_threshold if early_stop else None, esc_stop_rule if early_stop == 'esc' else None, 'on' if recover else 'off')).encode()).hexdigest()[:12]}
    if return_internal_windows:
        result['_internal_root_windows'] = internal_windows
    return result
