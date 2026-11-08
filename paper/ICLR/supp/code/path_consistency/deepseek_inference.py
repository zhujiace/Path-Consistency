from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from .deepconf_offline import lowest_group_confidence, token_confidence
from .deepseek_api import DeepSeekClient, assistant_prefix_message, continuation_text
from .math_eval import canonicalize_vote_answer, extract_last_boxed
from .math_protocol import decode_token_pieces, instruction_for_task, run_sc_pc_question, select_shared_prefix

def _step_from_item(item: dict) -> dict:
    top = {}
    for row in item.get('top_logprobs') or []:
        token = row.get('token')
        if token is None:
            continue
        try:
            top[token] = float(row.get('logprob'))
        except (TypeError, ValueError):
            continue
    token = item.get('token')
    if token is not None:
        try:
            top[token] = float(item.get('logprob'))
        except (TypeError, ValueError):
            pass
    return top

def parse_logprob_channel(items: list[dict]) -> tuple[list[str], list[Optional[float]], list[Optional[float]]]:
    tokens = []
    sampled = []
    confidences = []
    for item in items or []:
        token = item.get('token')
        if token is None:
            continue
        tokens.append(str(token))
        try:
            sampled.append(float(item.get('logprob')))
        except (TypeError, ValueError):
            sampled.append(None)
        confidences.append(token_confidence(_step_from_item(item)))
    return (tokens, sampled, confidences)

def reasoning_span_length(tokens: list[str], reasoning_text: str, content_text: str) -> int:
    haystack = (reasoning_text or '') + ('\n' + content_text if content_text else '')
    marker_positions = [pos for marker in ('\\boxed', 'Final Answer', 'final answer') if (pos := haystack.find(marker)) >= 0]
    if not marker_positions or not tokens:
        return len(tokens)
    marker_position = min(marker_positions)
    if marker_position >= len(reasoning_text or ''):
        return len(tokens)
    (low, high) = (0, len(tokens))
    while low < high:
        middle = (low + high) // 2
        prefix = decode_token_pieces(tokens[:middle])
        if len(prefix) >= marker_position:
            high = middle
        else:
            low = middle + 1
    return low

class DeepSeekMathInference:

    def __init__(self, client: DeepSeekClient, selector: str='blc', prefix_variant: str='reasoning_content', reasoning_effort: str='high', want_logprobs: bool=True, logprob_top_k: int=20, task: str='math', selection_seed: int=0):
        self.client = client
        self.selector = selector
        self.prefix_variant = prefix_variant
        self.reasoning_effort = reasoning_effort
        self.want_logprobs = want_logprobs
        self.logprob_top_k = logprob_top_k
        self.task = task
        self.selection_seed = selection_seed
        self.model_type = 'deepseek-v4-flash'
        self.current_prompt = ''
        self.tokenizer_init_seconds = 0.0
        self.engine_init_seconds = 0.0

    def build_prompt(self, problem: str) -> tuple[str, list]:
        content = f'{problem.strip()}\n\n{instruction_for_task(self.task)}'
        self.current_prompt = content
        return (content, [content])

    def _one_branch(self, prompt: str, prefix_ids: list, max_tokens: int, logprobs: int, request_tag: str) -> dict:
        prefix_text = decode_token_pieces(prefix_ids) if prefix_ids else ''
        messages = [{'role': 'user', 'content': prompt}]
        if prefix_text:
            messages.append(assistant_prefix_message(prefix_text, self.prefix_variant))
        started = time.perf_counter()
        response = self.client.complete(messages, max_tokens=max_tokens, reasoning_effort=self.reasoning_effort, thinking=True, logprobs=self.want_logprobs and logprobs > 0, top_logprobs=min(self.logprob_top_k, max(logprobs, 0) or self.logprob_top_k), request_tag=request_tag)
        elapsed = time.perf_counter() - started
        reasoning = continuation_text(prefix_text, response.get('reasoning_content') or '')
        content = response.get('content') or ''
        full_reasoning = prefix_text + reasoning
        full_text = full_reasoning + ('\n' + content if content else '')
        (new_tokens, sampled_new, conf_new) = parse_logprob_channel(response.get('reasoning_logprobs') or [])
        if prefix_text and new_tokens:
            joined = decode_token_pieces(new_tokens)
            if joined.startswith(prefix_text):
                cut = 0
                acc = ''
                for (index, token) in enumerate(new_tokens):
                    acc += token
                    if acc == prefix_text:
                        cut = index + 1
                        break
                    if len(acc) >= len(prefix_text):
                        cut = index + 1
                        break
                new_tokens = new_tokens[cut:]
                sampled_new = sampled_new[cut:]
                conf_new = conf_new[cut:]
        if not new_tokens:
            new_tokens = list(reasoning) if reasoning else []
            sampled_new = [None] * len(new_tokens)
            conf_new = [None] * len(new_tokens)
        full_ids = list(prefix_ids) + list(new_tokens)
        sampled = [None] * len(prefix_ids) + sampled_new
        confidences = [None] * len(prefix_ids) + conf_new
        raw_answer = extract_last_boxed(full_text) or extract_last_boxed(content)
        usage = response.get('usage') or {}
        details = usage.get('completion_tokens_details') or {}
        generated = int(details.get('reasoning_tokens') or usage.get('completion_tokens') or len(new_tokens))
        return {'full_token_ids': full_ids, 'new_token_ids': new_tokens, 'sampled_logprobs': sampled, 'token_confidence': confidences, 'full_text': full_text, 'suffix_text': reasoning + ('\n' + content if content else ''), 'reasoning_content': full_reasoning, 'content': content, 'raw_answer': raw_answer, 'answer': canonicalize_vote_answer(self.task, raw_answer), 'reasoning_tokens': reasoning_span_length(full_ids, full_reasoning, content), 'finish_reason': response.get('finish_reason'), 'stop_reason': None, 'generated_tokens': generated, 'prompt_tokens': int(usage.get('prompt_tokens') or 0), 'prefix_tokens': len(prefix_ids), 'truncated': response.get('finish_reason') == 'length', 'lgc_2048': lowest_group_confidence([value for value in confidences if value is not None], 2048) if any((value is not None for value in confidences)) else 0.0, 'api_id': response.get('id'), 'api_model': response.get('model'), 'system_fingerprint': response.get('system_fingerprint'), 'usage': usage, 'logprobs_available': bool(response.get('reasoning_logprobs')), 'branch_elapsed_s': elapsed, 'attempt': response.get('attempt', 0)}

    def _generate_window(self, base_prompt_ids: list, prefix_ids: list, window_size: int, sampling_seed: int, temperature: float, top_p: float, top_k: int, max_tokens: int, logprobs: int, presence_penalty: float) -> tuple[list[dict], float]:
        del temperature, top_p, top_k, presence_penalty
        prompt = self.current_prompt
        if base_prompt_ids and isinstance(base_prompt_ids[0], str):
            prompt = base_prompt_ids[0]
        start = time.perf_counter()
        traces = [None] * window_size
        with ThreadPoolExecutor(max_workers=window_size) as pool:
            futures = {pool.submit(self._one_branch, prompt, prefix_ids, max_tokens, logprobs, f'seed{sampling_seed}:b{index}'): index for index in range(window_size)}
            for future in as_completed(futures):
                traces[futures[future]] = future.result()
        return (traces, time.perf_counter() - start)

    def _select_prefix(self, traces: list[dict], modal_answer: str, current_prefix_ids: list, next_ratio: float) -> Optional[dict]:
        return select_shared_prefix(traces, modal_answer, current_prefix_ids, next_ratio, decode_token_pieces, selector=self.selector, selection_seed=self.selection_seed)

    def run_question(self, question_id: str, problem: str, method: str, **kwargs) -> dict:
        result = run_sc_pc_question(self, question_id=question_id, problem=problem, method=method, **kwargs)
        result['selector'] = self.selector
        result['selection_seed'] = self.selection_seed
        result['prefix_variant'] = self.prefix_variant
        totals = getattr(self.client, 'usage_totals', None)
        result['api_usage_totals'] = dict(totals) if isinstance(totals, dict) else {}
        return result
