from __future__ import annotations
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional
import httpx
DEFAULT_ENV_PATH = Path('.env')
DEFAULT_BETA_URL = 'https://api.deepseek.com/beta'
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 550}

class DeepSeekAPIError(RuntimeError):

    def __init__(self, message: str, status_code: Optional[int]=None, body: str=''):
        super().__init__(message)
        self.status_code = status_code
        self.body = body

def load_deepseek_env(path: Path=DEFAULT_ENV_PATH) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith('#') or '=' not in stripped:
                continue
            (key, value) = stripped.split('=', 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    api_key = os.environ.get('DEEPSEEK_API_KEY') or values.get('DEEPSEEK_API_KEY')
    if not api_key:
        raise DeepSeekAPIError('DEEPSEEK_API_KEY is missing')
    base_url = os.environ.get('DEEPSEEK_BASE_URL') or values.get('DEEPSEEK_BASE_URL') or DEFAULT_BETA_URL
    if not base_url.rstrip('/').endswith('/beta'):
        base_url = DEFAULT_BETA_URL
    model = os.environ.get('DEEPSEEK_MODEL') or values.get('DEEPSEEK_MODEL') or 'deepseek-v4-flash'
    return {'api_key': api_key, 'base_url': base_url.rstrip('/'), 'model': model}

def _redact(text: str) -> str:
    if not text:
        return text
    lowered = text.lower()
    if 'sk-' in lowered or 'bearer ' in lowered:
        return '<redacted>'
    return text[:800]

def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]

def _delta_text(delta: dict, field: str) -> str:
    value = delta.get(field)
    return value if isinstance(value, str) else ''

def _extend_logprob_items(bucket: list, payload) -> None:
    if not payload:
        return
    if isinstance(payload, dict):
        items = payload.get('content') or payload.get('reasoning_content') or []
        if items and isinstance(items, list) and isinstance(items[0], dict):
            bucket.extend(items)
            return
        if 'token' in payload:
            bucket.append(payload)
            return
    bucket.extend(_as_list(payload))

class DeepSeekClient:

    def __init__(self, env: Optional[dict[str, str]]=None, timeout_s: float=900.0, max_retries: int=5, max_inflight: int=32, user_id: str='pc-math500', on_event: Optional[Callable[[dict], None]]=None):
        cfg = env or load_deepseek_env()
        self.api_key = cfg['api_key']
        self.base_url = cfg['base_url']
        self.model = cfg['model']
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.user_id = user_id
        self.on_event = on_event
        self._sema = threading.BoundedSemaphore(max_inflight)
        self._lock = threading.Lock()
        self.usage_totals = {'requests': 0, 'retries': 0, 'late_duplicates': 0, 'prompt_tokens': 0, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 0, 'completion_tokens': 0, 'reasoning_tokens': 0, 'total_tokens': 0}
        self._http = httpx.Client(timeout=httpx.Timeout(timeout_s, connect=30.0), headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}, follow_redirects=False)

    def close(self) -> None:
        self._http.close()

    def _emit(self, event: dict) -> None:
        if self.on_event is None:
            return
        safe = dict(event)
        safe.pop('api_key', None)
        self.on_event(safe)

    def _accumulate_usage(self, usage: dict, retries: int) -> None:
        details = usage.get('completion_tokens_details') or {}
        with self._lock:
            self.usage_totals['requests'] += 1
            self.usage_totals['retries'] += retries
            self.usage_totals['prompt_tokens'] += int(usage.get('prompt_tokens') or 0)
            self.usage_totals['prompt_cache_hit_tokens'] += int(usage.get('prompt_cache_hit_tokens') or 0)
            self.usage_totals['prompt_cache_miss_tokens'] += int(usage.get('prompt_cache_miss_tokens') or 0)
            self.usage_totals['completion_tokens'] += int(usage.get('completion_tokens') or 0)
            self.usage_totals['reasoning_tokens'] += int(details.get('reasoning_tokens') or 0)
            self.usage_totals['total_tokens'] += int(usage.get('total_tokens') or 0)

    def complete(self, messages: list[dict], *, max_tokens: int=32768, reasoning_effort: str='high', thinking: bool=True, logprobs: bool=True, top_logprobs: int=20, stream: bool=True, extra: Optional[dict]=None, request_tag: Optional[str]=None) -> dict:
        payload: dict[str, Any] = {'model': self.model, 'messages': messages, 'max_tokens': max_tokens, 'reasoning_effort': reasoning_effort, 'thinking': {'type': 'enabled' if thinking else 'disabled'}, 'user_id': self.user_id, 'stream': stream}
        if logprobs:
            payload['logprobs'] = True
            payload['top_logprobs'] = top_logprobs
        if stream:
            payload['stream_options'] = {'include_usage': True}
        if extra:
            payload.update(extra)
        last_error: Optional[Exception] = None
        winner: Optional[dict] = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                with self._sema:
                    result = self._complete_stream(payload) if stream else self._complete_json(payload)
                result['attempt'] = attempt
                result['elapsed_s'] = time.perf_counter() - started
                result['request_tag'] = request_tag
                if winner is not None:
                    with self._lock:
                        self.usage_totals['late_duplicates'] += 1
                    self._emit({'event': 'late_duplicate_discarded', 'request_tag': request_tag, 'response_id': result.get('id')})
                    break
                winner = result
                self._accumulate_usage(result.get('usage') or {}, attempt)
                self._emit({'event': 'ok', 'request_tag': request_tag, 'attempt': attempt, 'status_code': 200, 'response_id': result.get('id'), 'model': result.get('model'), 'system_fingerprint': result.get('system_fingerprint'), 'usage': result.get('usage'), 'elapsed_s': result['elapsed_s'], 'finish_reason': result.get('finish_reason')})
                return winner
            except DeepSeekAPIError as error:
                last_error = error
                retry = error.status_code in RETRYABLE_STATUS and attempt < self.max_retries
                self._emit({'event': 'error', 'request_tag': request_tag, 'attempt': attempt, 'status_code': error.status_code, 'retry': retry, 'message': str(error)})
                if not retry:
                    raise
                time.sleep(min(2 ** attempt, 30))
            except (httpx.TimeoutException, httpx.TransportError) as error:
                last_error = error
                retry = attempt < self.max_retries
                self._emit({'event': 'transport_error', 'request_tag': request_tag, 'attempt': attempt, 'retry': retry, 'message': error.__class__.__name__})
                if not retry:
                    raise DeepSeekAPIError(f'transport failed: {error.__class__.__name__}') from error
                time.sleep(min(2 ** attempt, 30))
        raise DeepSeekAPIError(f'exhausted retries: {last_error}') from last_error

    def _complete_json(self, payload: dict) -> dict:
        response = self._http.post(f'{self.base_url}/chat/completions', json=payload)
        body = response.text.lstrip()
        if response.status_code != 200:
            raise DeepSeekAPIError(f'HTTP {response.status_code}', status_code=response.status_code, body=_redact(body))
        data = json.loads(body)
        return self._from_completion_object(data)

    def _complete_stream(self, payload: dict) -> dict:
        reasoning = []
        content = []
        reasoning_logprobs: list[dict] = []
        content_logprobs: list[dict] = []
        finish_reason = None
        response_id = None
        model = None
        fingerprint = None
        usage = {}
        with self._http.stream('POST', f'{self.base_url}/chat/completions', json=payload) as response:
            if response.status_code != 200:
                body = response.read().decode('utf-8', errors='replace')
                raise DeepSeekAPIError(f'HTTP {response.status_code}', status_code=response.status_code, body=_redact(body))
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith(':'):
                    continue
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    break
                chunk = json.loads(data)
                response_id = chunk.get('id') or response_id
                model = chunk.get('model') or model
                fingerprint = chunk.get('system_fingerprint') or fingerprint
                if chunk.get('usage'):
                    usage = chunk['usage']
                for choice in chunk.get('choices') or []:
                    finish_reason = choice.get('finish_reason') or finish_reason
                    delta = choice.get('delta') or {}
                    reasoning.append(_delta_text(delta, 'reasoning_content'))
                    content.append(_delta_text(delta, 'content'))
                    logprobs = choice.get('logprobs') or {}
                    _extend_logprob_items(reasoning_logprobs, logprobs.get('reasoning_content'))
                    _extend_logprob_items(content_logprobs, logprobs.get('content'))
        return {'id': response_id, 'model': model, 'system_fingerprint': fingerprint, 'finish_reason': finish_reason or 'stop', 'reasoning_content': ''.join(reasoning), 'content': ''.join(content), 'reasoning_logprobs': reasoning_logprobs, 'content_logprobs': content_logprobs, 'usage': usage}

    def _from_completion_object(self, data: dict) -> dict:
        choice = (data.get('choices') or [{}])[0]
        message = choice.get('message') or {}
        logprobs = choice.get('logprobs') or {}
        return {'id': data.get('id'), 'model': data.get('model'), 'system_fingerprint': data.get('system_fingerprint'), 'finish_reason': choice.get('finish_reason') or 'stop', 'reasoning_content': message.get('reasoning_content') or '', 'content': message.get('content') or '', 'reasoning_logprobs': _as_list(logprobs.get('reasoning_content')), 'content_logprobs': _as_list(logprobs.get('content')), 'usage': data.get('usage') or {}}

def assistant_prefix_message(prefix_text: str, variant: str='reasoning_content') -> dict:
    if variant == 'reasoning_content':
        return {'role': 'assistant', 'content': '', 'reasoning_content': prefix_text, 'prefix': True}
    if variant == 'reasoning_only':
        return {'role': 'assistant', 'reasoning_content': prefix_text, 'prefix': True}
    if variant == 'content':
        return {'role': 'assistant', 'content': prefix_text, 'prefix': True}
    if variant == 'both':
        return {'role': 'assistant', 'content': prefix_text, 'reasoning_content': prefix_text, 'prefix': True}
    raise ValueError(f'unknown prefix variant: {variant}')

def continuation_text(prefix: str, returned: str) -> str:
    if not returned:
        return ''
    if prefix and returned.startswith(prefix):
        return returned[len(prefix):]
    return returned
