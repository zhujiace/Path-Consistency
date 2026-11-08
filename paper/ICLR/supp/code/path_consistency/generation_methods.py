from __future__ import annotations
from typing import Optional, Sequence
from .confidence import beta_confidence
AC_STOP_THRESHOLD = 0.95
METHOD_ALIASES = {'ac+pc': 'pc_ac', 'ac_pc': 'pc_ac', 'pc+ac': 'pc_ac', 'esc+pc': 'pc_esc', 'esc_pc': 'pc_esc', 'pc+esc': 'pc_esc'}
GENERATION_METHODS: dict[str, dict] = {'sc': {'share_prefix': False, 'early_stop': None, 'root': None, 'label': 'SC'}, 'pc': {'share_prefix': True, 'early_stop': None, 'root': 'sc', 'label': 'PC'}, 'ac': {'share_prefix': False, 'early_stop': 'ac', 'root': None, 'label': 'AC'}, 'esc': {'share_prefix': False, 'early_stop': 'esc', 'root': None, 'label': 'ESC'}, 'pc_ac': {'share_prefix': True, 'early_stop': 'ac', 'root': 'ac', 'label': 'AC+PC'}, 'pc_esc': {'share_prefix': True, 'early_stop': 'esc', 'root': 'esc', 'label': 'ESC+PC'}}

def canonicalize_method(name: str) -> str:
    key = name.strip().lower()
    return METHOD_ALIASES.get(key, key)

def method_spec(name: str) -> dict:
    key = canonicalize_method(name)
    spec = GENERATION_METHODS.get(key)
    if spec is None:
        raise ValueError(f'unknown generation method: {name}')
    return spec

def parse_generation_methods(raw: str | Sequence[str]) -> list[str]:
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(',') if part.strip()]
    else:
        parts = [str(part).strip() for part in raw if str(part).strip()]
    methods: list[str] = []
    seen: set[str] = set()
    for part in parts:
        name = canonicalize_method(part)
        if name not in GENERATION_METHODS:
            raise ValueError(f"unknown generation method {part!r}; allowed: {', '.join(GENERATION_METHODS)}")
        if name not in seen:
            methods.append(name)
            seen.add(name)
    if not methods:
        raise ValueError('no generation methods')
    ordered: list[str] = []
    placed: set[str] = set()
    for name in methods:
        root = GENERATION_METHODS[name]['root']
        if root and root in seen and (root not in placed):
            ordered.append(root)
            placed.add(root)
        if name not in placed:
            ordered.append(name)
            placed.add(name)
    return ordered

def pair_root_method(method: str) -> Optional[str]:
    return method_spec(method)['root']

def is_root_for_any(method: str, methods: Sequence[str]) -> bool:
    return any((pair_root_method(other) == method for other in methods))

def method_label(name: str) -> str:
    return method_spec(name)['label']
ESC_STOP_WINDOW = 'window'
ESC_STOP_TWO_WINDOW_OR_8 = 'two_window_or_8'

def _unanimous_answer(answers: Sequence[Optional[str]]) -> Optional[str]:
    valid = [answer for answer in answers if answer is not None]
    if valid and len(valid) == len(answers) and (len(set(valid)) == 1):
        return valid[0]
    return None

def should_early_stop(kind: Optional[str], window_answers: Sequence[Optional[str]], all_answers: Sequence[Optional[str]], threshold: float=AC_STOP_THRESHOLD, esc_stop_rule: str=ESC_STOP_WINDOW, window_size: int=4) -> bool:
    if kind == 'ac':
        return beta_confidence(list(all_answers)) >= threshold
    if kind == 'esc':
        rule = esc_stop_rule or ESC_STOP_WINDOW
        if rule == ESC_STOP_WINDOW:
            return _unanimous_answer(window_answers) is not None
        if rule == ESC_STOP_TWO_WINDOW_OR_8:
            valid = [answer for answer in all_answers if answer is not None]
            cumulative_8 = len(valid) >= 8 and len(set(valid)) == 1
            consecutive = False
            if window_size > 0 and len(all_answers) >= 2 * window_size:
                prev = list(all_answers)[-2 * window_size:-window_size]
                curr = list(all_answers)[-window_size:]
                left = _unanimous_answer(prev)
                right = _unanimous_answer(curr)
                consecutive = left is not None and left == right
            return consecutive or cumulative_8
        raise ValueError(f'unknown esc_stop_rule: {rule}')
    return False
