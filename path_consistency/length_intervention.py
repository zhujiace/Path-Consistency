"""Matched-history prefix-length intervention (paper §5.3 / appendix I).

Freeze window-0 SC traces and one BLC source, then resample one window at
r in {0, 0.2, 0.4, 0.6, 0.8}. Not a test-time PVC controller.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from .math_eval import grade_answer, majority_vote
from .math_protocol import (
    decode_token_pieces,
    gate_snapshot,
    safe_ratio_endpoint,
    select_shared_prefix,
)

LENGTH_RATIOS = (0.0, 0.2, 0.4, 0.6, 0.8)
NONZERO_RATIOS = (0.2, 0.4, 0.6, 0.8)


def window_traces(windows: list[dict] | None, index: int) -> list[dict]:
    if not windows:
        return []
    for window in windows:
        if int(window.get("window") or 0) == index:
            return list(window.get("traces") or [])
    return []


def compact_trace(trace: dict) -> dict:
    return {
        "branch_id": trace.get("branch_id"),
        "answer": trace.get("answer"),
        "raw_answer": trace.get("raw_answer"),
        "generated_tokens": trace.get("generated_tokens") or 0,
        "prompt_tokens": trace.get("prompt_tokens") or 0,
        "prefix_tokens": trace.get("prefix_tokens") or 0,
        "truncated": bool(trace.get("truncated")),
        "logprobs_available": bool(trace.get("logprobs_available")),
        "window_seconds": trace.get("branch_elapsed_s") or trace.get("window_seconds"),
        "api_id": trace.get("api_id"),
    }


def _endpoint_record(source: dict, ratio: float) -> Optional[dict]:
    ids = list(source.get("full_token_ids") or [])[: int(source.get("reasoning_tokens") or 0)]
    if not ids:
        return None
    endpoint = safe_ratio_endpoint(ids, ratio, 0, decode_token_pieces)
    if endpoint is None:
        return None
    prefix_ids = ids[:endpoint]
    prefix_text = decode_token_pieces(prefix_ids)
    return {
        "ratio": ratio,
        "endpoint": endpoint,
        "actual_ratio": endpoint / len(ids),
        "prefix_tokens": endpoint,
        "prefix_sha256": hashlib.sha256(prefix_text.encode()).hexdigest()[:12],
    }


def build_question_snapshot(
    *,
    question_id: str,
    windows: list[dict],
    gold: str,
    task: str = "math",
    beta_threshold: float = 0.8,
    selection_seed: int = 0,
) -> dict[str, Any]:
    root = window_traces(windows, 0)
    recanonicalized = list(root)
    answers = [trace.get("answer") for trace in recanonicalized]
    gate = gate_snapshot(answers, beta_threshold)
    history_vote = majority_vote(answers)
    tail_answers = []
    for index in range(2, 8):
        tail_answers.extend(trace.get("answer") for trace in window_traces(windows, index))
    snapshot: dict[str, Any] = {
        "question_id": question_id,
        "n_root_traces": len(recanonicalized),
        "n_tail_traces": len(tail_answers),
        "gate": gate,
        "history_answers": answers,
        "history_vote": history_vote,
        "history_correct": grade_answer(task, history_vote, gold),
        "tail_answers": tail_answers,
        "source_id": None,
        "source_reasoning_tokens": None,
        "endpoints": {},
        "missing_ratios": [],
        "common_support": False,
        "reason": None,
    }
    if len(recanonicalized) < 4:
        snapshot["reason"] = "short_root"
        snapshot["missing_ratios"] = list(NONZERO_RATIOS)
        return snapshot
    if not gate.get("modal_answer"):
        snapshot["reason"] = "no_modal"
        snapshot["missing_ratios"] = list(NONZERO_RATIOS)
        return snapshot
    selected = select_shared_prefix(
        recanonicalized,
        gate["modal_answer"],
        [],
        0.2,
        decode_token_pieces,
        selector="blc",
        selection_seed=selection_seed,
    )
    if not selected:
        snapshot["reason"] = "no_source_at_20"
        snapshot["missing_ratios"] = list(NONZERO_RATIOS)
        return snapshot
    source = next(
        (
            trace
            for trace in recanonicalized
            if trace.get("branch_id") == selected["source_id"]
        ),
        None,
    )
    if source is None:
        snapshot["reason"] = "source_missing_in_root"
        snapshot["missing_ratios"] = list(NONZERO_RATIOS)
        return snapshot
    snapshot["source_id"] = selected["source_id"]
    snapshot["source_reasoning_tokens"] = source.get("reasoning_tokens")
    snapshot["source_generated_tokens"] = source.get("generated_tokens")
    snapshot["source_blc_score"] = selected.get("blc_score")
    endpoints = {}
    missing = []
    for ratio in NONZERO_RATIOS:
        record = _endpoint_record(source, ratio)
        if record is None:
            missing.append(ratio)
        else:
            endpoints[str(ratio)] = record
    snapshot["endpoints"] = endpoints
    snapshot["missing_ratios"] = missing
    snapshot["gate_passed"] = bool(gate.get("passed"))
    snapshot["common_support"] = (
        not missing
        and bool(gate.get("passed"))
        and len(tail_answers) == 24
        and source.get("full_token_ids")
    )
    if not snapshot["common_support"] and not snapshot["reason"]:
        if not gate.get("passed"):
            snapshot["reason"] = "gate_failed"
        elif len(tail_answers) != 24:
            snapshot["reason"] = "short_tail"
        elif missing:
            snapshot["reason"] = "partial_endpoints"
    return snapshot


def prefix_ids_for_ratio(source_trace: dict, snapshot: dict, ratio: float) -> list:
    if ratio <= 0:
        return []
    record = (snapshot.get("endpoints") or {}).get(str(ratio)) or (
        snapshot.get("endpoints") or {}
    ).get(f"{ratio:.1f}")
    if not record:
        return []
    ids = list(source_trace.get("full_token_ids") or [])
    return ids[: int(record["endpoint"])]


def final_vote_answers(snapshot: dict, intervened_answers: list) -> list:
    return list(snapshot.get("history_answers") or []) + list(intervened_answers) + list(
        snapshot.get("tail_answers") or []
    )
