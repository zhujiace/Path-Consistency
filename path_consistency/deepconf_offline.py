"""Offline DeepConf aggregation compatible with the official implementation.

This module performs no generation. It consumes the same completed traces as
SC and implements Lowest Group Confidence (2,048-token window), followed by
confidence filtering and confidence-weighted voting from arXiv:2508.15260.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Optional

import numpy as np


def token_confidence(
    logprob_step: Optional[dict], top_k: int = 20
) -> Optional[float]:
    """Negative mean log-probability of the top-k tokens.

    vLLM may include the sampled token in addition to the requested top-k, so
    sorting and slicing is required for parity with DeepConf's logits-based
    implementation.
    """
    if not logprob_step:
        return None
    values = []
    for item in logprob_step.values():
        value = getattr(item, "logprob", item)
        try:
            numeric = float(value)
            if math.isfinite(numeric):
                values.append(numeric)
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    values = sorted(values, reverse=True)[:top_k]
    return round(-float(np.mean(values)), 3)


def group_confidences(
    confidences: Iterable[float], window_size: int = 2048
) -> list[float]:
    """Return overlapping sliding-window confidence means."""
    values = [
        float(value)
        for value in confidences
        if value is not None and math.isfinite(float(value))
    ]
    if not values:
        return [0.0]
    if len(values) <= window_size:
        return [float(np.mean(values))]
    current = sum(values[:window_size])
    means = [current / window_size]
    for start in range(1, len(values) - window_size + 1):
        current += values[start + window_size - 1] - values[start - 1]
        means.append(current / window_size)
    return means


def lowest_group_confidence(
    confidences: Iterable[float], window_size: int = 2048
) -> float:
    """Return the minimum overlapping group mean used by DeepConf."""
    return float(min(group_confidences(confidences, window_size)))


def bottom_group_confidence(
    confidences: Iterable[float],
    window_size: int = 2048,
    bottom_fraction: float = 0.10,
) -> float:
    """Mean of the lowest-confidence 10% of overlapping groups."""
    means = group_confidences(confidences, window_size)
    keep = max(1, int(len(means) * bottom_fraction))
    return float(np.mean(sorted(means)[:keep]))


def tail_confidence(
    confidences: Iterable[float], tail_tokens: int = 2048
) -> float:
    values = [
        float(value)
        for value in confidences
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.mean(values[-tail_tokens:])) if values else 0.0


def mean_confidence(confidences: Iterable[float]) -> float:
    values = [
        float(value)
        for value in confidences
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.mean(values)) if values else 0.0


def weighted_vote(answers: list[Optional[str]], weights: list[float]) -> Optional[str]:
    totals = defaultdict(float)
    first_position = {}
    for index, (answer, weight) in enumerate(zip(answers, weights)):
        if answer is None:
            continue
        totals[answer] += float(weight)
        first_position.setdefault(answer, index)
    if not totals:
        return None
    return max(totals, key=lambda answer: (totals[answer], -first_position[answer]))


def aggregate_confidence(
    traces: list[dict],
    score_name: str,
    retain_fraction: Optional[float],
    window_size: int = 2048,
) -> dict:
    """Filter by a trace score and confidence-weight retained answers."""
    valid = [trace for trace in traces if trace.get("answer") is not None]
    if not valid:
        return {
            "answer": None,
            "retained": 0,
            "valid_traces": 0,
            "retain_fraction": retain_fraction,
            "score_name": score_name,
        }
    scored = []
    for index, trace in enumerate(valid):
        confidences = trace.get("token_confidence", [])
        if score_name == "lgc":
            score = trace.get(f"lgc_{window_size}")
            if score is None:
                score = lowest_group_confidence(confidences, window_size)
        elif score_name == "bottom10":
            score = bottom_group_confidence(confidences, window_size)
        elif score_name == "tail":
            score = tail_confidence(confidences, window_size)
        elif score_name == "mean":
            score = mean_confidence(confidences)
        else:
            raise ValueError(f"Unknown DeepConf score: {score_name}")
        if score is None or not math.isfinite(float(score)):
            score = lowest_group_confidence(trace.get("token_confidence", []), window_size)
        scored.append((score, index, trace))
    if retain_fraction is None:
        retained = scored
    else:
        threshold = float(
            np.percentile(
                [row[0] for row in scored], (1.0 - retain_fraction) * 100.0
            )
        )
        retained = [row for row in scored if row[0] >= threshold]
    retained = sorted(retained, key=lambda row: (-row[0], row[1]))
    keep = len(retained)
    answers = [row[2]["answer"] for row in retained]
    weights = [row[0] for row in retained]
    return {
        "answer": weighted_vote(answers, weights),
        "retained": keep,
        "valid_traces": len(valid),
        "retain_fraction": retain_fraction,
        "score_name": score_name,
        "mean_lgc": float(np.mean(weights)),
        "min_lgc": float(np.min(weights)),
        "max_lgc": float(np.max(weights)),
        "retained_branch_ids": [row[2].get("branch_id") for row in retained],
    }


def aggregate_lgc(
    traces: list[dict], retain_fraction: float, window_size: int = 2048
) -> dict:
    """Backward-compatible LGC aggregation helper."""
    return aggregate_confidence(traces, "lgc", retain_fraction, window_size)


def aggregate_deepconf_offline(traces: list[dict], window_size: int = 2048) -> dict:
    """Return the paper's offline confidence-weighting/filtering policies."""
    return {
        "offline_mean": aggregate_confidence(traces, "mean", None, window_size),
        "offline_bottom10_90": aggregate_confidence(
            traces, "bottom10", 0.90, window_size
        ),
        "offline_bottom10_10": aggregate_confidence(
            traces, "bottom10", 0.10, window_size
        ),
        "offline_tail_90": aggregate_confidence(traces, "tail", 0.90, window_size),
        "offline_tail_10": aggregate_confidence(traces, "tail", 0.10, window_size),
        "offline_lgc_90": aggregate_confidence(traces, "lgc", 0.90, window_size),
        "offline_lgc_10": aggregate_confidence(traces, "lgc", 0.10, window_size),
    }
