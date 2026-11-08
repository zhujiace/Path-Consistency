"""Offline majority vote over the first k SC traces.

A dedicated SC run with B=k, m=4, and the same window seeds is identical to
taking the first k traces of an existing SC-32 JSONL: windows are i.i.d. and
sampling_seed depends only on (seed, window_index). No extra generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from path_consistency.math_eval import grade_answer, majority_vote

DEFAULT_SC_GLOB = (
    "/home/26xyw/data/path-consistency-results/gsm8k/"
    "gsm8k_llama31_b32_mt2048_c08_seed0_shard*_sc.jsonl"
)
DEFAULT_OUT = (
    "/home/26xyw/data/path-consistency-results/gsm8k/reports/"
    "sc_prefix_k_llama31_c08.json"
)


def prefix_k_from_row(row: dict[str, Any], k: int) -> dict[str, Any]:
    traces = list(row.get("traces") or [])
    taken = traces[:k]
    voted = majority_vote(trace.get("answer") for trace in taken)
    gold = row.get("gold")
    task = row.get("dataset_task") or "gsm8k"
    window_ids = {trace.get("window") for trace in taken if trace.get("window") is not None}
    latency = 0.0
    for event in row.get("windows") or []:
        if event.get("window") in window_ids:
            latency += float(event.get("generation_seconds") or 0.0)
    return {
        "k": k,
        "n_traces": len(taken),
        "voted": voted,
        "correct": bool(grade_answer(task, voted, gold)),
        "generated_tokens": sum(int(trace.get("generated_tokens") or 0) for trace in taken),
        "generation_seconds": latency,
        "n_windows": len(window_ids),
    }


def iter_jsonl(paths: Iterable[Path]):
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def summarize_prefix_k(
    rows: Iterable[dict[str, Any]],
    ks: list[int],
) -> dict[str, Any]:
    n = 0
    official_sc = 0
    buckets = {
        k: {"correct": 0, "tokens": 0.0, "seconds": 0.0, "short": 0} for k in ks
    }
    question_ids = []
    for row in rows:
        n += 1
        question_ids.append(row.get("question_id"))
        pred = (row.get("predictions") or {}).get("SC") or {}
        if pred.get("correct"):
            official_sc += 1
        for k in ks:
            stats = prefix_k_from_row(row, k)
            buckets[k]["correct"] += int(stats["correct"])
            buckets[k]["tokens"] += stats["generated_tokens"]
            buckets[k]["seconds"] += stats["generation_seconds"]
            if stats["n_traces"] < k:
                buckets[k]["short"] += 1
    by_k = {}
    for k, bucket in buckets.items():
        by_k[str(k)] = {
            "k": k,
            "n": n,
            "correct": bucket["correct"],
            "accuracy": (bucket["correct"] / n) if n else None,
            "tok_per_q": (bucket["tokens"] / n) if n else None,
            "latency_s": (bucket["seconds"] / n) if n else None,
            "n_short": bucket["short"],
        }
    return {
        "n": n,
        "official_sc_correct": official_sc,
        "official_sc_accuracy": (official_sc / n) if n else None,
        "by_k": by_k,
        "question_ids_head": question_ids[:8],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default=DEFAULT_SC_GLOB)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--k",
        default="4,8,12,16,20,24,28,32",
        help="comma-separated prefix lengths",
    )
    args = parser.parse_args()
    ks = [int(part) for part in args.k.split(",") if part.strip()]
    paths = sorted(Path(p) for p in __import__("glob").glob(args.glob))
    if not paths:
        raise SystemExit(f"no SC jsonl matched {args.glob}")
    summary = summarize_prefix_k(iter_jsonl(paths), ks)
    summary["source_glob"] = args.glob
    summary["source_files"] = [str(path) for path in paths]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: summary["by_k"][k] for k in summary["by_k"]}, ensure_ascii=False, indent=2))
    print(f"wrote {out} n={summary['n']} official_sc={summary['official_sc_correct']}")


if __name__ == "__main__":
    main()
