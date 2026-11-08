#!/usr/bin/env python3
"""Summarize official CPT qwen/math.py outputs on GSM8K (CPU only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/home/26xyw/projects/path-consistency")
sys.path.insert(0, str(ROOT))
from path_consistency.math_eval import (  # noqa: E402
    extract_gsm8k_numeric,
    extract_last_boxed,
    grade_answer,
    majority_vote,
)


def _pred_from_completion(comp: dict) -> str | None:
    text = comp.get("text") or ""
    boxed = extract_last_boxed(text) if text else None
    return boxed or extract_gsm8k_numeric(text)


def _tokens_for_question(data: dict, bb: dict | None) -> int:
    flops = 0
    if isinstance(bb, dict):
        for run in bb.get("runs") or []:
            summary = run.get("flops_trace_summary") or {}
            flops += int(summary.get("output_tokens") or 0)
    if flops:
        return flops
    return sum(int(comp.get("tokens") or 0) for comp in data.get("completions") or [])


def summarize_dir(results_dir: Path) -> dict:
    bb_dir = results_dir.parent / f"{results_dir.name}_bb"
    files = sorted(
        (path for path in results_dir.glob("*.json") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    per = []
    for path in files:
        data = json.loads(path.read_text())
        bb = None
        bb_path = bb_dir / f"{path.stem}_bb.json"
        if bb_path.exists():
            try:
                bb = json.loads(bb_path.read_text())
            except json.JSONDecodeError:
                bb = None
        gold = data.get("answer")
        comps = data.get("completions") or []
        preds = [_pred_from_completion(comp) for comp in comps]
        maj = majority_vote(preds)
        latency = (data.get("latency_summary_sec") or {}).get("total_latency_sec")
        per.append(
            {
                "index": int(path.stem),
                "question_id": data.get("question_id"),
                "gold": gold,
                "n_completions": len(comps),
                "majority": maj,
                "correct": bool(grade_answer("gsm8k", maj, gold)),
                "tokens": _tokens_for_question(data, bb),
                "latency_s": float(latency or 0.0),
            }
        )
    n = len(per)
    correct = sum(int(row["correct"]) for row in per)
    return {
        "status": "complete" if n >= 1319 else ("running" if n else "missing"),
        "n_questions": n,
        "planned": 1319,
        "correct": correct,
        "accuracy": (correct / n) if n else None,
        "tok_per_q": (sum(row["tokens"] for row in per) / n) if n else None,
        "latency_s": (sum(row["latency_s"] for row in per) / n) if n else None,
        "results_dir": str(results_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="/home/26xyw/data/path-consistency-results/deepconf-cpt/cpt_llama31_gsm8k_n32/results",
    )
    parser.add_argument(
        "--out",
        default="/home/26xyw/data/path-consistency-results/deepconf-cpt/cpt_llama31_gsm8k_n32.summary.json",
    )
    args = parser.parse_args()
    summary = summarize_dir(Path(args.results))
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
