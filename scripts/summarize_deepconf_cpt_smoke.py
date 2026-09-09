#!/usr/bin/env python3
"""Write a combined DeepConf/CPT smoke report JSON (CPU only)."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/26xyw/projects/path-consistency")
sys.path.insert(0, str(ROOT))
from path_consistency.math_eval import (  # noqa: E402
    extract_gsm8k_numeric,
    extract_last_boxed,
    math_equal,
)

OUT = Path("/home/26xyw/data/path-consistency-results/deepconf-cpt")


def _majority(answers: list[str | None]) -> str | None:
    valid = [a for a in answers if a]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def score_pred(text, gold: str) -> tuple[str | None, bool]:
    boxed = extract_last_boxed(text) if text else None
    if not boxed:
        boxed = extract_gsm8k_numeric(text)
    return boxed, math_equal(boxed, gold)


def summarize_cpt(results_dir: Path, subset_path: Path) -> dict:
    golds = []
    if subset_path.exists():
        with subset_path.open() as handle:
            for line in handle:
                if line.strip():
                    golds.append(json.loads(line))
    files = sorted(results_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 10**9)
    files = [p for p in files if p.stem.isdigit()]
    per = []
    total_tokens = 0
    for path in files:
        data = json.loads(path.read_text())
        idx = int(path.stem)
        gold = golds[idx]["answer"] if idx < len(golds) else None
        comps = data.get("completions") or []
        preds = []
        toks = 0
        n_think = 0
        for c in comps:
            text = c.get("text") or ""
            reason = c.get("reasoning_content") or ""
            if reason:
                n_think += 1
            pred, ok = score_pred(text, gold or "")
            preds.append(pred)
            toks += int(c.get("tokens") or 0)
        total_tokens += toks
        maj = _majority(preds)
        per.append(
            {
                "idx": idx,
                "gold": gold,
                "n_completions": len(comps),
                "n_with_thinking": n_think,
                "tokens": toks,
                "majority": maj,
                "majority_correct": math_equal(maj, gold) if gold is not None else False,
            }
        )
    n = len(per)
    return {
        "status": "complete" if n else "missing",
        "n_questions": n,
        "majority_accuracy": (
            sum(1 for r in per if r["majority_correct"]) / n if n else None
        ),
        "mean_tokens": total_tokens / n if n else None,
        "thinking_spans": sum(r["n_with_thinking"] for r in per),
        "per_question": per,
        "results_dir": str(results_dir),
        "fair_comparison": False,
        "label": "CPT-adapted-llama (official qwen/math.py; no thinking switch)",
    }


def main() -> None:
    dc_summaries = sorted(OUT.glob("deepconf_llama31_gsm8k_smoke*.summary.json"))
    dc = json.loads(dc_summaries[-1].read_text()) if dc_summaries else {}
    subset = OUT / f"{dc.get('tag', 'deepconf_llama31_gsm8k_smoke10')}_subset.jsonl"
    cpt_dir = OUT / "cpt_llama31_gsm8k_smoke10" / "results"
    cpt_exit = None
    exit_path = OUT / "cpt_exit_code.txt"
    if exit_path.exists():
        cpt_exit = exit_path.read_text().strip()
    blocked = OUT / "cpt_blocked.json"
    if blocked.exists():
        cpt = json.loads(blocked.read_text())
        cpt["fair_comparison"] = False
    elif cpt_dir.exists():
        cpt = summarize_cpt(cpt_dir, subset)
        cpt["exit_code"] = cpt_exit
    else:
        cpt = {
            "status": "blocked",
            "exit_code": cpt_exit,
            "fair_comparison": False,
            "reason": "no CPT result directory",
        }
    report = {
        "task": "deepconf_cpt_llama31_gsm8k_smoke",
        "deepconf": dc,
        "cpt": cpt,
        "cpt_gap": {
            "official_entry": "code/qwen/math.py",
            "official_models": "Qwen (enable_thinking=True, </think> chunk split) or GPT-OSS",
            "llama": (
                "Llama-3.1-8B-Instruct has no thinking switch. "
                "apply_chat_template(enable_thinking=True) is a no-op. "
                "Workers never emit </think>, so the official think-truncate / "
                "finalize path never runs. Blackboard writes depend on the model "
                "following a Qwen-oriented BB_WRITE protocol. This is not a fair "
                "CPT comparison and must not be used as matched-budget evidence."
            ),
            "next": "Re-run official CPT on Qwen3-30B-A3B after the MATH-500 4-GPU window.",
        },
    }
    dest = OUT / "smoke_llama31_gsm8k.json"
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])
    print("wrote", dest)


if __name__ == "__main__":
    main()
