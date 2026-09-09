#!/usr/bin/env python3
"""Single-GPU DeepConf smoke on Llama-3.1-8B-Instruct × GSM8K subset.

Uses the official DeepConf voting / confidence code under
``~/data/path-consistency-baselines/deepconf`` and the vLLM 0.8.5 adapter in
``path_consistency/deepconf_vllm08.py``. Does not upgrade the PC conda env.

CUDA_VISIBLE_DEVICES is set from ``--gpu`` before importing torch/vLLM.
GSM8K is read from JSONL (no pyarrow / datasets).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

TZ = timezone(timedelta(hours=8))


def _set_cuda_visible_devices() -> None:
    for index, value in enumerate(sys.argv):
        if value == "--gpu" and index + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[index + 1]
            return
        if value.startswith("--gpu="):
            os.environ["CUDA_VISIBLE_DEVICES"] = value.split("=", 1)[1]
            return


_set_cuda_visible_devices()
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/data/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from vllm import SamplingParams  # noqa: E402

from path_consistency.deepconf_vllm08 import (  # noqa: E402
    DeepThinkLLMv08,
    compact_trace,
    extract_answer,
)
from path_consistency.math_eval import (  # noqa: E402
    extract_gsm8k_numeric,
    format_gsm8k_gold,
    math_equal,
)
from path_consistency.math_protocol import MATH_INSTRUCTION  # noqa: E402

MODEL_DEFAULT = (
    "/home/26xyw/data/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/"
    "snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)
DATA_DEFAULT = "/home/26xyw/data/Datasets/gsm8k_test.jsonl"
OUT_DEFAULT = "/home/26xyw/data/path-consistency-results/deepconf-cpt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--dataset", default=DATA_DEFAULT)
    parser.add_argument("--outdir", default=OUT_DEFAULT)
    parser.add_argument("--tag", default="deepconf_llama31_gsm8k_smoke16")
    parser.add_argument("--max-questions", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=2027)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["online", "offline"], default="online")
    parser.add_argument("--warmup-traces", type=int, default=8)
    parser.add_argument("--total-budget", type=int, default=16)
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--confidence-percentile", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--logprobs", type=int, default=20)
    return parser.parse_args()


def load_subset(path: str, n: int, seed: int) -> list[dict]:
    rows = []
    with open(path) as handle:
        for source_row, line in enumerate(handle):
            if not line.strip():
                continue
            rec = json.loads(line)
            rec["_source_row"] = source_row
            rows.append(rec)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def write_cpt_jsonl(rows: list[dict], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as handle:
        for rec in rows:
            handle.write(
                json.dumps(
                    {
                        "question_id": rec["_source_row"],
                        "question": rec["question"],
                        "answer": format_gsm8k_gold(rec.get("target")),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_prompt(tokenizer, question: str) -> str:
    content = f"{question.strip()}\n\n{MATH_INSTRUCTION}"
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def score_answer(pred, gold: str) -> bool:
    if pred is None:
        return False
    boxed = extract_answer(str(pred)) if "boxed" in str(pred) else str(pred)
    if boxed is None or boxed == "":
        boxed = extract_gsm8k_numeric(str(pred))
    return math_equal(boxed, gold)


def compact_output(result) -> dict:
    traces = [compact_trace(t) for t in (result.all_traces or [])]
    return {
        "final_answer": result.final_answer,
        "voted_answer": result.voted_answer,
        "voting_results": result.voting_results,
        "conf_bar": result.conf_bar,
        "mode": result.mode,
        "total_tokens": result.total_tokens,
        "warmup_tokens": result.warmup_tokens,
        "final_tokens": result.final_tokens,
        "generation_time": result.generation_time,
        "warmup_gen_time": result.warmup_gen_time,
        "final_gen_time": result.final_gen_time,
        "total_time": result.total_time,
        "total_traces_count": result.total_traces_count,
        "config": result.config,
        "n_early_stopped": sum(
            1
            for t in result.all_traces or []
            if t.get("stop_reason") == "gconf_threshold"
        ),
        "traces": traces,
    }


def already_done(jsonl_path: Path) -> set[int]:
    done = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            done.add(int(rec["source_row"]))
    return done


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = outdir / f"{args.tag}.jsonl"
    summary_path = outdir / f"{args.tag}.summary.json"
    subset_path = outdir / f"{args.tag}_subset.jsonl"

    rows = load_subset(args.dataset, args.max_questions, args.sample_seed)
    write_cpt_jsonl(rows, subset_path)
    golds = [format_gsm8k_gold(r.get("target")) for r in rows]

    wall_start = time.time()
    engine = DeepThinkLLMv08(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_logprobs=max(20, args.logprobs),
        max_num_seqs=max(args.total_budget, args.budget, 8),
    )
    probe_prompt = build_prompt(engine.tokenizer, "What is 1+1?")
    probe = engine.probe_request_logits_processor(probe_prompt, n=4)
    print("logits_processor_probe", json.dumps(probe, ensure_ascii=False))
    mode = args.mode
    if mode == "online" and not probe["ok"]:
        print("Refusing DeepConf-online; falling back to DeepConf-offline")
        mode = "offline"

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        logprobs=args.logprobs,
    )

    done = already_done(jsonl_path)
    records = []
    if jsonl_path.exists():
        with jsonl_path.open() as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))

    with jsonl_path.open("a") as handle:
        for idx, rec in enumerate(rows):
            source_row = int(rec["_source_row"])
            if source_row in done:
                print(f"skip source_row={source_row}")
                continue
            prompt = build_prompt(engine.tokenizer, rec["question"])
            q_start = time.time()
            result = engine.deepthink(
                prompt=prompt,
                mode=mode,
                warmup_traces=args.warmup_traces,
                total_budget=args.total_budget,
                confidence_percentile=args.confidence_percentile,
                budget=args.budget,
                window_size=args.window_size,
                sampling_params=sampling,
                compute_multiple_voting=True,
                base_seed=args.seed * 1_000_003 + source_row,
                allow_online=(mode == "online"),
            )
            gold = golds[idx]
            voting_eval = {}
            for name, payload in (result.voting_results or {}).items():
                if not payload:
                    voting_eval[name] = {"answer": None, "correct": False}
                    continue
                ans = payload.get("answer")
                voting_eval[name] = {
                    "answer": ans,
                    "correct": score_answer(ans, gold),
                    "num_votes": payload.get("num_votes"),
                    "confidence": payload.get("confidence"),
                }
            row_out = {
                "source_row": source_row,
                "question": rec["question"],
                "gold": gold,
                "label": (
                    "DeepConf-online"
                    if mode == "online"
                    else "DeepConf-offline"
                ),
                "question_seconds": time.time() - q_start,
                "final_correct": score_answer(result.final_answer, gold),
                "voting_eval": voting_eval,
                "result": compact_output(result),
            }
            handle.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            handle.flush()
            records.append(row_out)
            print(
                f"q{idx} row={source_row} mode={mode} "
                f"final={result.final_answer!r} gold={gold} "
                f"ok={row_out['final_correct']} tok={result.total_tokens} "
                f"t={row_out['question_seconds']:.1f}s"
            )

    n = len(records)
    method_names = sorted(
        {name for rec in records for name in rec.get("voting_eval", {})}
    )
    acc = {
        name: sum(
            1 for rec in records if rec.get("voting_eval", {}).get(name, {}).get("correct")
        )
        / n
        if n
        else None
        for name in method_names
    }
    summary = {
        "tag": args.tag,
        "status": "complete",
        "label": "DeepConf-online" if mode == "online" else "DeepConf-offline",
        "official_vs_adapted": (
            "official DeepConf control flow + voting; "
            "vLLM 0.8.5 SamplingParams.logits_processors adapter "
            "(not vLLM 0.10.2 WrappedPerReqLogitsProcessor)"
            if mode == "online"
            else "official offline voting on full traces; no online EOS"
        ),
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "n_questions": n,
        "source_rows": [r["source_row"] for r in records],
        "subset_jsonl": str(subset_path),
        "jsonl": str(jsonl_path),
        "model": args.model,
        "flags": vars(args),
        "probe": probe,
        "mode": mode,
        "final_accuracy": sum(1 for r in records if r.get("final_correct")) / n if n else None,
        "voting_accuracy": acc,
        "mean_tokens": (
            sum(r["result"]["total_tokens"] for r in records) / n if n else None
        ),
        "mean_question_seconds": (
            sum(r["question_seconds"] for r in records) / n if n else None
        ),
        "wall_seconds": time.time() - wall_start,
        "early_stopped_traces": sum(
            r["result"].get("n_early_stopped") or 0 for r in records
        ),
        "note_window": (
            "Official LGC window is 2048 tokens. GSM8K traces are usually "
            "shorter, so online EOS often never fires; token saving is not "
            "the paper AIME setting."
        ),
        "finished_at": datetime.now(TZ).isoformat(),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
