#!/usr/bin/env python3
"""PoLR (arXiv 2601.21494) on the shared Llama/Qwen vLLM backend.

Sample N prefixes of length L_p, TF-IDF + agglomerative clustering, expand the
dominant cluster, majority vote. Default hyperparameters match the paper
(L_p=256, cosine average-linkage threshold 1.0, SVD dim=10) with N=32 so the
GSM8K Table 2 comparison uses the same branch budget as SC.

Example:
    /home/26xyw/.conda/envs/PC/bin/python run_polr.py \\
      --gpus 2 --dataset ~/data/Datasets/gsm8k_test.jsonl --tag polr_llama31_gsm8k_n32_lp256_seed0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _set_cuda_visible_devices() -> None:
    for index, value in enumerate(sys.argv):
        if value == "--gpus" and index + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[index + 1]
            return
        if value.startswith("--gpus="):
            os.environ["CUDA_VISIBLE_DEVICES"] = value.split("=", 1)[1]
            return


_set_cuda_visible_devices()
from path_consistency.vllm_nccl import install_cuda122_nccl  # noqa: E402

install_cuda122_nccl()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if "--vllm-v1" not in sys.argv:
    os.environ.setdefault("VLLM_USE_V1", "0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="2")
    parser.add_argument(
        "--model",
        default=(
            "/home/26xyw/data/huggingface/hub/models--meta-llama--"
            "Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
        ),
    )
    parser.add_argument(
        "--dataset", default="/home/26xyw/data/Datasets/gsm8k_test.jsonl"
    )
    parser.add_argument(
        "--outdir",
        default="/home/26xyw/data/path-consistency-results/gsm8k",
    )
    parser.add_argument("--tag", default="polr_llama31_gsm8k_n32_lp256_seed0")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=2027)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-prefixes", type=int, default=32)
    parser.add_argument("--prefix-len", type=int, default=256)
    parser.add_argument("--distance-threshold", type=float, default=1.0)
    parser.add_argument("--feature-dim", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--logprobs", type=int, default=0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--physical-window-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--num-gpu-blocks-override", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--cudagraph-capture-sizes", default="32,16,8,4,2,1")
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--enable-custom-all-reduce", action="store_true")
    parser.add_argument("--allow-cudagraph", action="store_true")
    parser.add_argument("--vllm-v1", action="store_true")
    parser.add_argument("--store-text", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument(
        "--answer-fallback",
        choices=["auto", "none", "gsm8k"],
        default="auto",
    )
    return parser.parse_args()


def _capture_sizes(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(value) for value in raw.split(",") if value.strip()]


def _strip_trace_payload(trace: dict, store_text: bool) -> dict:
    """Drop token-id arrays from the JSONL; they are only needed during decode."""
    dropped = dict(trace)
    dropped.pop("full_token_ids", None)
    dropped.pop("new_token_ids", None)
    dropped.pop("sampled_logprobs", None)
    dropped.pop("token_confidence", None)
    if not store_text:
        dropped.pop("full_text", None)
        dropped.pop("suffix_text", None)
    return dropped


def run_one(
    engine,
    item: dict,
    args: argparse.Namespace,
    question_seed: int,
) -> dict:
    from path_consistency.polr import (
        assemble_polr_result,
        cluster_prefixes,
        indices_to_expand,
    )

    _content, prompt_ids = engine.build_prompt(item["problem"])
    prefix_traces, prefix_seconds = engine._generate_window(
        prompt_ids,
        [],
        args.n_prefixes,
        question_seed,
        args.temperature,
        args.top_p,
        args.top_k,
        args.prefix_len,
        args.logprobs,
        args.presence_penalty,
    )
    clustering = cluster_prefixes(
        [trace.get("full_text") or "" for trace in prefix_traces],
        distance_threshold=args.distance_threshold,
        n_components=args.feature_dim,
    )
    expand_indices = indices_to_expand(prefix_traces, clustering, args.prefix_len)
    expanded_traces = []
    expand_seconds = 0.0
    remaining = max(1, args.max_tokens - args.prefix_len)
    if expand_indices:
        continuations, expand_seconds = engine.generate_continuations(
            prompt_ids,
            [prefix_traces[index]["full_token_ids"] for index in expand_indices],
            question_seed + 10_000,
            args.temperature,
            args.top_p,
            args.top_k,
            remaining,
            args.logprobs,
            args.presence_penalty,
        )
        for index, continued in zip(expand_indices, continuations):
            continued["prefix_index"] = index
            expanded_traces.append(continued)
    assembled = assemble_polr_result(
        prefix_traces,
        expanded_traces,
        clustering,
        prefix_seconds,
        expand_seconds,
        args.prefix_len,
        item["answer"],
        item["task"],
        store_text=args.store_text,
    )
    assembled.update(
        {
            "question_id": item["question_id"],
            "source_row": item.get("source_row"),
            "dataset_task": item["task"],
            "method": "PoLR",
            "predictions": {
                "PoLR": {
                    "answer": assembled["answer"],
                    "correct": assembled["correct"],
                }
            },
        }
    )
    if not args.store_text:
        assembled["traces"] = [
            _strip_trace_payload(trace, store_text=False)
            for trace in assembled["traces"]
        ]
    return assembled


def main() -> None:
    args = parse_args()
    from run_math import (
        combined_sha256,
        file_sha256,
        load_eval_records,
        load_rows,
        stable_question_seed,
    )
    from path_consistency.math_protocol import instruction_for_task
    from path_consistency.polr import summarize_polr_rows

    dataset_path = Path(args.dataset).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.outdir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data, dataset_task = load_eval_records(
        dataset_path, args.max_questions, args.sample_seed
    )
    if args.answer_fallback == "auto":
        answer_fallback = "gsm8k" if dataset_task == "gsm8k" else None
    elif args.answer_fallback == "none":
        answer_fallback = None
    else:
        answer_fallback = args.answer_fallback

    capture_sizes = _capture_sizes(args.cudagraph_capture_sizes)
    implementation_paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().parent / "path_consistency" / "polr.py",
        Path(__file__).resolve().parent / "path_consistency" / "math_inference.py",
        Path(__file__).resolve().parent / "path_consistency" / "math_eval.py",
    ]
    config = {
        "protocol_version": "polr-tfidf-v1",
        "implementation_sha256": combined_sha256(implementation_paths),
        "model": str(model_path),
        "dataset": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "selected_question_ids": [item["question_id"] for item in data],
        "dataset_task": dataset_task,
        "answer_fallback": answer_fallback,
        "method": "polr",
        "n_prefixes": args.n_prefixes,
        "prefix_len": args.prefix_len,
        "distance_threshold": args.distance_threshold,
        "feature_dim": args.feature_dim,
        "embedding": "tfidf",
        "clustering": "agglomerative_average_cosine",
        "expand": "all_dominant_cluster",
        "max_questions": args.max_questions,
        "sample_seed": args.sample_seed,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "logprobs": args.logprobs,
        "presence_penalty": args.presence_penalty,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "physical_window_size": args.physical_window_size,
        "max_num_seqs": args.max_num_seqs,
        "num_gpu_blocks_override": args.num_gpu_blocks_override,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "cudagraph_capture_sizes": capture_sizes,
        "enable_prefix_caching": not args.disable_prefix_caching,
        "disable_custom_all_reduce": not args.enable_custom_all_reduce,
        "enforce_eager": not args.allow_cudagraph,
        "vllm_engine": "V1" if args.vllm_v1 else "V0",
        "gpus": args.gpus,
        "prompt": "{problem}\\n\\n" + instruction_for_task(dataset_task).replace("\n", "\\n"),
        "chat_template_enable_thinking_if_supported": not args.disable_thinking,
        "citation": "arXiv:2601.21494 Path of Least Resistance",
    }
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=list).encode()
    ).hexdigest()[:12]
    base_name = f"{args.tag}_{fingerprint}"
    manifest_path = output_dir / f"{base_name}.run.json"
    result_path = output_dir / f"{base_name}_polr.jsonl"
    summary_path = output_dir / f"{base_name}.summary.json"
    run_record = {
        "status": "loading_engine",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "config": config,
        "fingerprint": fingerprint,
        "result_paths": {"polr": str(result_path)},
        "summary_path": str(summary_path),
    }
    manifest_path.write_text(json.dumps(run_record, indent=2, ensure_ascii=False))

    previous_excepthook = sys.excepthook

    def record_failure(error_type, error, traceback) -> None:
        run_record.update(
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": getattr(error_type, "__name__", str(error_type)),
                "error_message": str(error),
            }
        )
        manifest_path.write_text(json.dumps(run_record, indent=2, ensure_ascii=False))
        previous_excepthook(error_type, error, traceback)

    sys.excepthook = record_failure

    from path_consistency.math_inference import MathInference

    engine = MathInference(
        str(model_path),
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        enable_prefix_caching=not args.disable_prefix_caching,
        disable_custom_all_reduce=not args.enable_custom_all_reduce,
        enforce_eager=not args.allow_cudagraph,
        enable_thinking=not args.disable_thinking,
        answer_fallback=answer_fallback,
        physical_window_size=args.physical_window_size,
        max_num_seqs=args.max_num_seqs,
        num_gpu_blocks_override=args.num_gpu_blocks_override,
        max_num_batched_tokens=args.max_num_batched_tokens,
        cudagraph_capture_sizes=capture_sizes,
        task=dataset_task,
    )
    engine_times = {
        "model_type": engine.model_type,
        "tokenizer_init_seconds": engine.tokenizer_init_seconds,
        "engine_init_seconds": engine.engine_init_seconds,
    }
    run_record.update({"status": "running", "engine_times": engine_times})
    manifest_path.write_text(json.dumps(run_record, indent=2, ensure_ascii=False))

    done = {row["question_id"] for row in load_rows(result_path)}
    total_start = time.perf_counter()
    with result_path.open("a") as handle:
        for question_number, item in enumerate(data, start=1):
            if item["question_id"] in done:
                continue
            row = run_one(
                engine,
                item,
                args,
                stable_question_seed(args.seed, item["question_id"]),
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            done.add(item["question_id"])
            live = summarize_polr_rows(load_rows(result_path))
            summary_path.write_text(
                json.dumps(
                    {
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "config": config,
                        "engine_times": engine_times,
                        "methods": {"PoLR": live},
                        "parent_costs": {"polr": {
                            "avg_generated_tokens": live["avg_generated_tokens"],
                            "total_generated_tokens": live["total_generated_tokens"],
                            "avg_generation_seconds": live["avg_generation_seconds"],
                            "total_generation_seconds": live["total_generation_seconds"],
                            "truncations": live["truncations"],
                        }},
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            print(
                json.dumps(
                    {
                        "progress": f"{question_number}/{len(data)}",
                        "question_id": item["question_id"],
                        "correct": row["correct"],
                        "answer": row["answer"],
                        "kappa": round(row["kappa"], 3),
                        "dominant_size": row["dominant_size"],
                        "n_expanded": row["n_expanded"],
                        "tokens": row["total_generated_tokens"],
                        "generation_seconds": round(row["generation_seconds"], 3),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    rows = load_rows(result_path)
    polr_stats = summarize_polr_rows(rows)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "engine_times": engine_times,
        "methods": {"PoLR": polr_stats},
        "parent_costs": {
            "polr": {
                "avg_generated_tokens": polr_stats["avg_generated_tokens"],
                "total_generated_tokens": polr_stats["total_generated_tokens"],
                "avg_generation_seconds": polr_stats["avg_generation_seconds"],
                "total_generation_seconds": polr_stats["total_generation_seconds"],
                "truncations": polr_stats["truncations"],
            }
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    run_record.update(
        {
            "status": "complete",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_elapsed_seconds": time.perf_counter() - total_start,
            "summary_path": str(summary_path),
        }
    )
    manifest_path.write_text(json.dumps(run_record, indent=2, ensure_ascii=False))
    polr = summary["methods"]["PoLR"]
    print(
        json.dumps(
            {
                "event": "complete",
                "tag": args.tag,
                "fingerprint": fingerprint,
                "questions": polr["questions"],
                "correct": polr["correct"],
                "accuracy": polr["accuracy"],
                "avg_generated_tokens": polr["avg_generated_tokens"],
                "avg_generation_seconds": polr["avg_generation_seconds"],
                "avg_kappa": polr["avg_kappa"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
