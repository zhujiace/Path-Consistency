#!/usr/bin/env python3
"""Run paired MATH500 SC/PC pilots with Qwen3 and offline DeepConf views.

Example:
    /home/26xyw/.conda/envs/PC/bin/python run_math.py \
      --gpus 1,2,3,4 --methods sc,pc --max-questions 10 --tag initial10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


DEEPCONF_LABELS = {
    "offline_mean": "Offline-Mean",
    "offline_bottom10_90": "Offline-Bottom10@90%",
    "offline_bottom10_10": "Offline-Bottom10@10%",
    "offline_tail_90": "Offline-Tail@90%",
    "offline_tail_10": "Offline-Tail@10%",
    "offline_lgc_90": "Offline-LGC@90%",
    "offline_lgc_10": "Offline-LGC@10%",
}

# Llama engineering pilot IDs; primary MATH-500 stats use the remaining 490.
ENGINEERING_SOURCE_ROWS = [303, 457, 359, 315, 316, 415, 407, 370, 151, 332]


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

# torch cu118 NCCL cannot capture CUDA graphs; install the lab CUDA-12.2
# NCCL before any later torch/vllm import (same recipe as Qwen3-30B-A3B).
install_cuda122_nccl()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if "--vllm-v1" not in sys.argv:
    # Default remains V0+eager. Pass --vllm-v1 --allow-cudagraph to match
    # the Qwen3-30B-A3B serving path (V1 + CUDA graph + CUDA-12.2 NCCL).
    os.environ.setdefault("VLLM_USE_V1", "0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="sc,pc", help="comma-separated: sc,pc")
    parser.add_argument("--gpus", default="1,2,3,4")
    parser.add_argument("--model", default="/home/26xyw/data/Qwen3-30B-A3B")
    parser.add_argument(
        "--dataset", default="/home/26xyw/data/Datasets/math500_test.jsonl"
    )
    parser.add_argument(
        "--outdir",
        default="/home/26xyw/data/path-consistency-results/math500",
    )
    parser.add_argument("--tag", default="qwen3_30b_a3b_math500")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument(
        "--source-rows",
        default=None,
        help="comma-separated zero-based dataset rows; overrides random selection",
    )
    parser.add_argument("--sample-seed", type=int, default=2027)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-branches", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--beta-threshold", type=float, default=0.8)
    parser.add_argument("--ratios", default="0,0.2,0.4,0.6,0.8")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--logprobs", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--physical-window-size", type=int, default=0)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument(
        "--num-gpu-blocks-override",
        type=int,
        default=None,
        help="Per-rank KV blocks (16 tokens each). Cap this so CUDA-graph "
        "capture still has free memory after weight load.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Prefill batch cap; also bounds CUDA-graph capture shapes.",
    )
    parser.add_argument(
        "--cudagraph-capture-sizes",
        default=None,
        help="comma-separated CUDA-graph capture sizes, e.g. 32,16,8,4,2,1",
    )
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--enable-custom-all-reduce", action="store_true")
    parser.add_argument("--allow-cudagraph", action="store_true")
    parser.add_argument("--vllm-v1", action="store_true")
    parser.add_argument("--store-text", action="store_true")
    parser.add_argument("--store-confidence", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--answer-fallback",
        choices=["auto", "none", "gsm8k"],
        default="auto",
        help="none/gsm8k, or auto from dataset fields",
    )
    parser.add_argument(
        "--exclude-engineering-split",
        action="store_true",
        help="Drop the 10 MATH-500 engineering-pilot source rows (primary 490)",
    )
    return parser.parse_args()


def _record_task(item: dict) -> str:
    dataset = str(item.get("dataset") or item.get("task") or "").lower()
    if dataset == "gpqa" or dataset.startswith("gpqa") or "choices" in item:
        return "gpqa"
    if dataset.startswith("aime") or item.get("task") == "aime":
        return "aime"
    if "problem" in item and "answer" in item:
        return "math"
    if "question" in item and "target" in item:
        return "gsm8k"
    return "unknown"


def load_eval_records(
    path: Path,
    max_questions: int | None,
    sample_seed: int,
    source_rows: list[int] | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> tuple[list[dict], str]:
    """Load MATH-500, GSM8K, AIME, or GPQA JSONL into the shared SC/PC schema."""
    from path_consistency.math_eval import (
        extract_aime_integer,
        extract_gpqa_letter,
        format_gsm8k_gold,
    )
    from path_consistency.math_protocol import format_gpqa_stem

    records = []
    task = None
    with path.open() as handle:
        for row_number, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            record_task = _record_task(item)
            if record_task == "gpqa":
                choices = item["choices"]
                gold = extract_gpqa_letter(item.get("answer"))
                if gold is None:
                    raise ValueError(f"{path} row {row_number} has no A/B/C/D gold")
                records.append(
                    {
                        "question_id": str(
                            item.get("unique_id")
                            or item.get("Record ID")
                            or f"gpqa_{row_number:03d}"
                        ),
                        "problem": format_gpqa_stem(item["question"], choices),
                        "answer": gold,
                        "subject": item.get("domain") or item.get("High-level domain"),
                        "level": item.get("subdomain") or item.get("Subdomain"),
                        "source_row": row_number,
                        "year": None,
                        "task": "gpqa",
                    }
                )
            elif record_task == "aime":
                gold = extract_aime_integer(item.get("answer"))
                if gold is None:
                    raise ValueError(f"{path} row {row_number} is not an AIME integer")
                year = item.get("year")
                records.append(
                    {
                        "question_id": str(
                            item.get("unique_id", f"aime_{year}_{row_number:02d}")
                        ),
                        "problem": item["problem"],
                        "answer": gold,
                        "subject": f"AIME {year}" if year else "AIME",
                        "level": year,
                        "source_row": row_number,
                        "year": year,
                        "task": "aime",
                    }
                )
            elif record_task == "math":
                records.append(
                    {
                        "question_id": str(item.get("unique_id", f"math_{row_number}")),
                        "problem": item["problem"],
                        "answer": str(item["answer"]),
                        "subject": item.get("subject"),
                        "level": item.get("level"),
                        "source_row": row_number,
                        "year": None,
                        "task": "math",
                    }
                )
            elif record_task == "gsm8k":
                records.append(
                    {
                        "question_id": f"gsm8k_{row_number:04d}",
                        "problem": item["question"],
                        "answer": format_gsm8k_gold(item["target"]),
                        "subject": "gsm8k",
                        "level": None,
                        "source_row": row_number,
                        "year": None,
                        "task": "gsm8k",
                    }
                )
            else:
                raise ValueError(
                    f"{path} row {row_number} is not a supported eval JSONL record"
                )
            if task is None:
                task = record_task
            elif task != record_task:
                raise ValueError(f"{path} mixes {task} and {record_task} records")
    if not records:
        raise ValueError(f"{path} is empty")
    if source_rows is not None:
        invalid = [index for index in source_rows if index < 0 or index >= len(records)]
        if invalid:
            raise ValueError(f"Invalid --source-rows values: {invalid}")
        records = [records[index] for index in source_rows]
    elif max_questions is not None and max_questions < len(records):
        rng = random.Random(sample_seed)
        indices = list(range(len(records)))
        rng.shuffle(indices)
        records = [records[index] for index in indices[:max_questions]]
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")
    if num_shards > 1:
        records = [
            record
            for index, record in enumerate(records)
            if index % num_shards == shard_index
        ]
    return records, task or "math"


def load_math500(
    path: Path,
    max_questions: int | None,
    sample_seed: int,
    source_rows: list[int] | None = None,
) -> list[dict]:
    records, _task = load_eval_records(
        path, max_questions, sample_seed, source_rows=source_rows
    )
    return records


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combined_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.name).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def stable_question_seed(seed: int, question_id: str) -> int:
    payload = f"{seed}:{question_id}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2**31 - 1)


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _present(rows: list[dict], key: str) -> list:
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None:
            values.append(value)
    return values


def _mean(rows: list[dict], key: str) -> float | None:
    values = _present(rows, key)
    if not values:
        return None
    return sum(values) / len(values)


def _sum_present(rows: list[dict], key: str) -> float | None:
    values = _present(rows, key)
    if not values:
        return None
    return sum(values)


def _max_present(rows: list[dict], key: str) -> float | None:
    values = _present(rows, key)
    if not values:
        return None
    return max(values)


def summarize(rows: list[dict], config: dict, engine_times: dict) -> dict:
    prediction_names = sorted(
        {
            name
            for row in rows
            for name in row.get("predictions", {})
        }
    )
    methods = {}
    for name in prediction_names:
        eligible = [row["predictions"][name] for row in rows if name in row["predictions"]]
        methods[name] = {
            "questions": len(eligible),
            "correct": sum(bool(item["correct"]) for item in eligible),
            "accuracy": (
                sum(bool(item["correct"]) for item in eligible) / len(eligible)
                if eligible
                else None
            ),
        }
    parent_costs = {}
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        if not selected:
            continue
        parent_costs[method] = {
            "questions": len(selected),
            "total_generated_tokens": sum(row["total_generated_tokens"] for row in selected),
            "avg_generated_tokens": sum(
                row["total_generated_tokens"] for row in selected
            )
            / len(selected),
            "avg_prompt_tokens": sum(row["total_prompt_tokens"] for row in selected)
            / len(selected),
            "avg_prefix_tokens": sum(row["total_prefix_tokens"] for row in selected)
            / len(selected),
            "avg_request_prompt_tokens": sum(
                row.get("total_request_prompt_tokens", row["total_prompt_tokens"])
                for row in selected
            )
            / len(selected),
            "total_generation_seconds": sum(row["generation_seconds"] for row in selected),
            "avg_generation_seconds": sum(row["generation_seconds"] for row in selected)
            / len(selected),
            "total_actual_generation_seconds": sum(
                row.get("actual_generation_seconds", row["generation_seconds"])
                for row in selected
            ),
            "reused_root_windows": sum(
                row.get("reused_root_windows", 0) for row in selected
            ),
            "parser_failures": sum(row["parser_failures"] for row in selected),
            "truncations": sum(row["truncations"] for row in selected),
            "avg_unique_kv_tokens": _mean(selected, "total_unique_kv_tokens"),
            "avg_logical_unique_kv_tokens": _mean(
                selected, "total_logical_unique_kv_tokens"
            ),
            "avg_naive_kv_tokens": _mean(selected, "total_naive_kv_tokens"),
            "total_unique_kv_tokens": _sum_present(selected, "total_unique_kv_tokens"),
            "total_naive_kv_tokens": _sum_present(selected, "total_naive_kv_tokens"),
            "avg_unique_kv_bytes": _mean(selected, "total_unique_kv_bytes"),
            "peak_gpu_cache_usage": _max_present(selected, "peak_gpu_cache_usage"),
            "avg_prefix_cache_hit_tokens": _mean(
                selected, "total_prefix_cache_hit_tokens"
            ),
        }
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "engine_times": engine_times,
        "completed_rows": len(rows),
        "methods": methods,
        "parent_costs": parent_costs,
    }


def main() -> None:
    args = parse_args()
    methods = [method.strip().lower() for method in args.methods.split(",") if method.strip()]
    if not methods or any(method not in {"sc", "pc"} for method in methods):
        raise SystemExit("--methods must contain only sc and/or pc")
    if set(methods) == {"sc", "pc"}:
        # SC must run first so PC can reuse the exact no-prefix windows. This
        # makes the no-op part of PC a common-random-number comparison.
        methods = ["sc", "pc"]
    ratios = tuple(float(value) for value in args.ratios.split(","))
    if ratios[0] != 0.0 or sorted(ratios) != list(ratios):
        raise SystemExit("--ratios must be increasing and start at 0")
    capture_sizes = None
    if args.cudagraph_capture_sizes:
        capture_sizes = [
            int(value)
            for value in args.cudagraph_capture_sizes.split(",")
            if value.strip()
        ]
    capture_sizes = (
        [int(value) for value in args.cudagraph_capture_sizes.split(",") if value.strip()]
        if args.cudagraph_capture_sizes
        else None
    )

    dataset_path = Path(args.dataset).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.outdir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read JSONL before importing vLLM/torch. Loading dataset libraries after a
    # CUDA context has been created is unsafe on this machine.
    requested_rows = (
        [int(value) for value in args.source_rows.split(",")]
        if args.source_rows
        else None
    )
    data, dataset_task = load_eval_records(
        dataset_path,
        args.max_questions,
        args.sample_seed,
        requested_rows,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    excluded_engineering = []
    if args.exclude_engineering_split:
        if dataset_task != "math":
            raise SystemExit("--exclude-engineering-split only applies to MATH-500")
        before = len(data)
        data = [item for item in data if item["source_row"] not in ENGINEERING_SOURCE_ROWS]
        excluded_engineering = list(ENGINEERING_SOURCE_ROWS)
        if (
            args.num_shards == 1
            and requested_rows is None
            and args.max_questions is None
            and len(data) != before - len(ENGINEERING_SOURCE_ROWS)
        ):
            raise SystemExit(
                f"engineering split filter expected {before - len(ENGINEERING_SOURCE_ROWS)} "
                f"questions, got {len(data)}"
            )
    if args.answer_fallback == "auto":
        answer_fallback = "gsm8k" if dataset_task == "gsm8k" else None
    elif args.answer_fallback == "none":
        answer_fallback = None
    else:
        answer_fallback = args.answer_fallback
    from path_consistency.math_protocol import instruction_for_task

    implementation_paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().parent / "path_consistency" / "math_inference.py",
        Path(__file__).resolve().parent / "path_consistency" / "math_protocol.py",
        Path(__file__).resolve().parent / "path_consistency" / "math_eval.py",
        Path(__file__).resolve().parent / "path_consistency" / "confidence.py",
        Path(__file__).resolve().parent / "path_consistency" / "deepconf_offline.py",
        Path(__file__).resolve().parent / "path_consistency" / "vllm_nccl.py",
        Path(__file__).resolve().parent / "path_consistency" / "kv_accounting.py",
    ]
    config = {
        "protocol_version": "math-pc-paired-root-v2",
        "implementation_sha256": combined_sha256(implementation_paths),
        "model": str(model_path),
        "dataset": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "selected_question_ids": [item["question_id"] for item in data],
        "dataset_task": dataset_task,
        "answer_fallback": answer_fallback,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "methods": methods,
        "max_questions": args.max_questions,
        "requested_source_rows": requested_rows,
        "excluded_engineering_source_rows": excluded_engineering,
        "sample_seed": args.sample_seed,
        "seed": args.seed,
        "max_branches": args.max_branches,
        "window_size": args.window_size,
        "beta_threshold": args.beta_threshold,
        "ratios": ratios,
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
        "pc_length_policy": "provisional_fixed_plus_20_percent",
        "kv_metrics": (
            "unique_kv_tokens=shared prompt/prefix + decode; "
            "naive_kv_tokens charges prompt per sequence; "
            "reused SC roots contribute 0 GPU unique KV"
        ),
        "deepconf_protocol": (
            "official offline top-20 confidence; Mean, Bottom-10% Group, Tail(2048), "
            "and LGC(2048); percentile retain top 90%/10%; weighted vote"
        ),
        "paired_common_root_windows": set(methods) == {"sc", "pc"},
    }
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=list).encode()
    ).hexdigest()[:12]
    base_name = f"{args.tag}_{fingerprint}"
    manifest_path = output_dir / f"{base_name}.run.json"
    result_paths = {
        method: output_dir / f"{base_name}_{method}.jsonl" for method in methods
    }
    summary_path = output_dir / f"{base_name}.summary.json"

    run_record = {
        "status": "loading_engine",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "config": config,
        "fingerprint": fingerprint,
        "result_paths": {key: str(value) for key, value in result_paths.items()},
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

    from path_consistency.math_eval import grade_answer
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

    done = {
        method: {row["question_id"] for row in load_rows(path)}
        for method, path in result_paths.items()
    }

    def infer(
        item: dict,
        method: str,
        paired_root_windows: list[dict] | None = None,
        return_internal_windows: bool = False,
    ) -> dict:
        return engine.run_question(
            question_id=item["question_id"],
            problem=item["problem"],
            method=method,
            max_branches=args.max_branches,
            window_size=args.window_size,
            beta_threshold=args.beta_threshold,
            ratios=ratios,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            logprobs=args.logprobs,
            presence_penalty=args.presence_penalty,
            seed=stable_question_seed(args.seed, item["question_id"]),
            store_text=args.store_text,
            store_confidence=args.store_confidence,
            paired_root_windows=paired_root_windows,
            return_internal_windows=return_internal_windows,
        )

    total_start = time.perf_counter()
    for question_number, item in enumerate(data, start=1):
        paired_root_windows = None
        for method in methods:
            if item["question_id"] in done[method]:
                if (
                    method == "sc"
                    and "pc" in methods
                    and item["question_id"] not in done["pc"]
                ):
                    cache_output = infer(
                        item, "sc", return_internal_windows=True
                    )
                    paired_root_windows = cache_output.pop("_internal_root_windows")
                    print(
                        json.dumps(
                            {
                                "progress": f"{question_number}/{len(data)}",
                                "question_id": item["question_id"],
                                "method": "sc",
                                "event": "rebuilt_pair_cache_for_resume",
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                continue
            output = infer(
                item,
                method,
                paired_root_windows=paired_root_windows if method == "pc" else None,
                return_internal_windows=method == "sc" and "pc" in methods,
            )
            if method == "sc" and "_internal_root_windows" in output:
                paired_root_windows = output.pop("_internal_root_windows")
            gold = item["answer"]
            first_answer = output["traces"][0]["answer"] if output["traces"] else None
            parent_name = "SC" if method == "sc" else "PC"
            predictions = {
                parent_name: {
                    "answer": output["answer"],
                    "correct": grade_answer(dataset_task, output["answer"], gold),
                },
                f"Pass@1({parent_name})": {
                    "answer": first_answer,
                    "correct": grade_answer(dataset_task, first_answer, gold),
                },
            }
            if args.logprobs >= 20:
                for policy_name, policy in output["deepconf"].items():
                    suffix = "" if method == "sc" else "+PC"
                    display_name = DEEPCONF_LABELS[policy_name] + suffix
                    predictions[display_name] = {
                        "answer": policy["answer"],
                        "correct": grade_answer(dataset_task, policy["answer"], gold),
                        "retained": policy["retained"],
                        "valid_traces": policy["valid_traces"],
                    }
            row = {
                **output,
                "gold": gold,
                "subject": item["subject"],
                "level": item["level"],
                "source_row": item["source_row"],
                "year": item.get("year"),
                "dataset_task": dataset_task,
                "predictions": predictions,
            }
            with result_paths[method].open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            done[method].add(item["question_id"])
            print(
                json.dumps(
                    {
                        "progress": f"{question_number}/{len(data)}",
                        "question_id": item["question_id"],
                        "method": method,
                        "predictions": predictions,
                        "generated_tokens": output["total_generated_tokens"],
                        "generation_seconds": round(output["generation_seconds"], 3),
                        "actual_generation_seconds": round(
                            output.get(
                                "actual_generation_seconds",
                                output["generation_seconds"],
                            ),
                            3,
                        ),
                        "reused_root_windows": output.get("reused_root_windows", 0),
                        "unique_kv_tokens": output.get("total_unique_kv_tokens"),
                        "naive_kv_tokens": output.get("total_naive_kv_tokens"),
                        "unique_kv_mib": (
                            None
                            if output.get("total_unique_kv_bytes") is None
                            else round(output["total_unique_kv_bytes"] / (1024 * 1024), 2)
                        ),
                        "peak_gpu_cache_usage": output.get("peak_gpu_cache_usage"),
                        "parser_failures": output["parser_failures"],
                        "truncations": output["truncations"],
                        "final_prefix_ratio": output["final_prefix_ratio"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        rows = [row for path in result_paths.values() for row in load_rows(path)]
        summary = summarize(rows, config, engine_times)
        summary["run_elapsed_seconds"] = time.perf_counter() - total_start
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    rows = [row for path in result_paths.values() for row in load_rows(path)]
    summary = summarize(rows, config, engine_times)
    summary["run_elapsed_seconds"] = time.perf_counter() - total_start
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    run_record.update(
        {
            "status": "complete",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_elapsed_seconds": summary["run_elapsed_seconds"],
            "summary_path": str(summary_path),
        }
    )
    manifest_path.write_text(json.dumps(run_record, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
