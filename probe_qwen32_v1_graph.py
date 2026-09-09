#!/usr/bin/env python3
"""Short vLLM V1 CUDA-graph probe for Qwen3-32B TP=4.

Reuses the Qwen3-30B-A3B serving recipe from llm-ai-infra:
CUDA-12.2 NCCL, V1 (do not set VLLM_USE_V1=0), CUDA graph on,
leave KV headroom for capture. Not a MATH-500 run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for index, value in enumerate(sys.argv):
    if value == "--gpus" and index + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[index + 1]
        break
    if value.startswith("--gpus="):
        os.environ["CUDA_VISIBLE_DEVICES"] = value.split("=", 1)[1]
        break
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2,3,4")

os.environ.setdefault("HF_HOME", str(Path.home() / "data" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.pop("VLLM_USE_V1", None)

from path_consistency.vllm_nccl import install_cuda122_nccl  # noqa: E402

NCCL_INFO = install_cuda122_nccl()

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="1,2,3,4")
    parser.add_argument("--model", default="/home/26xyw/data/Qwen3-32B")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--num-gpu-blocks-override", type=int, default=200)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument(
        "--prompt",
        default=(
            "Write the integers from 1 to 80 as a comma-separated list. "
            "Do not add any other text."
        ),
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--capture-sizes",
        default="32,16,8,4,2,1",
        help="CUDA-graph capture sizes, comma-separated (dense 32B cannot "
        "capture A3B's 512-token max shape on 24 GiB).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture_sizes = [int(x) for x in args.capture_sizes.split(",") if x.strip()]
    print(
        json.dumps(
            {
                "event": "probe_start",
                "model": args.model,
                "gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "tensor_parallel_size": args.tensor_parallel_size,
                "max_model_len": args.max_model_len,
                "enforce_eager": args.enforce_eager,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "max_num_seqs": args.max_num_seqs,
                "num_gpu_blocks_override": args.num_gpu_blocks_override,
                "capture_sizes": capture_sizes,
                "vllm_use_v1_env": os.environ.get("VLLM_USE_V1"),
                "nccl": NCCL_INFO,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True
    )
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    engine_start = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        num_gpu_blocks_override=args.num_gpu_blocks_override,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        compilation_config={"cudagraph_capture_sizes": capture_sizes},
        trust_remote_code=True,
    )
    engine_init_seconds = time.perf_counter() - engine_start
    print(
        json.dumps(
            {
                "event": "engine_ready",
                "engine_init_seconds": round(engine_init_seconds, 3),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    params = SamplingParams(
        n=args.n,
        max_tokens=args.max_tokens,
        temperature=0.6 if args.n > 1 else 0.0,
        top_p=0.95 if args.n > 1 else 1.0,
        seed=0,
    )
    generate_start = time.perf_counter()
    outputs = llm.generate(
        [{"prompt_token_ids": prompt_ids}], params, use_tqdm=False
    )
    generate_seconds = time.perf_counter() - generate_start
    sequences = outputs[0].outputs
    generated = sum(len(seq.token_ids) for seq in sequences)
    tok_s = generated / generate_seconds if generate_seconds > 0 else 0.0
    print(
        json.dumps(
            {
                "event": "generate_done",
                "n": args.n,
                "generated_tokens": generated,
                "generate_seconds": round(generate_seconds, 3),
                "tokens_per_second": round(tok_s, 2),
                "preview": sequences[0].text[:400] if sequences else "",
                "finish_reason": sequences[0].finish_reason if sequences else None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
