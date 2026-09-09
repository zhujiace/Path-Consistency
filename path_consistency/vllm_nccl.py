"""Install a CUDA-12.x NCCL before importing torch/vLLM.

torch 2.6.0+cu118 ships NCCL 2.21.5 built against CUDA 11.0. That build cannot
capture CUDA graphs and raises ``NCCL error: invalid usage`` during TP
all-reduce warmup. On the 4090 lab, Qwen3-30B-A3B already used a matching
NCCL 2.21.5 compiled with CUDA 12.2; reuse that .so plus cudart 12.2.

On a new server this is a no-op unless the files exist, or the operator sets
``PC_NCCL_SO`` / ``PC_CUDA_LIB``. If graphs capture cleanly with the stock
NCCL, leave both unset.
"""

from __future__ import annotations

import os
from pathlib import Path

NCCL_SO = Path("/home/26xyw/data/qwen3.8-infra/cache/nccl/libnccl.so.2")
CUDA_122_LIB = Path("/usr/local/cuda-12.2/lib64")


def _existing_file(*candidates: Path) -> Path | None:
    for path in candidates:
        if path.is_file():
            return path
    return None


def _existing_dir(*candidates: Path) -> Path | None:
    for path in candidates:
        if path.is_dir():
            return path
    return None


def install_cuda122_nccl() -> dict[str, str]:
    """Mutate process env. Must run before importing torch or vllm."""
    applied: dict[str, str] = {}
    nccl_candidates = []
    for key in ("PC_NCCL_SO", "VLLM_NCCL_SO_PATH"):
        value = os.environ.get(key)
        if value:
            nccl_candidates.append(Path(value))
    nccl_candidates.append(NCCL_SO)
    nccl = _existing_file(*nccl_candidates)
    if nccl is not None:
        os.environ["VLLM_NCCL_SO_PATH"] = str(nccl)
        applied["VLLM_NCCL_SO_PATH"] = str(nccl)

    cuda_candidates = []
    cuda_override = os.environ.get("PC_CUDA_LIB")
    if cuda_override:
        cuda_candidates.append(Path(cuda_override))
    cuda_candidates.append(CUDA_122_LIB)
    cuda_lib = _existing_dir(*cuda_candidates)
    if cuda_lib is not None:
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        if str(cuda_lib) not in ld.split(":"):
            os.environ["LD_LIBRARY_PATH"] = (
                f"{cuda_lib}:{ld}" if ld else str(cuda_lib)
            )
        applied["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]

    applied["nccl_so_exists"] = str(nccl is not None)
    applied["cuda122_lib_exists"] = str(cuda_lib is not None)
    return applied
