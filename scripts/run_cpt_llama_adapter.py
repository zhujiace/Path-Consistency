"""Launch official CPT ``qwen.math`` on one RTX 4090 with Llama-3.1-8B.

Official ``LLM()`` does not pass ``max_model_len``. Llama-3.1-8B-Instruct
defaults to 131072, which exceeds KV capacity on one 4090. This launcher
only sets engine limits (max_model_len / gpu memory / eager). It does not
add a thinking switch: Llama has none. This is backbone-matched CPT vs
Llama SC/PC on GSM8K, not a fair Qwen thinking-on CPT comparison.
"""

from __future__ import annotations

import os
import runpy
import sys


def _set_cuda() -> None:
    for index, value in enumerate(sys.argv):
        if value == "--gpu" and index + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[index + 1]
            del sys.argv[index : index + 2]
            return
        if value.startswith("--gpu="):
            os.environ["CUDA_VISIBLE_DEVICES"] = value.split("=", 1)[1]
            del sys.argv[index]
            return


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw.strip() == "" else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw.strip() == "" else float(raw)


_set_cuda()
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/data/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

CPT_CODE = "/home/26xyw/data/path-consistency-baselines/CPT/code"
if CPT_CODE not in sys.path:
    sys.path.insert(0, CPT_CODE)

import vllm  # noqa: E402


class Llama4090LLM(vllm.LLM):
    def __init__(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["max_model_len"] = _env_int("PC_CPT_MAX_MODEL_LEN", 4096)
        kwargs["gpu_memory_utilization"] = _env_float("PC_CPT_GPU_MEM", 0.90)
        kwargs["max_num_seqs"] = _env_int("PC_CPT_MAX_NUM_SEQS", 32)
        kwargs["enforce_eager"] = os.environ.get("PC_CPT_ENFORCE_EAGER", "1") != "0"
        kwargs.setdefault("trust_remote_code", True)
        print(
            "[cpt-llama-adapter] engine overrides: "
            f"max_model_len={kwargs.get('max_model_len')} "
            f"gpu_memory_utilization={kwargs.get('gpu_memory_utilization')} "
            f"max_num_seqs={kwargs.get('max_num_seqs')} "
            f"enforce_eager={kwargs.get('enforce_eager')} "
            "(official LLM() has none of these)"
        )
        super().__init__(*args, **kwargs)


vllm.LLM = Llama4090LLM
sys.argv[0] = "qwen.math"
runpy.run_module("qwen.math", run_name="__main__")
