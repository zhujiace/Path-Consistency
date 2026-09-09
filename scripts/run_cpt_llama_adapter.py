#!/usr/bin/env python3
"""Launch official CPT ``qwen.math`` with a Llama-4090 engine constraint.

Official ``LLM()`` does not pass ``max_model_len``. Llama-3.1-8B-Instruct
defaults to 131072, which exceeds KV capacity on one RTX 4090 (~31k tokens).
This launcher only sets ``max_model_len`` (and eager for V0 stability). It
does **not** add a thinking switch, so the run is CPT-adapted-llama, not a
fair Qwen CPT comparison.
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
        kwargs.setdefault("max_model_len", 4096)
        kwargs.setdefault("gpu_memory_utilization", 0.85)
        kwargs["enforce_eager"] = True
        kwargs.setdefault("trust_remote_code", True)
        print(
            "[cpt-llama-adapter] engine overrides: "
            f"max_model_len={kwargs.get('max_model_len')} "
            f"gpu_memory_utilization={kwargs.get('gpu_memory_utilization')} "
            "enforce_eager=True (official LLM() has none of these)"
        )
        super().__init__(*args, **kwargs)


vllm.LLM = Llama4090LLM
sys.argv[0] = "qwen.math"
runpy.run_module("qwen.math", run_name="__main__")
