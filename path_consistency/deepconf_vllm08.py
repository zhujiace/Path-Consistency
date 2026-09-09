"""Official DeepConf semantics on the frozen vLLM 0.8.5 / PC conda stack.

facebookresearch/deepconf (arXiv 2508.15260) pins vLLM 0.10.2 and registers
``WrappedPerReqLogitsProcessor`` through ``LLM(logits_processors=...)`` plus
``SamplingParams.extra_args``. Those engine APIs are absent from vLLM 0.8.5.
This adapter:

* loads the official ``utils`` / ``outputs`` modules without importing
  ``deepconf.wrapper`` (which pulls the v0.10 processor);
* copies the request-level ``ConfPerReqLogitsProcessor`` body verbatim;
* attaches it via vLLM 0.8.5 ``SamplingParams.logits_processors`` (V0);
* keeps official offline generation + voting unchanged.

Online early-stop is therefore **official algorithm, engine-API adapted**.
If the V0 logits-processor probe fails, callers must label the run
``DeepConf-offline``. Do not present this as a vLLM 0.10.2 bitwise replica.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from vllm import LLM, SamplingParams

OFFICIAL_DEEPCONF_ROOT = Path(
    "/home/26xyw/data/path-consistency-baselines/deepconf"
)


def _load_official_module(module_name: str, filename: str):
    path = OFFICIAL_DEEPCONF_ROOT / "deepconf" / filename
    if not path.exists():
        raise FileNotFoundError(f"Official DeepConf file missing: {path}")
    # Register a dummy parent so relative imports inside outputs/utils keep working
    # if they ever grow. The files currently import nothing from the package.
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_official_utils = _load_official_module(
    "deepconf_official_utils", "utils.py"
)
_official_outputs = _load_official_module(
    "deepconf_official_outputs", "outputs.py"
)

process_batch_results = _official_utils.process_batch_results
process_batch_results_offline = _official_utils.process_batch_results_offline
weighted_majority_vote = _official_utils.weighted_majority_vote
compute_all_voting_results = _official_utils.compute_all_voting_results
extract_answer = _official_utils.extract_answer
DeepThinkOutput = _official_outputs.DeepThinkOutput


def compute_confidence_finite_top20(logprobs):
    """Official mean-of-top-k confidence, skipping non-finite vLLM 0.8.5 slots.

    Official ``utils.compute_confidence`` averages every dict value. On vLLM
    0.8.5 the logprob dict often contains ``-inf`` padding, so the mean is
    ``-inf`` and ``conf_bar`` becomes NaN. Drop non-finite entries and keep the
    highest 20, matching DeepConf's top-20 definition.
    """
    import math

    confs = []
    for token_logprobs in logprobs or []:
        if not token_logprobs:
            continue
        values = []
        for item in token_logprobs.values():
            value = getattr(item, "logprob", item)
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                values.append(numeric)
        if not values:
            continue
        values = sorted(values, reverse=True)[:20]
        confs.append(round(-float(np.mean(values)), 3))
    return confs


_official_utils.compute_confidence = compute_confidence_finite_top20


class ConfPerReqLogitsProcessor:
    """Request-level confidence EOS mask from official ``processors.py``.

    Official source (facebookresearch/deepconf ``ConfPerReqLogitsProcessor``):
    softmax → top-k logprob mean → sliding group mean; if the group is full and
    the mean is below ``threshold``, keep only the EOS logit.
    """

    def __init__(
        self, threshold: float, eos_token_id: int, conf_group_size: int, conf_topk: int
    ) -> None:
        from collections import deque

        self.threshold = threshold
        self.conf_list = []
        self.eos_token_id = eos_token_id
        self.conf_topk = conf_topk
        self.conf_group_list = deque(maxlen=conf_group_size)
        self.conf_grouped = 0.0
        self.conf_group_size = conf_group_size

    def compute_conf(self, logits: torch.Tensor) -> float:
        probabilities = torch.softmax(logits, dim=-1)
        top_probs, _ = torch.topk(probabilities, self.conf_topk, dim=-1)
        log_probs = torch.log(top_probs)
        return -log_probs.sum().item() / self.conf_topk

    def __call__(self, output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        flat = logits.view(-1)
        new_conf = self.compute_conf(flat)

        if len(self.conf_group_list) < self.conf_group_size:
            self.conf_group_list.append(new_conf)
            self.conf_grouped += new_conf
        else:
            self.conf_grouped -= self.conf_group_list.popleft()
            self.conf_group_list.append(new_conf)
            self.conf_grouped += new_conf

        if (
            len(self.conf_group_list) >= self.conf_group_size
            and self.conf_grouped / len(self.conf_group_list) < self.threshold
        ):
            val_to_keep = flat[self.eos_token_id].item()
            flat[:] = float("-inf")
            flat[self.eos_token_id] = val_to_keep
        return logits


class ForceEosAfterN:
    """One-shot probe processor: force EOS after ``n`` new tokens."""

    def __init__(self, n: int, eos_token_id: int) -> None:
        self.n = n
        self.eos_token_id = eos_token_id

    def __call__(self, output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        if len(output_ids) >= self.n:
            flat = logits.view(-1)
            val_to_keep = flat[self.eos_token_id].item()
            flat[:] = float("-inf")
            flat[self.eos_token_id] = val_to_keep
        return logits


class DeepThinkLLMv08:
    """Official ``DeepThinkLLM`` control flow with a vLLM 0.8.5 engine."""

    def __init__(self, model: str, **vllm_kwargs):
        self.model_name = model
        default_kwargs = {
            "tensor_parallel_size": len(
                os_environ_devices()
            ),
            "enable_prefix_caching": True,
            "trust_remote_code": True,
            "enforce_eager": True,
        }
        default_kwargs.update(vllm_kwargs)
        # Official wrapper injects WrappedPerReqLogitsProcessor here; 0.8.5 LLM
        # has no logits_processors constructor argument.
        print("Initializing vLLM engine (DeepConf v0.8.5 adapter, no LLM logits_processors)...")
        llm_init_start = time.time()
        self.llm = LLM(model=model, **default_kwargs)
        llm_init_time = time.time() - llm_init_start
        print(f"vLLM engine initialized in {llm_init_time:.2f} seconds")

        tokenizer_init_start = time.time()
        self.tokenizer = self.llm.get_tokenizer()
        tokenizer_init_time = time.time() - tokenizer_init_start
        self.init_times = {
            "llm_init_time": llm_init_time,
            "tokenizer_init_time": tokenizer_init_time,
        }
        print(f"Tokenizer initialized in {tokenizer_init_time:.2f} seconds")

    def probe_request_logits_processor(self, prompt: str, n: int = 4) -> dict:
        """Return whether V0 applies per-request logits processors (EOS after n)."""
        eos = self.tokenizer.eos_token_id
        params = SamplingParams(
            temperature=0.0,
            max_tokens=32,
            logprobs=1,
            logits_processors=[ForceEosAfterN(n, eos)],
        )
        outputs = self.llm.generate([prompt], params)
        text = outputs[0].outputs[0].text
        n_tokens = len(outputs[0].outputs[0].token_ids or [])
        finish = outputs[0].outputs[0].finish_reason
        ok = n_tokens <= n + 2
        return {
            "ok": bool(ok),
            "n_tokens": n_tokens,
            "finish_reason": finish,
            "text_preview": (text or "")[:120],
            "note": (
                "V0 SamplingParams.logits_processors applied"
                if ok
                else "processor did not stop the sequence; refuse DeepConf-online"
            ),
        }

    def deepthink(
        self,
        prompt: str,
        mode: str = "offline",
        warmup_traces: int = 16,
        total_budget: int = 32,
        confidence_percentile: int = 10,
        budget: int = 32,
        window_size: int = 2048,
        sampling_params: Optional[SamplingParams] = None,
        compute_multiple_voting: bool = True,
        base_seed: int = 0,
        allow_online: bool = True,
    ) -> "DeepThinkOutput":
        total_start_time = time.time()
        output = DeepThinkOutput()
        output.mode = mode
        output.llm_init_time = self.init_times["llm_init_time"]
        output.tokenizer_init_time = self.init_times["tokenizer_init_time"]

        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                max_tokens=2048,
                logprobs=20,
            )

        output.config = {
            "model": self.model_name,
            "mode": mode,
            "window_size": window_size,
            "compute_multiple_voting": compute_multiple_voting,
            "engine": "vllm-0.8.5-adapter",
            "official_repo": str(OFFICIAL_DEEPCONF_ROOT),
            "base_seed": base_seed,
        }

        if mode == "online":
            if not allow_online:
                raise RuntimeError(
                    "DeepConf-online requested but logits-processor probe failed"
                )
            output.config.update(
                {
                    "warmup_traces": warmup_traces,
                    "total_budget": total_budget,
                    "confidence_percentile": confidence_percentile,
                    "online_adapter": "SamplingParams.logits_processors",
                }
            )
            self._deepthink_online(
                prompt,
                output,
                warmup_traces,
                total_budget,
                confidence_percentile,
                window_size,
                sampling_params,
                base_seed,
            )
        else:
            output.config.update({"budget": budget})
            self._deepthink_offline(
                prompt, output, budget, window_size, sampling_params, base_seed
            )

        if compute_multiple_voting and output.all_traces:
            if output.mode == "online":
                output.voting_results = compute_all_voting_results(
                    output.all_voting_traces
                )
            else:
                output.voting_results = compute_all_voting_results(output.all_traces)
            if "majority" in output.voting_results and output.voting_results["majority"]:
                output.voted_answer = output.voting_results["majority"]["answer"]
                output.final_answer = output.voted_answer

        output.total_time = time.time() - total_start_time
        return output

    def _deepthink_online(
        self,
        prompt: str,
        output: DeepThinkOutput,
        warmup_traces: int,
        total_budget: int,
        confidence_percentile: int,
        window_size: int,
        sampling_params: SamplingParams,
        base_seed: int,
    ) -> DeepThinkOutput:
        processing_start = time.time()
        warmup_params_list = []
        for param_id in range(warmup_traces):
            warmup_params = copy.deepcopy(sampling_params)
            warmup_params.logprobs = 20
            warmup_params.n = 1
            warmup_params.seed = int(base_seed + param_id)
            warmup_params.logits_processors = None
            warmup_params_list.append(warmup_params)

        warmup_gen_start = time.time()
        warmup_outputs = self.llm.generate(
            [prompt for _ in range(warmup_traces)], warmup_params_list
        )
        output.warmup_gen_time = time.time() - warmup_gen_start

        warmup_result = process_batch_results(warmup_outputs, window_size)
        output.conf_bar = float(
            np.percentile(warmup_result["min_confs"], 100 - confidence_percentile)
        )
        output.warmup_min_confs = warmup_result["min_confs"]
        output.warmup_traces = warmup_result["traces"]
        output.warmup_tokens = warmup_result["total_tokens"]

        final_n = max(0, total_budget - warmup_traces)
        final_params_list = []
        eos = self.tokenizer.eos_token_id
        for param_id in range(final_n):
            final_params = copy.deepcopy(sampling_params)
            final_params.logprobs = 20
            final_params.n = 1
            final_params.seed = int(base_seed + warmup_traces + param_id)
            final_params.logits_processors = [
                ConfPerReqLogitsProcessor(
                    output.conf_bar, eos, window_size, 20
                )
            ]
            final_params_list.append(final_params)

        if final_n:
            final_gen_start = time.time()
            final_outputs = self.llm.generate(
                [prompt for _ in range(final_n)], final_params_list
            )
            output.final_gen_time = time.time() - final_gen_start
            final_result = process_batch_results(final_outputs, window_size)
            output.final_min_confs = final_result["min_confs"]
            output.final_traces = final_result["traces"]
            output.final_tokens = final_result["total_tokens"]
            for trace in output.final_traces:
                if trace["min_conf"] < output.conf_bar:
                    trace["stop_reason"] = "gconf_threshold"
        else:
            output.final_traces = []
            output.final_tokens = 0

        output.all_traces = output.warmup_traces + output.final_traces
        output.total_tokens = output.warmup_tokens + output.final_tokens
        output.total_traces_count = len(output.all_traces)
        self._perform_basic_voting(output)
        output.processing_time = time.time() - processing_start
        return output

    def _deepthink_offline(
        self,
        prompt: str,
        output: DeepThinkOutput,
        budget: int,
        window_size: int,
        sampling_params: SamplingParams,
        base_seed: int,
    ) -> DeepThinkOutput:
        sampling_params_list = []
        for param_id in range(budget):
            sampling_params_x = copy.deepcopy(sampling_params)
            sampling_params_x.logprobs = 20
            sampling_params_x.n = 1
            sampling_params_x.seed = int(base_seed + param_id)
            sampling_params_list.append(sampling_params_x)

        generation_start = time.time()
        vllm_outputs = self.llm.generate(
            [prompt for _ in range(budget)], sampling_params_list
        )
        output.generation_time = time.time() - generation_start
        processed = process_batch_results_offline(vllm_outputs, window_size)
        output.all_traces = processed["traces"]
        output.total_tokens = processed["total_tokens"]
        output.total_traces_count = len(output.all_traces)
        output.avg_tokens_per_trace = (
            output.total_tokens / output.total_traces_count
            if output.total_traces_count
            else 0
        )
        self._perform_basic_voting(output)
        output.processing_time = time.time() - generation_start
        return output

    def _perform_basic_voting(self, output: DeepThinkOutput) -> None:
        voting_answers = []
        voting_weights = []
        if output.mode == "online":
            output.all_voting_traces = []
            for trace in output.warmup_traces:
                if (
                    trace.get("min_conf", 0) >= output.conf_bar
                    and trace.get("extracted_answer")
                ):
                    voting_answers.append(trace["extracted_answer"])
                    voting_weights.append(trace.get("min_conf", 1.0))
                    output.all_voting_traces.append(trace)
            for trace in output.final_traces:
                if trace.get("stop_reason") == "gconf_threshold":
                    continue
                if trace.get("extracted_answer"):
                    voting_answers.append(trace["extracted_answer"])
                    voting_weights.append(trace.get("min_conf", 1.0))
                    output.all_voting_traces.append(trace)
        else:
            for trace in output.all_traces:
                if trace.get("extracted_answer"):
                    voting_answers.append(trace["extracted_answer"])
                    voting_weights.append(1.0)

        output.voting_answers = voting_answers
        output.voting_weights = voting_weights
        output.voted_answer = weighted_majority_vote(voting_answers, voting_weights)
        output.final_answer = output.voted_answer
        if output.mode == "online":
            output.avg_tokens_per_warmup_trace = (
                output.warmup_tokens / len(output.warmup_traces)
                if output.warmup_traces
                else 0
            )
            output.avg_tokens_per_final_trace = (
                output.final_tokens / len(output.final_traces)
                if output.final_traces
                else 0
            )


def os_environ_devices() -> list[str]:
    import os

    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    return [part for part in raw.split(",") if part != ""]


def compact_trace(trace: dict) -> dict:
    """Drop bulky token-id arrays before writing JSONL."""
    return {
        "stop_reason": trace.get("stop_reason"),
        "text": trace.get("text"),
        "num_tokens": trace.get("num_tokens"),
        "min_conf": trace.get("min_conf"),
        "extracted_answer": trace.get("extracted_answer"),
        "confs": trace.get("confs"),
    }
