"""Qwen3 MATH inference for SC and shared-prefix Path-Consistency (PC)."""

from __future__ import annotations

import time
from typing import Optional

from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams

from .deepconf_offline import lowest_group_confidence, token_confidence
from .kv_accounting import kv_bytes_per_token, snapshot_vllm_gpu_kv
from .math_eval import (
    canonicalize_vote_answer,
    extract_gsm8k_numeric,
    extract_last_boxed,
)
from .math_protocol import (  # re-exported for tests and analysis
    MATH_INSTRUCTION,
    instruction_for_task,
    gate_snapshot,
    lower_tail_local_score,
    run_sc_pc_question,
    safe_ratio_endpoint as _safe_ratio_endpoint,
    select_shared_prefix,
)


def safe_ratio_endpoint(
    full_reasoning_ids: list[int],
    ratio: float,
    minimum: int,
    tokenizer,
    max_backtrack: int = 32,
) -> Optional[int]:
    return _safe_ratio_endpoint(
        full_reasoning_ids,
        ratio,
        minimum,
        lambda ids: tokenizer.decode(ids, skip_special_tokens=False),
        max_backtrack=max_backtrack,
    )


def _step_logprobs(output) -> tuple[list[Optional[float]], list[Optional[float]]]:
    sampled = []
    confidences = []
    steps = output.logprobs or []
    for token_id, step in zip(output.token_ids, steps):
        selected = step.get(token_id) if step else None
        if selected is None and step:
            selected = next(iter(step.values()))
        value = getattr(selected, "logprob", selected)
        try:
            sampled.append(float(value))
        except (TypeError, ValueError):
            sampled.append(None)
        confidences.append(token_confidence(step))
    missing = len(output.token_ids) - len(sampled)
    if missing > 0:
        sampled.extend([None] * missing)
        confidences.extend([None] * missing)
    return sampled, confidences


def _reasoning_length(
    token_ids: list[int], close_think_id: Optional[int], tokenizer
) -> int:
    if close_think_id is not None and close_think_id in token_ids:
        return token_ids.index(close_think_id)

    # Fallback for models/runs without an explicit </think> token. Locate the
    # earliest final-answer marker as a token sequence, preserving exact IDs.
    marker_starts = []
    for marker in ("\\boxed", "Final Answer", "final answer"):
        marker_ids = tokenizer.encode(marker, add_special_tokens=False)
        if not marker_ids:
            continue
        width = len(marker_ids)
        for start in range(0, len(token_ids) - width + 1):
            if token_ids[start : start + width] == marker_ids:
                marker_starts.append(start)
                break
    if marker_starts:
        return min(marker_starts)

    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    character_starts = [
        pos
        for marker in ("\\boxed", "Final Answer", "final answer")
        if (pos := decoded.find(marker)) >= 0
    ]
    if not character_starts:
        return len(token_ids)
    marker_position = min(character_starts)
    low, high = 0, len(token_ids)
    while low < high:
        middle = (low + high) // 2
        prefix_text = tokenizer.decode(
            token_ids[:middle], skip_special_tokens=False
        )
        if len(prefix_text) >= marker_position:
            high = middle
        else:
            low = middle + 1
    return max(0, low - 1)


class MathInference:
    """One shared vLLM engine serving paired SC/PC MATH experiments."""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 4,
        dtype: str = "bfloat16",
        max_model_len: int = 40960,
        gpu_memory_utilization: float = 0.90,
        seed: int = 0,
        enable_prefix_caching: bool = True,
        disable_custom_all_reduce: bool = True,
        enforce_eager: bool = True,
        enable_thinking: bool = True,
        answer_fallback: Optional[str] = None,
        physical_window_size: int = 0,
        max_num_seqs: int = 8,
        num_gpu_blocks_override: Optional[int] = None,
        max_num_batched_tokens: Optional[int] = None,
        cudagraph_capture_sizes: Optional[list[int]] = None,
        task: str = "math",
    ):
        self.model_path = model_path
        self.enable_thinking = enable_thinking
        self.answer_fallback = answer_fallback
        self.task = task
        self.physical_window_size = max(0, int(physical_window_size))
        self.max_num_seqs = max_num_seqs
        model_config = AutoConfig.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )
        self.model_type = str(getattr(model_config, "model_type", "unknown"))
        tokenizer_start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )
        self.tokenizer_init_seconds = time.perf_counter() - tokenizer_start
        self.close_think_id = None
        if self.model_type.startswith("qwen3"):
            candidate = self.tokenizer.convert_tokens_to_ids("</think>")
            if candidate != self.tokenizer.unk_token_id:
                self.close_think_id = candidate

        engine_start = time.perf_counter()
        llm_kwargs = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": dtype,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "seed": seed,
            "enable_prefix_caching": enable_prefix_caching,
            "disable_custom_all_reduce": disable_custom_all_reduce,
            "enforce_eager": enforce_eager,
            "max_num_seqs": max_num_seqs,
            "trust_remote_code": True,
        }
        if num_gpu_blocks_override is not None:
            llm_kwargs["num_gpu_blocks_override"] = num_gpu_blocks_override
        if max_num_batched_tokens is not None:
            llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        if cudagraph_capture_sizes:
            llm_kwargs["compilation_config"] = {
                "cudagraph_capture_sizes": list(cudagraph_capture_sizes)
            }
        self.llm = LLM(**llm_kwargs)
        self.engine_init_seconds = time.perf_counter() - engine_start
        dtype_name = str(dtype).lower()
        dtype_bytes = 2 if "16" in dtype_name else 4
        self.kv_bytes_per_token = kv_bytes_per_token(
            model_config,
            dtype_bytes=dtype_bytes,
            tensor_parallel_size=tensor_parallel_size,
        )

    def build_prompt(self, problem: str) -> tuple[str, list[int]]:
        content = f"{problem.strip()}\n\n{instruction_for_task(self.task)}"
        messages = [{"role": "user", "content": content}]
        template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
        }
        if self.model_type.startswith("qwen3"):
            template_kwargs["enable_thinking"] = self.enable_thinking
        prompt_ids = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        return content, list(prompt_ids)

    def _generate_window(
        self,
        base_prompt_ids: list[int],
        prefix_ids: list[int],
        window_size: int,
        sampling_seed: int,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        logprobs: int,
        presence_penalty: float,
    ) -> tuple[list[dict], float]:
        physical = self.physical_window_size or window_size
        physical = min(max(physical, 1), window_size)
        traces = []
        elapsed = 0.0
        offset = 0
        remaining = window_size
        while remaining > 0:
            batch = min(physical, remaining)
            chunk, batch_elapsed = self._generate_physical_batch(
                base_prompt_ids,
                prefix_ids,
                batch,
                sampling_seed + offset,
                temperature,
                top_p,
                top_k,
                max_tokens,
                logprobs,
                presence_penalty,
            )
            traces.extend(chunk)
            elapsed += batch_elapsed
            offset += batch
            remaining -= batch
        return traces, elapsed

    def _generate_physical_batch(
        self,
        base_prompt_ids: list[int],
        prefix_ids: list[int],
        batch_size: int,
        sampling_seed: int,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        logprobs: int,
        presence_penalty: float,
    ) -> tuple[list[dict], float]:
        sampling = self._sampling_params(
            n=batch_size,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            logprobs=logprobs,
            presence_penalty=presence_penalty,
            seed=sampling_seed,
        )
        prompt_ids = base_prompt_ids + prefix_ids
        before_kv = snapshot_vllm_gpu_kv(self.llm)
        start = time.perf_counter()
        request_outputs = self.llm.generate(
            [{"prompt_token_ids": prompt_ids}], sampling, use_tqdm=False
        )
        elapsed = time.perf_counter() - start
        after_kv = snapshot_vllm_gpu_kv(self.llm)
        request = request_outputs[0]
        cached_tokens = getattr(request, "num_cached_tokens", None)
        traces = [
            self._pack_output(
                output,
                prefix_ids,
                prompt_ids,
                cached_tokens,
                before_kv,
                after_kv,
            )
            for output in request.outputs
        ]
        return traces, elapsed

    def generate_continuations(
        self,
        base_prompt_ids: list[int],
        prefix_ids_list: list[list[int]],
        sampling_seed: int,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        logprobs: int,
        presence_penalty: float,
    ) -> tuple[list[dict], float]:
        """Continue distinct prefixes (n=1 each). Used by PoLR expansion."""
        if not prefix_ids_list:
            return [], 0.0
        physical = self.physical_window_size or len(prefix_ids_list)
        physical = min(max(physical, 1), len(prefix_ids_list))
        traces: list[dict] = []
        elapsed = 0.0
        offset = 0
        while offset < len(prefix_ids_list):
            chunk_prefixes = prefix_ids_list[offset : offset + physical]
            sampling = self._sampling_params(
                n=1,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                logprobs=logprobs,
                presence_penalty=presence_penalty,
                seed=sampling_seed + offset,
            )
            prompts = [
                {"prompt_token_ids": base_prompt_ids + prefix_ids}
                for prefix_ids in chunk_prefixes
            ]
            before_kv = snapshot_vllm_gpu_kv(self.llm)
            start = time.perf_counter()
            request_outputs = self.llm.generate(prompts, sampling, use_tqdm=False)
            elapsed += time.perf_counter() - start
            after_kv = snapshot_vllm_gpu_kv(self.llm)
            for prefix_ids, request in zip(chunk_prefixes, request_outputs):
                cached_tokens = getattr(request, "num_cached_tokens", None)
                prompt_ids = base_prompt_ids + prefix_ids
                traces.append(
                    self._pack_output(
                        request.outputs[0],
                        prefix_ids,
                        prompt_ids,
                        cached_tokens,
                        before_kv,
                        after_kv,
                    )
                )
            offset += len(chunk_prefixes)
        return traces, elapsed

    def _sampling_params(
        self,
        n: int,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        logprobs: int,
        presence_penalty: float,
        seed: int,
    ) -> SamplingParams:
        kwargs = {
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "presence_penalty": presence_penalty,
            "seed": seed,
        }
        if logprobs and logprobs > 0:
            kwargs["logprobs"] = logprobs
        return SamplingParams(**kwargs)

    def _pack_output(
        self,
        output,
        prefix_ids: list[int],
        prompt_ids: list[int],
        cached_tokens,
        before_kv: dict,
        after_kv: dict,
    ) -> dict:
        new_ids = list(output.token_ids)
        full_ids = prefix_ids + new_ids
        full_text = self.tokenizer.decode(full_ids, skip_special_tokens=False)
        suffix_text = self.tokenizer.decode(new_ids, skip_special_tokens=False)
        raw_answer = extract_last_boxed(full_text)
        if raw_answer is None and self.answer_fallback == "gsm8k":
            raw_answer = extract_gsm8k_numeric(full_text)
        sampled_logprobs, confidences = _step_logprobs(output)
        reasoning_tokens = _reasoning_length(
            full_ids, self.close_think_id, self.tokenizer
        )
        return {
            "full_token_ids": full_ids,
            "new_token_ids": new_ids,
            "sampled_logprobs": sampled_logprobs,
            "token_confidence": confidences,
            "full_text": full_text,
            "suffix_text": suffix_text,
            "raw_answer": raw_answer,
            "answer": canonicalize_vote_answer(self.task, raw_answer),
            "reasoning_tokens": reasoning_tokens,
            "finish_reason": output.finish_reason,
            "stop_reason": output.stop_reason,
            "generated_tokens": len(new_ids),
            "prompt_tokens": len(prompt_ids),
            "prefix_tokens": len(prefix_ids),
            "truncated": output.finish_reason == "length",
            "lgc_2048": lowest_group_confidence(confidences, 2048),
            "num_cached_tokens": cached_tokens,
            "gpu_cache_usage_before": before_kv.get("gpu_cache_usage"),
            "gpu_cache_usage_after": after_kv.get("gpu_cache_usage"),
            "gpu_kv_blocks_used": after_kv.get("gpu_kv_blocks_used"),
            "gpu_kv_blocks_total": after_kv.get("gpu_kv_blocks_total"),
            "gpu_kv_block_size": after_kv.get("gpu_kv_block_size"),
            "kv_bytes_per_token": self.kv_bytes_per_token,
        }

    def _select_prefix(
        self,
        traces: list[dict],
        modal_answer: str,
        current_prefix_ids: list[int],
        next_ratio: float,
    ) -> Optional[dict]:
        return select_shared_prefix(
            traces,
            modal_answer,
            current_prefix_ids,
            next_ratio,
            lambda ids: self.tokenizer.decode(ids, skip_special_tokens=False),
            selector="blc",
        )

    def run_question(self, question_id: str, problem: str, method: str, **kwargs) -> dict:
        return run_sc_pc_question(
            self,
            question_id=question_id,
            problem=problem,
            method=method,
            **kwargs,
        )
