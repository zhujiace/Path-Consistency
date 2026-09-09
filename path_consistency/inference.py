"""vLLM-backed parallel inference for self-consistency / path-consistency.

Parallelism design (per the user requirement): every branch of one prefix
level (window) shares the same prompt, so one vLLM call with n=window_size
generates the whole window at once. Self-consistency uses the same windowed
batching so that per-level latency and token stats are comparable.
"""

import time
from collections import Counter

from vllm import LLM, SamplingParams

from .answers import extract_answer, integrate_answer
from .confidence import beta_confidence
from .prefix import get_prefix, sample_prefix


class PCInference:
    def __init__(
        self,
        model_path: str,
        max_branch: int = 20,
        max_level: int = 3,
        confidence_thres: float = 0.8,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 512,
        gpu_memory_utilization: float = 0.9,
        seed: int = 0,
        early_stop: str = None,     # None | "ac" | "esc" | "margin"
        early_stop_thresh: float = 0.9,   # beta confidence for "ac"
        margin_min: int = 2,              # top1-top2 count margin for "margin"
    ):
        self.max_branch = max_branch
        self.max_level = max_level
        self.confidence_thres = confidence_thres
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = seed
        self.early_stop = early_stop
        self.early_stop_thresh = early_stop_thresh if early_stop_thresh is not None else 0.9
        self.margin_min = margin_min
        self.gpu_memory_utilization = gpu_memory_utilization

        self.llm = LLM(
            model=model_path,
            dtype="float16",
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=4096,
            seed=seed,
        )

    # ------------------------------------------------------------------ #
    def _gen_batch(self, prompts: list[str], n: int):
        """Generate n samples for each prompt (one batch per window).

        Returns (output_texts, wall_time, tokens_per_sample).
        """
        sampling = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            n=n,
            stop=["\nQ:"],  # halt when the model starts the "next" exemplar
        )
        t0 = time.time()
        outputs = self.llm.generate(prompts, sampling, use_tqdm=False)
        elapsed = time.time() - t0

        # one prompt per window -> outputs has a single entry
        texts = [o.text for o in outputs[0].outputs]
        token_counts = [len(o.token_ids) for o in outputs[0].outputs]
        return texts, elapsed, token_counts

    # ------------------------------------------------------------------ #
    def _should_stop(self, window_answers: list, answers_all: list) -> bool:
        if self.early_stop == "ac":
            return beta_confidence(answers_all) >= self.early_stop_thresh
        if self.early_stop == "esc":
            # stop when every branch of the current window agrees (ICLR'24)
            return len(set(window_answers)) == 1
        if self.early_stop == "margin":
            c = Counter(answers_all).most_common(2)
            if len(c) == 1:
                return len(answers_all) >= 2
            return (c[0][1] - c[1][1]) >= self.margin_min and len(answers_all) >= 5
        return False

    def inference(self, prompt: str):
        """Run self-consistency (threshold >= 1.0), path-consistency, or an
        early-stopping baseline (AC/ESC/margin) at window granularity."""
        window = self.max_branch // (self.max_level + 1)
        n_windows = self.max_branch // window

        answers = []          # per-branch extracted answers
        reasoning = []        # prefix + generation (full path), for extraction
        generations = []      # only the newly generated text
        times = []            # per-branch latency (window wall-time / n)
        tokens_gen = []       # newly generated tokens per branch
        prefix_levels = []    # prefix level used by each branch
        n_branches = 0

        prefix_list = []
        prefix_level = 0
        prefixes_used = []   # prefix string used by each branch (window)

        for w in range(n_windows):
            # one window = one prompt (same prefix level), generated in parallel
            prefix = sample_prefix(prefix_list)
            prompt_plus_prefix = prompt + prefix
            texts, elapsed, toks = self._gen_batch([prompt_plus_prefix], n=window)
            n_branches += len(texts)

            win_answers = []
            for i in range(window):
                gen = texts[i]
                reasoning.append(prefix + gen)
                generations.append(gen)
                times.append(elapsed / window)
                tokens_gen.append(toks[i])
                prefix_levels.append(prefix_level)
                prefixes_used.append(prefix)
                a = extract_answer(gen)
                answers.append(a)
                win_answers.append(a)

            if self._should_stop(win_answers, answers):
                break

            if self.early_stop is None and w < n_windows - 1:
                # window boundary: gate + extract (PC only)
                prefix_list, prefix_level, _ = get_prefix(
                    answers, self.confidence_thres, reasoning,
                    prefix_list, prefix_level, frq=window,
                )

        final_answer = integrate_answer(answers)
        return {
            "answer": final_answer,
            "answers": answers,
            "latency": times,
            "generations": generations,
            "tokens_gen": tokens_gen,
            "prefix_level": prefix_levels,
            "prefixes_used": prefixes_used,
            "n_branches": n_branches,
        }
