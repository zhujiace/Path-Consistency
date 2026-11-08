#!/usr/bin/env python3
"""Measure pairwise similarity of cumulative SC reasoning prefixes."""

from __future__ import annotations

import math
import argparse
import glob
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from path_consistency.math_eval import canonicalize_vote_answer


RATIOS = (0.2, 0.4, 0.6, 0.8)
DEFAULT_SC_GLOB = (
    "/data/26xyw/path-consistency-results/gsm8k/"
    "gsm8k_llama31_b32_mt2048_c08_seed0_shard*_*_sc.jsonl"
)
DEFAULT_TOKENIZER = (
    "/data/26xyw/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/"
    "snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)
DEFAULT_JSON_OUT = ROOT / "paper/ICLR/data/sc_prefix_similarity_snapshot.json"
DEFAULT_TABLE_OUT = ROOT / "paper/ICLR/tables/sc_prefix_similarity_rows.tex"
DEFAULT_SVG_OUT = ROOT / "paper/ICLR/figures/sc_prefix_similarity.svg"


_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[+\-*/=<>]")


def fractional_prefixes(text, ratios, tokenizer):
    """Decode cumulative prefixes at ceil(ratio * token length)."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return {ratio: "" for ratio in ratios}
    prefixes = {}
    for ratio in ratios:
        if not 0 < ratio <= 1:
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")
        endpoint = min(len(token_ids), max(1, math.ceil(ratio * len(token_ids))))
        prefixes[ratio] = tokenizer.decode(
            token_ids[:endpoint],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    return prefixes


def normalize_reasoning_tokens(text: str) -> tuple[str, ...]:
    """Return lowercase words, numbers, and arithmetic operators."""
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    normalized = normalized.replace("\\left", " ").replace("\\right", " ")
    return tuple(_TOKEN_RE.findall(normalized))


def word_ngram_counts(tokens: Sequence[str], n: int = 2) -> Counter:
    """Count ordered word n-grams, falling back to unigrams for short text."""
    if n < 1:
        raise ValueError("n must be positive")
    if not tokens:
        return Counter()
    if len(tokens) < n:
        return Counter((token,) for token in tokens)
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def cosine_counts(left: Mapping, right: Mapping) -> float:
    """Cosine similarity between two sparse count mappings."""
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def grouped_pairwise_similarity(vectors: Sequence[Mapping], answers: Sequence[str | None]) -> dict:
    """Average all unordered pairs and answer-stratified subsets."""
    if len(vectors) != len(answers):
        raise ValueError("vectors and answers must have the same length")
    buckets = {"overall": [], "same": [], "different": []}
    for left_index in range(len(vectors)):
        for right_index in range(left_index + 1, len(vectors)):
            score = cosine_counts(vectors[left_index], vectors[right_index])
            buckets["overall"].append(score)
            left_answer, right_answer = answers[left_index], answers[right_index]
            if left_answer is None or right_answer is None:
                continue
            if left_answer == right_answer:
                buckets["same"].append(score)
            else:
                buckets["different"].append(score)
    result = {}
    for name, scores in buckets.items():
        result[name] = fmean(scores) if scores else None
        result[name + "_n"] = len(scores)
    return result


def mean_between_similarity(left: Sequence[Mapping], right: Sequence[Mapping]) -> float:
    """Average cosine over every ordered cross-set pair."""
    if not left or not right:
        raise ValueError("both vector sets must be nonempty")
    return fmean(cosine_counts(a, b) for a in left for b in right)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def bootstrap_mean_ci(
    values: Iterable[float], repetitions: int = 5000, seed: int = 2027
) -> dict:
    """Question-cluster bootstrap percentile interval for a macro mean."""
    sample = [float(value) for value in values]
    if not sample:
        raise ValueError("bootstrap requires at least one value")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    rng = random.Random(seed)
    means = []
    for _ in range(repetitions):
        means.append(fmean(sample[rng.randrange(len(sample))] for _ in sample))
    means.sort()
    return {
        "n": len(sample),
        "mean": fmean(sample),
        "ci95": [_percentile(means, 0.025), _percentile(means, 0.975)],
        "bootstrap_repetitions": repetitions,
        "seed": seed,
    }


def _trace_answer(trace: Mapping) -> str | None:
    raw = trace.get("raw_answer")
    if raw is None:
        raw = trace.get("answer")
    return canonicalize_vote_answer("gsm8k", raw)


def branch_sort_key(trace: Mapping) -> tuple[int, int, str]:
    raw = trace.get("branch_id")
    match = re.search(r":w(\d+):b(\d+)$", str(raw))
    if match:
        return (int(match.group(1)), int(match.group(2)), str(raw))
    if isinstance(raw, int):
        return (int(trace.get("window") or 0), raw, str(raw))
    raise ValueError(f"unrecognized branch_id format: {raw!r}")


def _question_features(row: Mapping, tokenizer, ratios: Sequence[float]) -> dict:
    traces = sorted(row.get("traces") or [], key=branch_sort_key)
    vectors = {ratio: [] for ratio in ratios}
    answers = []
    truncated = []
    missing_text = 0
    for trace in traces:
        text = trace.get("full_text")
        if not isinstance(text, str) or not text:
            missing_text += 1
            text = ""
        prefixes = fractional_prefixes(text, ratios, tokenizer)
        for ratio in ratios:
            vectors[ratio].append(word_ngram_counts(normalize_reasoning_tokens(prefixes[ratio])))
        answers.append(_trace_answer(trace))
        truncated.append(bool(trace.get("truncated")))
    return {
        "vectors": vectors,
        "answers": tuple(answers),
        "truncated": tuple(truncated),
        "missing_text": missing_text,
        "n_traces": len(traces),
    }


def _aligned_cross_similarity(left: Sequence[Mapping], right: Sequence[Mapping]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("cross-question vectors must be nonempty and aligned")
    return fmean(cosine_counts(a, b) for a, b in zip(left, right))


def _attach_cross_baseline(
    record: dict,
    left: Mapping,
    right: Mapping,
    ratios,
    partner_source_row: int,
) -> None:
    record["baseline_partner_source_row"] = int(partner_source_row)
    for ratio in ratios:
        key = f"{ratio:.1f}"
        record["by_ratio"][key]["cross_question"] = _aligned_cross_similarity(
            left["vectors"][ratio], right["vectors"][ratio]
        )


def _summary_for_field(
    records: Sequence[Mapping],
    ratio_key: str,
    field: str,
    repetitions: int,
    seed: int,
) -> dict | None:
    values = [record["by_ratio"][ratio_key].get(field) for record in records]
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return bootstrap_mean_ci(values, repetitions=repetitions, seed=seed)


def analyze_rows(
    rows: Iterable[Mapping],
    tokenizer,
    ratios: Sequence[float] = RATIOS,
    expected_traces: int = 32,
    bootstrap_repetitions: int = 5000,
    seed: int = 2027,
    progress_every: int = 0,
) -> dict:
    """Compute question-macro similarity and a cyclic cross-question baseline."""
    ordered = sorted(rows, key=lambda row: int(row["source_row"]))
    if not ordered:
        raise ValueError("no rows supplied")
    source_rows = [int(row["source_row"]) for row in ordered]
    if len(source_rows) != len(set(source_rows)):
        raise ValueError("duplicate source_row values")

    processing_order = list(ordered)
    random.Random(seed).shuffle(processing_order)
    records = []
    first_features = None
    previous_features = None
    previous_record = None
    trace_count = 0
    truncated_count = 0
    missing_text_count = 0
    invalid_answer_count = 0

    first_source_row = None
    for index, row in enumerate(processing_order, start=1):
        features = _question_features(row, tokenizer, ratios)
        if features["n_traces"] != expected_traces:
            raise ValueError(
                f"source_row={row['source_row']} has {features['n_traces']} traces, "
                f"expected {expected_traces}"
            )
        trace_count += features["n_traces"]
        truncated_count += sum(features["truncated"])
        missing_text_count += features["missing_text"]
        invalid_answer_count += sum(answer is None for answer in features["answers"])

        record = {
            "source_row": int(row["source_row"]),
            "question_id": row.get("question_id"),
            "by_ratio": {},
        }
        for ratio in ratios:
            key = f"{ratio:.1f}"
            vectors = features["vectors"][ratio]
            grouped = grouped_pairwise_similarity(vectors, features["answers"])
            kept = [position for position, flag in enumerate(features["truncated"]) if not flag]
            untruncated = None
            if len(kept) >= 2:
                reduced_vectors = [vectors[position] for position in kept]
                reduced_answers = [features["answers"][position] for position in kept]
                untruncated = grouped_pairwise_similarity(
                    reduced_vectors, reduced_answers
                )["overall"]
            record["by_ratio"][key] = {
                "within_question": grouped["overall"],
                "within_pairs": grouped["overall_n"],
                "same_answer": grouped["same"],
                "same_answer_pairs": grouped["same_n"],
                "different_answer": grouped["different"],
                "different_answer_pairs": grouped["different_n"],
                "untruncated": untruncated,
            }

        if first_features is None:
            first_features = features
            first_source_row = int(row["source_row"])
        if previous_features is not None:
            _attach_cross_baseline(
                previous_record,
                previous_features,
                features,
                ratios,
                int(row["source_row"]),
            )
        records.append(record)
        previous_features = features
        previous_record = record
        if progress_every and (index % progress_every == 0 or index == len(processing_order)):
            print(f"processed {index}/{len(processing_order)} questions", flush=True)

    _attach_cross_baseline(
        previous_record,
        previous_features,
        first_features,
        ratios,
        first_source_row,
    )
    records.sort(key=lambda record: record["source_row"])

    by_ratio = {}
    fields = (
        "within_question",
        "cross_question",
        "same_answer",
        "different_answer",
        "untruncated",
    )
    for ratio_index, ratio in enumerate(ratios):
        key = f"{ratio:.1f}"
        block = {}
        for field_index, field in enumerate(fields):
            summary = _summary_for_field(
                records,
                key,
                field,
                bootstrap_repetitions,
                seed + ratio_index * 100 + field_index,
            )
            if summary is not None:
                pair_field = field + "_pairs"
                if field in ("same_answer", "different_answer"):
                    summary["total_pairs"] = sum(
                        int(record["by_ratio"][key][pair_field]) for record in records
                    )
                    summary["questions_with_pairs"] = sum(
                        record["by_ratio"][key][field] is not None for record in records
                    )
            block[field] = summary
        differences = [
            record["by_ratio"][key]["within_question"]
            - record["by_ratio"][key]["cross_question"]
            for record in records
        ]
        block["difference"] = bootstrap_mean_ci(
            differences,
            repetitions=bootstrap_repetitions,
            seed=seed + ratio_index * 100 + 50,
        )
        block["within_pairs"] = sum(
            int(record["by_ratio"][key]["within_pairs"]) for record in records
        )
        block["cross_pairs"] = len(records) * expected_traces
        by_ratio[key] = block

    return {
        "data_quality": {
            "questions": len(records),
            "traces": trace_count,
            "traces_per_question": expected_traces,
            "truncated_traces": truncated_count,
            "missing_text_traces": missing_text_count,
            "invalid_answer_traces": invalid_answer_count,
        },
        "by_ratio": by_ratio,
        "per_question": records,
    }


def _tex_interval(block: Mapping | None) -> str:
    if block is None:
        return "--"
    return f"{block['mean']:.3f} [{block['ci95'][0]:.3f}, {block['ci95'][1]:.3f}]"


def render_tex_rows(summary: Mapping) -> str:
    lines = ["% Generated by analysis/sc_prefix_similarity.py; question-macro bootstrap CIs."]
    for key in sorted(summary["by_ratio"], key=float):
        block = summary["by_ratio"][key]
        lines.append(
            f"{100 * float(key):.0f}\\% & "
            f"{_tex_interval(block['within_question'])} & "
            f"{_tex_interval(block['cross_question'])} & "
            f"{_tex_interval(block['difference'])} & "
            f"{_tex_interval(block['same_answer'])} & "
            f"{_tex_interval(block['different_answer'])} \\\\"
        )
    return "\n".join(lines) + "\n"


def render_svg(summary: Mapping) -> str:
    width, height = 760, 430
    left, right, top, bottom = 82, 28, 30, 68
    plot_width, plot_height = width - left - right, height - top - bottom
    keys = sorted(summary["by_ratio"], key=float)
    xs = [left + plot_width * index / (len(keys) - 1) for index in range(len(keys))]

    def y(value):
        return top + plot_height * (1.0 - float(value))

    def points(field, ci_index=None):
        values = []
        for key in keys:
            block = summary["by_ratio"][key][field]
            value = block["mean"] if ci_index is None else block["ci95"][ci_index]
            values.append(value)
        return " ".join(f"{x:.1f},{y(value):.1f}" for x, value in zip(xs, values))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, Helvetica, sans-serif" fill="#222">',
    ]
    for tick in range(0, 11, 2):
        value = tick / 10
        py = y(value)
        parts.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + plot_width}" y2="{py:.1f}" '
            'stroke="#E7E9ED" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{py + 5:.1f}" text-anchor="end" font-size="15">{value:.1f}</text>'
        )
    parts.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333" stroke-width="1.2"/>',
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333" stroke-width="1.2"/>',
        ]
    )
    for x, key in zip(xs, keys):
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_height + 28}" text-anchor="middle" font-size="16">{100 * float(key):.0f}%</text>'
        )

    upper = points("within_question", 1).split()
    lower = list(reversed(points("within_question", 0).split()))
    parts.append(
        f'<polygon points="{" ".join(upper + lower)}" fill="#625BD6" fill-opacity="0.16" stroke="none"/>'
    )
    parts.append(
        f'<polyline points="{points("within_question")}" fill="none" stroke="#625BD6" stroke-width="3"/>'
    )
    parts.append(
        f'<polyline points="{points("cross_question")}" fill="none" stroke="#7B818A" stroke-width="2.5" stroke-dasharray="8 6"/>'
    )
    for x, key in zip(xs, keys):
        within = summary["by_ratio"][key]["within_question"]["mean"]
        cross = summary["by_ratio"][key]["cross_question"]["mean"]
        parts.append(f'<circle cx="{x:.1f}" cy="{y(within):.1f}" r="4.5" fill="#625BD6"/>')
        parts.append(f'<circle cx="{x:.1f}" cy="{y(cross):.1f}" r="3.8" fill="#7B818A"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{y(within) - 11:.1f}" text-anchor="middle" font-size="14" fill="#5049C8">{within:.3f}</text>'
        )

    legend_x, legend_y = left + 22, top + 23
    parts.extend(
        [
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 34}" y2="{legend_y}" stroke="#625BD6" stroke-width="3"/>',
            f'<text x="{legend_x + 43}" y="{legend_y + 5}" font-size="15">Within-question SC pairs</text>',
            f'<line x1="{legend_x + 250}" y1="{legend_y}" x2="{legend_x + 284}" y2="{legend_y}" stroke="#7B818A" stroke-width="2.5" stroke-dasharray="8 6"/>',
            f'<text x="{legend_x + 293}" y="{legend_y + 5}" font-size="15">Cross-question baseline</text>',
            f'<text x="{left + plot_width / 2:.1f}" y="{height - 17}" text-anchor="middle" font-size="17">Relative position in trajectory</text>',
            f'<text x="21" y="{top + plot_height / 2:.1f}" text-anchor="middle" font-size="17" transform="rotate(-90 21 {top + plot_height / 2:.1f})">Mean word-bigram cosine similarity</text>',
            "</g>",
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(paths: Sequence[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", default=DEFAULT_SC_GLOB)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--table-out", type=Path, default=DEFAULT_TABLE_OUT)
    parser.add_argument("--svg-out", type=Path, default=DEFAULT_SVG_OUT)
    parser.add_argument("--expected-questions", type=int, default=1319)
    parser.add_argument("--expected-traces", type=int, default=32)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    paths = sorted(Path(path) for path in glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"no files matched {args.glob}")
    rows = _read_jsonl(paths)
    if len(rows) != args.expected_questions:
        raise SystemExit(f"found {len(rows)} questions, expected {args.expected_questions}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, use_fast=True
    )
    analysis = analyze_rows(
        rows,
        tokenizer=tokenizer,
        ratios=RATIOS,
        expected_traces=args.expected_traces,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
        progress_every=args.progress_every,
    )
    payload = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "verification_status": "ANALYZED",
            "version_label": "sc_prefix_similarity_v1",
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "model": "Llama-3.1-8B-Instruct",
            "dataset": "GSM8K test",
            "method": "SC",
            "ratios": list(RATIOS),
            "metric": "cosine similarity of normalized word-bigram count vectors",
            "aggregation": "all unordered pairs within each question, then question-macro mean",
            "cross_question_baseline": (
                "same branch index across a seeded random permutation of questions, "
                "paired as a cycle without self-pairs"
            ),
            "confidence_interval": "question-cluster percentile bootstrap",
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "seed": args.seed,
            "claim_boundary": (
                "Measures surface-level reasoning overlap; does not establish semantic equivalence, "
                "prefix correctness, or safe reuse."
            ),
        },
        "inputs": {
            "tokenizer": str(Path(args.tokenizer).resolve()),
            "source_glob": args.glob,
            "source_files": [
                {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in paths
            ],
        },
        **analysis,
    }
    for output in (args.json_out, args.table_out, args.svg_out):
        output.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    args.table_out.write_text(render_tex_rows(payload))
    args.svg_out.write_text(render_svg(payload))
    print(
        json.dumps(
            {
                "json": str(args.json_out),
                "table": str(args.table_out),
                "svg": str(args.svg_out),
                "questions": payload["data_quality"]["questions"],
                "traces": payload["data_quality"]["traces"],
                "by_ratio": {
                    key: {
                        "within": value["within_question"]["mean"],
                        "cross": value["cross_question"]["mean"],
                        "difference": value["difference"]["mean"],
                    }
                    for key, value in payload["by_ratio"].items()
                },
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
