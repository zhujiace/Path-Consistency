#!/usr/bin/env python3
"""Create a compact paired SC/PC/DeepConf report from MATH JSONL outputs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


DISPLAY_NAMES = {
    "DeepConf-high": "Offline-LGC@90%",
    "DeepConf-low": "Offline-LGC@10%",
    "DeepConf-high+PC": "Offline-LGC@90%+PC",
    "DeepConf-low+PC": "Offline-LGC@10%+PC",
}

OFFLINE_POLICIES = [
    "Offline-Mean",
    "Offline-Bottom10@90%",
    "Offline-Bottom10@10%",
    "Offline-Tail@90%",
    "Offline-Tail@10%",
    "Offline-LGC@90%",
    "Offline-LGC@10%",
]


def load(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["question_id"]] = row
    return rows


def percentile(values: list[float], quantile: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    position = (len(values) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(values) - 1)
    fraction = position - low
    return values[low] * (1 - fraction) + values[high] * fraction


def paired_bootstrap(
    pairs: list[tuple[float, float]], samples: int = 10_000, seed: int = 2027
) -> tuple[float, float]:
    if not pairs:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        chosen = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        estimates.append(sum(right - left for left, right in chosen) / len(chosen))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def method_row(rows: list[dict], prediction: str, parent: str) -> dict:
    eligible = [row for row in rows if prediction in row.get("predictions", {})]
    correct = sum(bool(row["predictions"][prediction]["correct"]) for row in eligible)
    return {
        "method": DISPLAY_NAMES.get(prediction, prediction),
        "n": len(eligible),
        "correct": correct,
        "accuracy": correct / len(eligible) if eligible else None,
        "avg_decode_tokens": (
            sum(row["total_generated_tokens"] for row in eligible) / len(eligible)
            if eligible
            else None
        ),
        "avg_logical_path_tokens": (
            sum(
                row["total_generated_tokens"] + row["total_prompt_tokens"]
                for row in eligible
            )
            / len(eligible)
            if eligible
            else None
        ),
        "avg_vllm_submitted_tokens": (
            sum(
                row["total_generated_tokens"]
                + row.get(
                    "total_request_prompt_tokens",
                    sum(
                        row["prompt_tokens"] + event.get("used_prefix_tokens", 0)
                        for event in row.get("windows", [])
                    ),
                )
                for row in eligible
            )
            / len(eligible)
            if eligible
            else None
        ),
        "avg_generation_seconds": (
            sum(row["generation_seconds"] for row in eligible) / len(eligible)
            if eligible
            else None
        ),
        "parser_failures": sum(row["parser_failures"] for row in eligible),
        "truncations": sum(row["truncations"] for row in eligible),
        "cost_parent": parent,
    }


def build_report(sc_rows: dict[str, dict], pc_rows: dict[str, dict]) -> dict:
    common = sorted(set(sc_rows) & set(pc_rows))
    sc = [sc_rows[key] for key in common]
    pc = [pc_rows[key] for key in common]
    table = []
    sc_names = ["SC"] + OFFLINE_POLICIES + ["DeepConf-high", "DeepConf-low"]
    pc_names = ["PC"] + [f"{name}+PC" for name in OFFLINE_POLICIES] + [
        "DeepConf-high+PC",
        "DeepConf-low+PC",
    ]
    for name in sc_names:
        if any(name in row.get("predictions", {}) for row in sc):
            table.append(method_row(sc, name, "SC"))
    for name in pc_names:
        if any(name in row.get("predictions", {}) for row in pc):
            table.append(method_row(pc, name, "PC"))

    accuracy_pairs = [
        (
            float(left["predictions"]["SC"]["correct"]),
            float(right["predictions"]["PC"]["correct"]),
        )
        for left, right in zip(sc, pc)
    ]
    accuracy_ci = paired_bootstrap(accuracy_pairs)
    sc_decode = sum(row["total_generated_tokens"] for row in sc)
    pc_decode = sum(row["total_generated_tokens"] for row in pc)
    sc_logical = sum(
        row["total_generated_tokens"] + row["total_prompt_tokens"] for row in sc
    )
    pc_logical = sum(
        row["total_generated_tokens"] + row["total_prompt_tokens"] for row in pc
    )
    sc_submitted = sum(
        row["total_generated_tokens"]
        + row.get(
            "total_request_prompt_tokens",
            sum(
                row["prompt_tokens"] + event.get("used_prefix_tokens", 0)
                for event in row.get("windows", [])
            ),
        )
        for row in sc
    )
    pc_submitted = sum(
        row["total_generated_tokens"]
        + row.get(
            "total_request_prompt_tokens",
            sum(
                row["prompt_tokens"] + event.get("used_prefix_tokens", 0)
                for event in row.get("windows", [])
            ),
        )
        for row in pc
    )
    sc_time = sum(row["generation_seconds"] for row in sc)
    pc_time = sum(row["generation_seconds"] for row in pc)
    prefix_windows = sum(
        event.get("used_prefix_tokens", 0) > 0
        for row in pc
        for event in row.get("windows", [])
    )
    total_windows = sum(len(row.get("windows", [])) for row in pc)
    decisions = {}
    for row in pc:
        for event in row.get("windows", []):
            decision = event.get("decision", "unknown")
            decisions[decision] = decisions.get(decision, 0) + 1

    transitions = {
        "both_correct": 0,
        "sc_only_correct": 0,
        "pc_only_correct": 0,
        "both_wrong": 0,
    }
    per_question = []
    for left, right in zip(sc, pc):
        sc_correct = bool(left["predictions"]["SC"]["correct"])
        pc_correct = bool(right["predictions"]["PC"]["correct"])
        if sc_correct and pc_correct:
            transitions["both_correct"] += 1
        elif sc_correct:
            transitions["sc_only_correct"] += 1
        elif pc_correct:
            transitions["pc_only_correct"] += 1
        else:
            transitions["both_wrong"] += 1
        left_tokens = left["total_generated_tokens"]
        right_tokens = right["total_generated_tokens"]
        left_seconds = left["generation_seconds"]
        right_seconds = right["generation_seconds"]
        events = right.get("windows", [])
        per_question.append(
            {
                "question_id": left["question_id"],
                "subject": left.get("subject"),
                "level": left.get("level"),
                "gold": left.get("gold"),
                "sc_answer": left["predictions"]["SC"]["answer"],
                "pc_answer": right["predictions"]["PC"]["answer"],
                "sc_correct": sc_correct,
                "pc_correct": pc_correct,
                "sc_decode_tokens": left_tokens,
                "pc_decode_tokens": right_tokens,
                "decode_token_reduction": (
                    1 - right_tokens / left_tokens if left_tokens else None
                ),
                "sc_generation_seconds": left_seconds,
                "pc_generation_seconds": right_seconds,
                "generation_time_reduction": (
                    1 - right_seconds / left_seconds if left_seconds else None
                ),
                "prefix_windows": sum(
                    event.get("used_prefix_tokens", 0) > 0 for event in events
                ),
                "decisions": [event.get("decision", "unknown") for event in events],
                "final_prefix_ratio": right.get("final_prefix_ratio"),
                "sc_truncations": left.get("truncations", 0),
                "pc_truncations": right.get("truncations", 0),
            }
        )

    return {
        "paired_questions": len(common),
        "question_ids": common,
        "table": table,
        "pc_minus_sc_accuracy": (
            sum(right - left for left, right in accuracy_pairs) / len(accuracy_pairs)
            if accuracy_pairs
            else None
        ),
        "pc_minus_sc_accuracy_bootstrap_95ci": accuracy_ci,
        "decode_token_reduction": 1 - pc_decode / sc_decode if sc_decode else None,
        "logical_path_token_reduction": (
            1 - pc_logical / sc_logical if sc_logical else None
        ),
        "vllm_submitted_token_reduction": (
            1 - pc_submitted / sc_submitted if sc_submitted else None
        ),
        "generation_time_reduction": 1 - pc_time / sc_time if sc_time else None,
        "prefix_window_rate": prefix_windows / total_windows if total_windows else None,
        "pc_window_decisions": decisions,
        "accuracy_transitions": transitions,
        "per_question": per_question,
    }


def markdown(report: dict) -> str:
    lines = [
        "# MATH500 初始实验汇总",
        "",
        f"配对完成题数：{report['paired_questions']}。正值 token/time reduction 表示 PC 更省。",
        "",
        "| Method | Correct/N | Accuracy | Decode tokens/q | vLLM submitted tokens/q | Gen sec/q | Parser fail | Trunc. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["table"]:
        accuracy = "—" if row["accuracy"] is None else f"{100 * row['accuracy']:.1f}%"
        decode = "—" if row["avg_decode_tokens"] is None else f"{row['avg_decode_tokens']:.1f}"
        submitted = "—" if row["avg_vllm_submitted_tokens"] is None else f"{row['avg_vllm_submitted_tokens']:.1f}"
        seconds = "—" if row["avg_generation_seconds"] is None else f"{row['avg_generation_seconds']:.2f}"
        lines.append(
            f"| {row['method']} | {row['correct']}/{row['n']} | {accuracy} | "
            f"{decode} | {submitted} | {seconds} | {row['parser_failures']} | {row['truncations']} |"
        )
    lo, hi = report["pc_minus_sc_accuracy_bootstrap_95ci"]
    lines.extend(
        [
            "",
            "## 配对诊断",
            "",
            f"- PC−SC accuracy：{100 * report['pc_minus_sc_accuracy']:.2f} pp；题级 bootstrap 95% CI [{100 * lo:.2f}, {100 * hi:.2f}] pp。",
            f"- Decode-token reduction：{100 * report['decode_token_reduction']:.2f}%。",
            f"- vLLM submitted-token reduction（decode + 每窗口一次 prompt；未扣 prefix-cache 命中）：{100 * report['vllm_submitted_token_reduction']:.2f}%。",
            f"- Logical-path token reduction（按每条分支重复计算 prompt）：{100 * report['logical_path_token_reduction']:.2f}%。",
            f"- Generation-time reduction：{100 * report['generation_time_reduction']:.2f}%。",
            f"- 使用非空共享前缀的窗口比例：{100 * report['prefix_window_rate']:.2f}%。",
            f"- PC 窗口决策：`{json.dumps(report['pc_window_decisions'], ensure_ascii=False, sort_keys=True)}`。",
            f"- 准确率转移：`{json.dumps(report['accuracy_transitions'], ensure_ascii=False, sort_keys=True)}`。",
            "",
            "## 逐题配对结果",
            "",
            "| Question | Subject/Lv. | SC→PC | Correct | Decode SC→PC | Token Δ | Time SC→PC (s) | Prefix windows | Decisions |",
            "|---|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["per_question"]:
        reduction = row["decode_token_reduction"]
        reduction_text = "—" if reduction is None else f"{100 * reduction:.1f}%"
        correctness = f"{int(row['sc_correct'])}→{int(row['pc_correct'])}"
        answers = f"{row['sc_answer']}→{row['pc_answer']}"
        decisions = ", ".join(row["decisions"])
        lines.append(
            f"| `{row['question_id']}` | {row['subject']}/L{row['level']} | "
            f"{answers} | {correctness} | {row['sc_decode_tokens']}→{row['pc_decode_tokens']} | "
            f"{reduction_text} | {row['sc_generation_seconds']:.2f}→{row['pc_generation_seconds']:.2f} | "
            f"{row['prefix_windows']} | {decisions} |"
        )
    lines.extend(
        [
            "",
            "> 该表是实现与决策 pilot；小样本 bootstrap 区间不用于论文显著性结论。DeepConf 行是同一完整 trace pool 上的离线聚合视图，不代表 online early stopping 的额外 token 节省。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sc", type=Path, required=True)
    parser.add_argument("--pc", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    report = build_report(load(args.sc), load(args.pc))
    if args.json_output:
        args.json_output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if args.markdown_output:
        args.markdown_output.write_text(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()
