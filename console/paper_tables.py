"""Planned ICLR 2027 paper tables filled from complete protocol runs.

Empty cells stay empty. A cell is filled only when a B=32 paired SC/PC run
covers the planned evaluation set in full. C=0.8 numbers are provisional,
not Dev-Mix C*. Llama MATH-50, smokes, and old Sentence-PC are excluded.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


MODELS = [
    {"id": "llama31", "label": "Llama-3.1-8B-Inst."},
    {"id": "qwen30", "label": "Qwen3-30B-A3B"},
    {"id": "deepseek", "label": "DeepSeek-V4-Flash"},
]

DATASETS_T1 = [
    {"id": "gsm8k", "label": "GSM8K", "n": 1319, "kind": "continuity"},
    {"id": "math500", "label": "MATH-500", "n": 490, "kind": "main"},
    {"id": "aime24", "label": "AIME 2024", "n": 30, "kind": "main"},
    {"id": "aime25", "label": "AIME 2025", "n": 30, "kind": "main"},
    {"id": "gpqa", "label": "GPQA-D", "n": 198, "kind": "main"},
]

METHODS_T1 = [
    {"id": "pass1", "label": "Pass@1"},
    {"id": "sc", "label": "SC"},
    {"id": "pc", "label": "PC"},
]

EXPECTED_N = {row["id"]: row["n"] for row in DATASETS_T1}

TABLE3_PLANNED = [
    {
        "model_id": "qwen30",
        "dataset_id": "aime_combined",
        "model": "Qwen3-30B-A3B",
        "dataset": "AIME24+25",
        "parent": "SC",
        "plus": "SC+PC",
    },
    {
        "model_id": "qwen30",
        "dataset_id": "aime_combined",
        "model": "Qwen3-30B-A3B",
        "dataset": "AIME24+25",
        "parent": "DeepConf",
        "plus": "DeepConf+PC",
    },
    {
        "model_id": "qwen30",
        "dataset_id": "gpqa",
        "model": "Qwen3-30B-A3B",
        "dataset": "GPQA-D",
        "parent": "SC",
        "plus": "SC+PC",
    },
    {
        "model_id": "qwen30",
        "dataset_id": "gpqa",
        "model": "Qwen3-30B-A3B",
        "dataset": "GPQA-D",
        "parent": "DeepConf",
        "plus": "DeepConf+PC",
    },
    {
        "model_id": "deepseek",
        "dataset_id": "aime_combined",
        "model": "DeepSeek-V4-Flash",
        "dataset": "AIME24+25",
        "parent": "SC",
        "plus": "SC+PC",
    },
    {
        "model_id": "deepseek",
        "dataset_id": "aime_combined",
        "model": "DeepSeek-V4-Flash",
        "dataset": "AIME24+25",
        "parent": "DeepConf-offline",
        "plus": "DeepConf-offline+PC",
    },
]


def _model_id(run: dict) -> str | None:
    blob = " ".join(
        str(run.get(key) or "")
        for key in ("model", "model_path", "model_type", "tag", "dataset")
    ).lower()
    if "deepseek" in blob:
        return "deepseek"
    if "30b-a3b" in blob or "30b_a3b" in blob or "qwen3-30b" in blob:
        return "qwen30"
    if "qwen3-32b" in blob or "qwen3_32b" in blob:
        return "qwen30"
    if "llama-3.1" in blob or "llama31" in blob or "llama_3.1" in blob:
        return "llama31"
    return None


def _dataset_id(run: dict) -> str | None:
    blob = " ".join(str(run.get(key) or "") for key in ("dataset", "tag")).lower()
    if "gsm8k" in blob:
        return "gsm8k"
    if "gpqa" in blob:
        return "gpqa"
    if "aime_2024" in blob or "aime24" in blob or "aime-2024" in blob:
        return "aime24"
    if "aime_2025" in blob or "aime25" in blob or "aime-2025" in blob:
        return "aime25"
    if "math500" in blob or "math-500" in blob or "math500_test" in blob:
        return "math500"
    return None


def _eligible_main_run(run: dict) -> bool:
    if (run.get("status") or "") != "complete":
        return False
    if (run.get("kind") or "") != "primary":
        return False
    tag = (run.get("tag") or "").lower()
    if any(
        part in tag
        for part in (
            "smoke",
            "diagnostic",
            "nothink",
            "pilot10",
            "initial50",
            "pairedroot",
        )
    ):
        return False
    if int(run.get("max_branches") or 0) != 32:
        return False
    planned = int(run.get("planned") or 0)
    paired = int(run.get("paired") or 0)
    if planned <= 0 or paired < planned:
        return False
    return _model_id(run) is not None and _dataset_id(run) is not None


def _ratio(correct: int, n: int) -> float | None:
    if n <= 0:
        return None
    return correct / n


def _cell(
    *,
    acc: float | None,
    correct: int | None,
    n: int | None,
    tok: float | None = None,
    tok_saving: float | None = None,
    latency_s: float | None = None,
    beta: float | None = None,
    source: str | None = None,
    note: str | None = None,
) -> dict[str, Any] | None:
    if acc is None or n is None or n <= 0:
        return None
    out: dict[str, Any] = {
        "acc": acc,
        "correct": correct,
        "n": n,
        "tok": tok,
        "tok_saving": tok_saving,
        "latency_s": latency_s,
        "beta": beta,
        "source": source,
        "note": note,
    }
    return out


def _merge_group(runs: list[dict]) -> dict[str, Any] | None:
    n_sc = sum(int(run.get("sc_questions") or 0) for run in runs)
    n_pc = sum(int(run.get("pc_questions") or 0) for run in runs)
    n_p1 = sum(int(run.get("pass1_questions") or 0) for run in runs)
    if not n_sc or n_sc != n_pc:
        return None
    sc_correct = sum(int(run.get("sc_correct") or 0) for run in runs)
    pc_correct = sum(int(run.get("pc_correct") or 0) for run in runs)
    p1_correct = sum(int(run.get("pass1_correct") or 0) for run in runs)
    sc_tok = sum(float(run.get("sc_total_generated") or 0) for run in runs)
    pc_tok = sum(float(run.get("pc_total_generated") or 0) for run in runs)
    sc_sec = sum(float(run.get("sc_total_generation_s") or 0) for run in runs)
    pc_sec = sum(float(run.get("pc_total_generation_s") or 0) for run in runs)
    sc_avg = sc_tok / n_sc if sc_tok else None
    pc_avg = pc_tok / n_pc if pc_tok else None
    tok_saving = (1.0 - pc_avg / sc_avg) if sc_avg and pc_avg else None
    betas = {run.get("beta_threshold") for run in runs}
    beta = next(iter(betas)) if len(betas) == 1 else None
    sources = ", ".join(run.get("tag") or "" for run in runs)
    note = None
    if beta is not None and abs(float(beta) - 1.0) > 1e-9:
        note = f"C={float(beta):.2g} provisional，不是 C*"
    return {
        "n": n_sc,
        "beta": None if beta is None else float(beta),
        "source": sources,
        "note": note,
        "pass1": _cell(
            acc=_ratio(p1_correct, n_p1),
            correct=p1_correct if n_p1 else None,
            n=n_p1 or None,
            tok=None,
            beta=None if beta is None else float(beta),
            source=sources,
        ),
        "sc": _cell(
            acc=_ratio(sc_correct, n_sc),
            correct=sc_correct,
            n=n_sc,
            tok=sc_avg,
            latency_s=sc_sec / n_sc if sc_sec else None,
            beta=None if beta is None else float(beta),
            source=sources,
            note=note,
        ),
        "pc": _cell(
            acc=_ratio(pc_correct, n_pc),
            correct=pc_correct,
            n=n_pc,
            tok=pc_avg,
            tok_saving=tok_saving,
            latency_s=pc_sec / n_pc if pc_sec else None,
            beta=None if beta is None else float(beta),
            source=sources,
            note=note,
        ),
    }


def collect_complete_cells(runs: list[dict]) -> dict[tuple[str, str], dict[str, Any]]:
    groups: dict[tuple[str, str, int, float], list[dict]] = defaultdict(list)
    for run in runs:
        if not _eligible_main_run(run):
            continue
        model = _model_id(run)
        dataset = _dataset_id(run)
        if model is None or dataset is None:
            continue
        beta = float(run.get("beta_threshold") or 0.0)
        key = (model, dataset, int(run.get("max_branches") or 0), round(beta, 2))
        groups[key].append(run)

    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for (model, dataset, _branches, _beta), group in groups.items():
        merged = _merge_group(group)
        if merged is None:
            continue
        expected = EXPECTED_N.get(dataset)
        if expected is not None and merged["n"] != expected:
            continue
        cells[(model, dataset)] = merged
    return cells


def _empty_row_cells() -> dict[str, None]:
    return {row["id"]: None for row in DATASETS_T1}


def build_table1(cells: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    rows = []
    filled = 0
    total = len(MODELS) * len(METHODS_T1) * len(DATASETS_T1)
    for model in MODELS:
        for method in METHODS_T1:
            row_cells = _empty_row_cells()
            for dataset in DATASETS_T1:
                hit = cells.get((model["id"], dataset["id"]))
                value = None if hit is None else hit.get(method["id"])
                row_cells[dataset["id"]] = value
                if value is not None:
                    filled += 1
            rows.append(
                {
                    "model_id": model["id"],
                    "model": model["label"],
                    "method_id": method["id"],
                    "method": method["label"],
                    "cells": row_cells,
                }
            )
    return {
        "id": "table1",
        "title": "Table 1：跨模型与任务的一般化（B=32）",
        "caption": (
            "主表格点是 模型 × 数据集 × 方法。Tok 为平均 decode/reasoning generated tokens；"
            "PC 的 Tok 下方为相对同格 SC 的减少百分比（正值=更少 token）。"
            "GSM8K 是附录连续性列，因已有完整新协议数字而写入；不是四主现代数据集。"
            "PC 数字均为 C=0.8 provisional，不是 Dev-Mix C*。"
            "Pass@1 只写准确率（SC pool 第一条轨迹）；其 Tok 未在 summary 中单独统计。"
            "Llama MATH-50、smoke、旧 Sentence-PC 不入表。"
        ),
        "datasets": DATASETS_T1,
        "models": MODELS,
        "methods": METHODS_T1,
        "rows": rows,
        "filled": filled,
        "total": total,
    }


def build_table2() -> dict[str, Any]:
    methods = [
        "SC",
        "DeepConf (*, dev-selected)",
        "CPT",
        "DTS-Stable (P1)",
        "PC",
        "DeepConf + PC",
    ]
    columns = [
        {"id": "aime_acc", "label": "AIME24+25 Acc"},
        {"id": "aime_tok", "label": "Total Tok"},
        {"id": "aime_lat", "label": "Latency"},
        {"id": "math_acc", "label": "MATH-500 Acc"},
        {"id": "math_tok", "label": "Total Tok"},
        {"id": "gpqa_acc", "label": "GPQA-D Acc"},
        {"id": "gpqa_tok", "label": "Total Tok"},
    ]
    return {
        "id": "table2",
        "title": "Table 2：现代基线与组合（Qwen3-30B-A3B，B=32）",
        "caption": (
            "计划在 Qwen3-30B-A3B 上比较 SC / DeepConf / CPT / DTS-Stable / PC / DeepConf+PC。"
            "格子只在对应完整 run 完成后填写。"
        ),
        "columns": columns,
        "rows": [{"method": name, "cells": {col["id"]: None for col in columns}} for name in methods],
        "filled": 0,
        "total": len(methods) * len(columns),
    }


def _delta_pp(parent: dict | None, plus: dict | None) -> float | None:
    if not parent or not plus or parent.get("acc") is None or plus.get("acc") is None:
        return None
    return 100.0 * (plus["acc"] - parent["acc"])


def _saving(parent: dict | None, plus: dict | None) -> float | None:
    if not parent or not plus:
        return None
    ptok = parent.get("tok")
    ctok = plus.get("tok")
    if not ptok or ctok is None:
        return None
    return 1.0 - ctok / ptok


def _delta_latency(parent: dict | None, plus: dict | None) -> float | None:
    if not parent or not plus:
        return None
    a = parent.get("latency_s")
    b = plus.get("latency_s")
    if a is None or b is None:
        return None
    return b - a


def _increment_row(
    *,
    model_id: str,
    dataset_id: str,
    model: str,
    dataset: str,
    parent_name: str,
    plus_name: str,
    parent: dict | None,
    plus: dict | None,
    planned: bool,
) -> dict[str, Any]:
    filled = parent is not None and plus is not None
    return {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "model": model,
        "dataset": dataset,
        "parent": parent_name,
        "plus": plus_name,
        "planned": planned,
        "parent_acc": None if parent is None else parent.get("acc"),
        "plus_acc": None if plus is None else plus.get("acc"),
        "delta_acc_pp": _delta_pp(parent, plus),
        "parent_tok": None if parent is None else parent.get("tok"),
        "plus_tok": None if plus is None else plus.get("tok"),
        "saving": _saving(parent, plus),
        "delta_latency_s": _delta_latency(parent, plus),
        "n": None if parent is None else parent.get("n"),
        "note": None if plus is None else plus.get("note"),
        "filled": filled,
    }


def build_table3(cells: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    extras = []
    extra_keys = [
        ("llama31", "gsm8k", "Llama-3.1-8B-Inst.", "GSM8K"),
        ("deepseek", "math500", "DeepSeek-V4-Flash", "MATH-500"),
        ("llama31", "math500", "Llama-3.1-8B-Inst.", "MATH-500"),
        ("qwen30", "math500", "Qwen3-30B-A3B", "MATH-500"),
        ("deepseek", "gsm8k", "DeepSeek-V4-Flash", "GSM8K"),
        ("qwen30", "gsm8k", "Qwen3-30B-A3B", "GSM8K"),
        ("llama31", "aime24", "Llama-3.1-8B-Inst.", "AIME 2024"),
        ("llama31", "aime25", "Llama-3.1-8B-Inst.", "AIME 2025"),
        ("llama31", "gpqa", "Llama-3.1-8B-Inst.", "GPQA-D"),
        ("qwen30", "aime24", "Qwen3-30B-A3B", "AIME 2024"),
        ("qwen30", "aime25", "Qwen3-30B-A3B", "AIME 2025"),
        ("deepseek", "aime24", "DeepSeek-V4-Flash", "AIME 2024"),
        ("deepseek", "aime25", "DeepSeek-V4-Flash", "AIME 2025"),
        ("deepseek", "gpqa", "DeepSeek-V4-Flash", "GPQA-D"),
        ("qwen30", "gpqa", "Qwen3-30B-A3B", "GPQA-D"),
    ]
    seen = set()
    for model_id, dataset_id, model, dataset in extra_keys:
        hit = cells.get((model_id, dataset_id))
        if hit is None:
            continue
        extras.append(
            _increment_row(
                model_id=model_id,
                dataset_id=dataset_id,
                model=model,
                dataset=dataset,
                parent_name="SC",
                plus_name="SC+PC",
                parent=hit.get("sc"),
                plus=hit.get("pc"),
                planned=False,
            )
        )
        seen.add((model_id, dataset_id, "SC"))

    planned_rows = []
    for spec in TABLE3_PLANNED:
        key = (spec["model_id"], spec["dataset_id"], spec["parent"])
        if key in seen:
            continue
        hit = cells.get((spec["model_id"], spec["dataset_id"]))
        parent = plus = None
        if spec["parent"] == "SC" and hit is not None:
            parent, plus = hit.get("sc"), hit.get("pc")
        planned_rows.append(
            _increment_row(
                model_id=spec["model_id"],
                dataset_id=spec["dataset_id"],
                model=spec["model"],
                dataset=spec["dataset"],
                parent_name=spec["parent"],
                plus_name=spec["plus"],
                parent=parent,
                plus=plus,
                planned=True,
            )
        )

    rows = extras + planned_rows
    filled = sum(1 for row in rows if row["filled"])
    return {
        "id": "table3",
        "title": "Table 3：parent-paired 增量",
        "caption": (
            "Latency 使用 generation_seconds（含配对根复用成本），不用 actual_generation_seconds。"
            "已完成的 SC→SC+PC 配对写在计划行之前；DeepConf parent 仍无完整实验。"
        ),
        "rows": rows,
        "filled": filled,
        "total": len(rows),
    }


def _ablation(
    aid: str,
    title: str,
    setting: str,
    columns: list[str],
    rows: list[dict[str, Any]],
    static_cols: int = 0,
) -> dict[str, Any]:
    value_cols = len(columns) - 1 - static_cols
    out_rows = []
    for row in rows:
        out_rows.append(
            {
                "label": row["label"],
                "detail": row.get("detail"),
                "cells": [None] * value_cols,
            }
        )
    return {
        "id": aid,
        "title": title,
        "caption": setting,
        "columns": columns,
        "static_cols": static_cols,
        "rows": out_rows,
        "filled": 0,
        "total": len(out_rows) * max(value_cols, 0),
    }


def build_ablations() -> list[dict[str, Any]]:
    setting = (
        "计划优先设置：Qwen3-30B-A3B ×（MATH-500 100 题 + AIME24/25 60 题）；"
        "除阈值曲线外均使用冻结的 C*。目前没有完整消融 run，格子留空。"
    )
    return [
        _ablation(
            "A",
            "Ablation A：Beta 阈值",
            setting + " 测试列只做冻结后的敏感性，不用于挑 C*。",
            ["C", "Dev Acc", "Test Acc", "Total Tok", "Gate Pass %", "Wrong Promotion %", "Rollback %"],
            [
                {"label": "0.00（纯多数）"},
                {"label": "0.70"},
                {"label": "0.80"},
                {"label": "0.90"},
                {"label": "0.95"},
                {"label": "1.00（SC）"},
            ],
        ),
        _ablation(
            "B",
            "Ablation B：前缀粒度与长度策略",
            setting,
            ["Length policy", "Level schedule", "Acc", "Total Tok", "Latency", "Avg final ratio", "Wrong Promotion %"],
            [
                {"label": "Shared Sentence-PC", "detail": "+1 sentence / level；其余组件同主方法"},
                {"label": "Original Sentence-PC", "detail": "B=20, m=5，per-branch random"},
                {"label": "Absolute-token PC", "detail": "+128 tokens / level"},
                {"label": "Relative-PC", "detail": "20→40→60→80%"},
                {"label": "Fixed-20", "detail": "始终 20%"},
                {"label": "Fixed-40", "detail": "始终 40%"},
                {"label": "PVC Adaptive", "detail": "advance / stay"},
                {"label": "PVC Adaptive (agreement only)", "detail": "无标签 advance / stay"},
            ],
            static_cols=1,
        ),
        _ablation(
            "C",
            "Ablation C：唯一前缀来源选择",
            setting,
            ["Source selector", "Acc", "Total Tok", "Diversity", "Wrong-source %", "Rollback %", "Selector overhead"],
            [
                {"label": "Random（原文）"},
                {"label": "Shortest trace"},
                {"label": "Token medoid"},
                {"label": "Highest mean logprob"},
                {"label": "BLC lower-tail confidence"},
            ],
        ),
        _ablation(
            "D",
            "Ablation D：共享方式与系统收益",
            setting,
            ["Prefix mode", "Acc", "Decode Tok", "Prefill Tok", "Latency", "Cache hit", "Distinct answers"],
            [
                {"label": "Per-branch random（原算法）"},
                {"label": "Shared random"},
                {"label": "Shared token-medoid"},
                {"label": "Shared BLC（主方法）"},
            ],
        ),
        _ablation(
            "E",
            "Ablation E：回退策略",
            setting,
            ["Rollback policy", "Acc", "Total Tok", "Failed trials", "Recovery Acc", "High-conf. wrong %"],
            [
                {"label": "No rollback / keep trial"},
                {"label": "Gate fail → reset empty（旧实现）"},
                {"label": "One-level rollback"},
                {"label": "One-level + blacklist + two-strike reset"},
                {"label": "主策略 + periodic unprefixed audit window"},
            ],
        ),
        _ablation(
            "F",
            "Ablation F：SC / DeepConf 与 PC 的 2×2 因子",
            setting + " offline 过滤诊断不代替本表。",
            ["Trace controller", "Prefix redirection", "Acc", "Decode Tok", "Total Tok", "Latency", "Completed traces"],
            [
                {"label": "SC", "detail": "off"},
                {"label": "SC", "detail": "PC"},
                {"label": "DeepConf", "detail": "off"},
                {"label": "DeepConf", "detail": "PC"},
            ],
            static_cols=1,
        ),
        _ablation(
            "G",
            "Ablation G：Prefix Value Curve",
            "干预式 PVC：独立 discovery/evaluation，拆 correct/wrong source。现有 offline 曲线不作数。",
            ["Ratio", "V_corr", "V_agree", "Correct-source V", "Wrong-source V", "Suffix Tok", "Distinct answers"],
            [
                {"label": "0%"},
                {"label": "20%"},
                {"label": "40%"},
                {"label": "60%"},
                {"label": "80%"},
            ],
        ),
    ]


def build_paper_tables(runs: list[dict]) -> dict[str, Any]:
    cells = collect_complete_cells(runs)
    table1 = build_table1(cells)
    table2 = build_table2()
    table3 = build_table3(cells)
    ablations = build_ablations()
    return {
        "notes": [
            "只写入 B=32、配对完整、覆盖计划题数的新协议实验。",
            "PC 当前数字是 C=0.8 provisional，主表不得标成 PC(C*)。",
            "Tok 口径：平均 generated reasoning/decode tokens；时延含配对根。",
        ],
        "table1": table1,
        "table2": table2,
        "table3": table3,
        "ablations": ablations,
        "complete_settings": [
            {
                "model": model_id,
                "dataset": dataset_id,
                "n": payload["n"],
                "beta": payload["beta"],
                "note": payload["note"],
            }
            for (model_id, dataset_id), payload in sorted(cells.items())
        ],
    }
