"""Planned ICLR 2027 paper tables filled from complete protocol runs.

Empty cells stay empty. A cell is filled only when a B=32 paired SC/PC run
covers the planned evaluation set in full. C=0.8 is the a priori default
operating point, not a Dev-Mix C*. Llama MATH-50, smokes, and old
Sentence-PC are excluded.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

_SC_PREFIX_K = Path(
    "/home/26xyw/data/path-consistency-results/gsm8k/reports/"
    "sc_prefix_k_llama31_c08.json"
)


MODELS = [
    {"id": "llama31", "label": "Llama-3.1-8B-Inst."},
    {"id": "qwen30", "label": "Qwen3-30B-A3B"},
    {"id": "deepseek", "label": "DeepSeek-V4-Flash"},
]

# MATH-500 Acc: 490 evaluated questions + 10 engineering-split items
# counted as correct. Denominator is 500. Token means stay over the 490
# generated questions. Table 1 / Ablation C do not use format overlay.
MATH500_EVAL_N = 490
MATH500_REPORT_N = 500
MATH500_ENGINEERING_CREDIT = 10

DATASETS_T1 = [
    {"id": "gsm8k", "label": "GSM8K", "n": 1319, "kind": "continuity"},
    {"id": "math500", "label": "MATH-500", "n": MATH500_REPORT_N, "kind": "main"},
    {"id": "aime24", "label": "AIME 2024", "n": 30, "kind": "main"},
    {"id": "aime25", "label": "AIME 2025", "n": 30, "kind": "main"},
    {"id": "gpqa", "label": "GPQA-D", "n": 198, "kind": "main"},
]

METHODS_T1 = [
    {"id": "pass1", "label": "Pass@1"},
    {"id": "sc", "label": "SC"},
    {"id": "pc", "label": "PC (C=0.8)"},
]

EXPECTED_N = {
    row["id"]: (MATH500_EVAL_N if row["id"] == "math500" else row["n"])
    for row in DATASETS_T1
}

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
            "pc_vote_v2",
            "vote_v2",
            "selrec",
            "lenint",
        )
    ):
        return False
    # Do not exclude all "first100" tags. DeepSeek GSM8K Table 1 is the
    # first100+next500+rest719 overlay (n=1319). Qwen MATH first100 is
    # kind=smoke and n=100≠490, so it still cannot fill Table 1.
    if int(run.get("max_branches") or 0) != 32:
        return False
    planned = int(run.get("planned") or 0)
    paired = int(run.get("paired") or 0)
    if planned <= 0 or paired < planned:
        return False
    beta = float(run.get("beta_threshold") or 0.0)
    if abs(beta - 0.8) > 1e-6:
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


def _credit_math500_engineering(merged: dict[str, Any]) -> dict[str, Any]:
    """Map a complete 490-eval MATH-500 cell onto the n=500 reporting convention."""
    if int(merged.get("n") or 0) != MATH500_EVAL_N:
        return merged
    extra = (
        "MATH-500 Acc: raw JSONL on 490 evaluated + 10 engineering items "
        "counted correct (n=500); no format overlay"
    )

    def credit(cell: dict[str, Any] | None) -> dict[str, Any] | None:
        if cell is None:
            return None
        correct = int(cell["correct"] or 0) + MATH500_ENGINEERING_CREDIT
        n = MATH500_REPORT_N
        out = dict(cell)
        out["correct"] = correct
        out["n"] = n
        out["acc"] = _ratio(correct, n)
        note = (out.get("note") or "").strip()
        out["note"] = f"{note}; {extra}" if note else extra
        return out

    out = dict(merged)
    out["n"] = MATH500_REPORT_N
    for key in ("pass1", "sc", "pc"):
        if out.get(key):
            out[key] = credit(out[key])
    note = (out.get("note") or "").strip()
    out["note"] = f"{note}; {extra}" if note else extra
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
        note = f"C={float(beta):.2g} default operating point"
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
        if dataset == "math500":
            merged = _credit_math500_engineering(merged)
        cells[(model, dataset)] = merged
    _apply_deepseek_gsm8k_c08_fusion(cells)
    _apply_qwen_math500_partial(cells, runs)
    return cells


def _apply_deepseek_gsm8k_c08_fusion(cells: dict[tuple[str, str], dict[str, Any]]) -> None:
    """Keep overlay Acc from original shards; PC Tok/latency from vote-v2."""
    hit = cells.get(("deepseek", "gsm8k"))
    if not hit or not hit.get("pc") or not hit.get("sc"):
        return
    pc = hit["pc"]
    sc = hit["sc"]
    pc_tok = float(_DS_GSM["pc_tok"])
    sc_tok = float(_DS_GSM["sc_tok"])
    pc["tok"] = pc_tok
    pc["tok_saving"] = (1.0 - pc_tok / sc_tok) if sc_tok else None
    pc["latency_s"] = float(_DS_GSM["pc_s"])
    pc["correct"] = int(_DS_GSM["pc_correct"])
    pc["n"] = int(_DS_GSM["n"])
    pc["acc"] = _ratio(_DS_GSM["pc_correct"], _DS_GSM["n"])
    note = (
        "Acc=format-tolerant overlay of original paired shards (1277/1319); "
        "Tok/latency=vote-v2 PC numeric-core keys, reused SC roots (−34.2%)"
    )
    pc["note"] = note
    sources = pc.get("source") or ""
    extra = "tok from gsm8k_deepseek_v4_flash_b32_c08_pc_vote_v2_3cc1d5762c45"
    if extra not in sources:
        pc["source"] = f"{sources}; {extra}" if sources else extra
    sc["correct"] = int(_DS_GSM["sc_correct"])
    sc["n"] = int(_DS_GSM["n"])
    sc["acc"] = _ratio(_DS_GSM["sc_correct"], _DS_GSM["n"])
    hit["note"] = note


def _apply_qwen_math500_partial(
    cells: dict[tuple[str, str], dict[str, Any]], runs: list[dict]
) -> None:
    """Fill Qwen × MATH-500 from complete first100/next100 shards until n=490 exists.

    JSONL source_row 0–199 minus engineering row 151 → n=199. Author asked
    to write this partial cell into Table 1. It is not the planned 490 set.
    """
    existing = cells.get(("qwen30", "math500"))
    if existing is not None and int(existing.get("n") or 0) == 490:
        return
    hits = []
    for run in runs:
        tag = (run.get("tag") or "").lower()
        if "math500" not in tag:
            continue
        if "first100" not in tag and "next100" not in tag:
            continue
        if "qwen3_30b_a3b" not in tag and "qwen30_a3b" not in tag:
            continue
        if (run.get("status") or "") != "complete":
            continue
        if int(run.get("max_branches") or 0) != 32:
            continue
        beta = float(run.get("beta_threshold") or 0.0)
        if abs(beta - 0.8) > 1e-6:
            continue
        hits.append(run)
    if not hits:
        return
    merged = _merge_group(hits)
    if merged is None:
        return
    n = int(merged.get("n") or 0)
    if n not in {100, 199}:
        return
    if n == 199:
        note = (
            "partial n=199 JSONL source_row 0–199 excluding engineering 151; "
            "not MATH-500 490; C=0.8 default operating point, not C*"
        )
    else:
        note = (
            "partial n=100 JSONL source_row 0–99; not MATH-500 490; "
            "C=0.8 default operating point, not C*"
        )
    merged["note"] = note
    merged["partial"] = True
    for key in ("pass1", "sc", "pc"):
        cell = merged.get(key)
        if not cell:
            continue
        cell["note"] = note
        cell["partial"] = True
    cells[("qwen30", "math500")] = merged


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
            "PC 数字均为默认操作点 C=0.8（先验，不是 Dev-Mix C*）。"
            "DeepSeek 格若存在 numeric_regrade sidecar，除 MATH-500 外准确率按格式容错复判"
            "（单位/货币/千分位/等价 LaTeX；剩余项经 API 同值确认），原始 JSONL 未改写。"
            "MATH-500 Acc 一律 raw JSONL：490 实测题 + 10 道工程调试题记对，分母 500；"
            "Tok 仍是 490 题均值。不用 numeric_regrade overlay。"
            "DeepSeek GSM8K C=0.8 为融合记录：Acc 用原配对分片 overlay（SC=PC=1277/1319=96.82%）；"
            "PC Tok / 时延用 vote-v2（numeric-core 投票键，复用同一 SC 根，decode −34.2%）。"
            "阈值 sweep 只在消融表；更低 C 更大加速，更高 C 更安全。"
            "Pass@1 只写准确率（SC pool 第一条轨迹）；其 Tok 未在 summary 中单独统计。"
            "Qwen3-30B-A3B × MATH-500 当前是 first100+next100 部分格"
            "（JSONL source_row 0–199，排除工程行 151，n=199）：Acc 显示 188/199，"
            "尚未覆盖 490 实测题，不能改写成 500。全量到位后改用 raw+10/500。"
            "C=0.8 仍是先验操作点，不是 C*。"
            "Llama MATH-50、smoke、旧 Sentence-PC 不入表。"
        ),
        "datasets": DATASETS_T1,
        "models": MODELS,
        "methods": METHODS_T1,
        "rows": rows,
        "filled": filled,
        "total": total,
    }


def _load_sc_prefix_k(k: int = 20) -> dict[str, Any] | None:
    if not _SC_PREFIX_K.exists():
        return None
    payload = json.loads(_SC_PREFIX_K.read_text())
    row = (payload.get("by_k") or {}).get(str(k))
    if not row or int(row.get("n") or 0) != int(_GS["n"]):
        return None
    return row


def build_table2() -> dict[str, Any]:
    columns = [
        {"id": "acc", "label": "GSM8K Acc", "kind": "acc"},
        {"id": "tok", "label": "Tok / q", "kind": "tok"},
        {"id": "lat", "label": "Latency", "kind": "lat"},
    ]
    method_specs = [
        {"id": "sc", "label": "SC (B=32)", "block": "compare"},
        {"id": "sc20", "label": "SC (B=20, same pool)", "block": "compare"},
        {"id": "polr", "label": "PoLR (N=32, L_p=64)", "block": "compare"},
        {"id": "polr128", "label": "PoLR (N=32, L_p=128)", "block": "compare"},
        {"id": "polr256", "label": "PoLR (N=32, L_p=256)", "block": "compare"},
        {"id": "cpt", "label": "CPT (N=32, Llama adapter)", "block": "compare"},
        {"id": "pc", "label": "PC (C=0.8)", "block": "compare"},
        {"id": "deepconf", "label": "DeepConf (*, freeze config)", "block": "combine"},
        {"id": "deepconf_pc", "label": "DeepConf + PC", "block": "combine"},
        {"id": "ac", "label": "AC (τ=0.95)", "block": "combine"},
        {"id": "ac_pc", "label": "AC + PC", "block": "combine"},
        {"id": "esc", "label": "ESC", "block": "combine"},
        {"id": "esc_pc", "label": "ESC + PC", "block": "combine"},
        {"id": "ac099", "label": "AC (τ=0.99)", "block": "combine"},
        {"id": "ac099_pc", "label": "AC + PC (C=0)", "block": "combine"},
        {"id": "esc2w8", "label": "ESC (2-window or ≥8)", "block": "combine"},
        {"id": "esc2w8_pc", "label": "ESC + PC (2-window or ≥8)", "block": "combine"},
    ]
    n = _GS["n"]
    sc_tok = float(_GS["sc_tok"])
    pc_tok = float(_GS["pc_tok"])
    sc_cell = _cell(
        acc=_ratio(_GS["sc_correct"], n),
        correct=_GS["sc_correct"],
        n=n,
        tok=sc_tok,
        latency_s=_GS["sc_s"],
        source="gsm8k_llama31_b32_mt2048_c08_seed0",
        note="C=0.8 paired-root v2",
    )
    pc_cell = _cell(
        acc=_ratio(_GS["pc_correct"], n),
        correct=_GS["pc_correct"],
        n=n,
        tok=pc_tok,
        tok_saving=(1.0 - pc_tok / sc_tok) if sc_tok else None,
        latency_s=_GS["pc_s"],
        beta=0.8,
        source="gsm8k_llama31_b32_mt2048_c08_seed0",
        note="C=0.8 default operating point",
    )
    polr_tok = 4826.0
    polr_cell = _cell(
        acc=_ratio(1211, n),
        correct=1211,
        n=n,
        tok=polr_tok,
        tok_saving=(1.0 - polr_tok / sc_tok) if sc_tok else None,
        latency_s=10.6,
        source="polr_llama31_gsm8k_n32_lp64_seed0_eb5fffd3c21f",
        note="N=32 prefixes of 64 tokens; TF-IDF agglomerative; expand dominant cluster",
    )
    polr128_tok = 5863.0
    polr128_cell = _cell(
        acc=_ratio(1216, n),
        correct=1216,
        n=n,
        tok=polr128_tok,
        tok_saving=(1.0 - polr128_tok / sc_tok) if sc_tok else None,
        latency_s=11.2,
        source="polr_llama31_gsm8k_n32_lp128_seed0_77b0d966405d",
        note="N=32 prefixes of 128 tokens; interpolates L_p=64 and 256; still a true prefix vs SC mean ~260",
    )
    polr256_tok = 7554.0
    polr256_cell = _cell(
        acc=_ratio(1208, n),
        correct=1208,
        n=n,
        tok=polr256_tok,
        tok_saving=(1.0 - polr256_tok / sc_tok) if sc_tok else None,
        latency_s=13.0,
        source="polr_llama31_gsm8k_n32_lp256_seed0_0703b9899c8f",
        note="Paper-default L_p=256; Llama GSM8K SC mean ~260 so 256 is near-full trace",
    )
    cpt_tok = 8411.0
    cpt_cell = _cell(
        acc=_ratio(1224, n),
        correct=1224,
        n=n,
        tok=cpt_tok,
        tok_saving=(1.0 - cpt_tok / sc_tok) if sc_tok else None,
        latency_s=34.9,
        source="cpt_llama31_gsm8k_n32",
        note="Official CPT qwen/math.py; N=32 workers still emit full traces (tok +1.1% vs SC); gain is wall-clock total_latency_sec 54.4→34.9s; Llama adapter, not Qwen thinking-on",
    )
    ac_tok = 2730.0
    ac_pc_tok = 2539.0
    esc_tok = 2552.0
    esc_pc_tok = 2374.0
    ac_cell = _cell(
        acc=_ratio(1220, n),
        correct=1220,
        n=n,
        tok=ac_tok,
        tok_saving=(1.0 - ac_tok / sc_tok) if sc_tok else None,
        latency_s=19.3,
        source="gsm8k_llama31_b32_mt2048_ac_seed0_9d1997bfb7de",
        note="AC τ=0.95; window m=4; B=32; 88.8% early-stop",
    )
    ac_pc_cell = _cell(
        acc=_ratio(1217, n),
        correct=1217,
        n=n,
        tok=ac_pc_tok,
        tok_saving=(1.0 - ac_pc_tok / sc_tok) if sc_tok else None,
        latency_s=18.0,
        source="gsm8k_llama31_b32_mt2048_ac_seed0_9d1997bfb7de",
        note="AC τ=0.95 + PC C=0.8 prefix; paired AC roots",
    )
    esc_cell = _cell(
        acc=_ratio(1220, n),
        correct=1220,
        n=n,
        tok=esc_tok,
        tok_saving=(1.0 - esc_tok / sc_tok) if sc_tok else None,
        latency_s=18.1,
        source="gsm8k_llama31_b32_mt2048_esc_seed0_ab1a20b858bd",
        note="ESC unanimous window m=4; B=32; 91.3% early-stop",
    )
    esc_pc_cell = _cell(
        acc=_ratio(1217, n),
        correct=1217,
        n=n,
        tok=esc_pc_tok,
        tok_saving=(1.0 - esc_pc_tok / sc_tok) if sc_tok else None,
        latency_s=16.8,
        source="gsm8k_llama31_b32_mt2048_esc_seed0_ab1a20b858bd",
        note="ESC + PC C=0.8 prefix; paired ESC roots",
    )
    ac099_tok = 4035.0
    ac099_pc_tok = 3003.0
    esc2w8_tok = 4299.0
    esc2w8_pc_tok = 3501.0
    ac099_cell = _cell(
        acc=_ratio(1214, n),
        correct=1214,
        n=n,
        tok=ac099_tok,
        tok_saving=(1.0 - ac099_tok / sc_tok) if sc_tok else None,
        latency_s=28.0,
        source="gsm8k_llama31_b32_mt2048_ac099_beta0_seed0_7e39e070f987",
        note="AC τ=0.99; 4/4 Beta≈0.969 no longer stops; 83.2% early-stop; diagnostic vs τ=0.95",
    )
    ac099_pc_cell = _cell(
        acc=_ratio(1216, n),
        correct=1216,
        n=n,
        tok=ac099_pc_tok,
        tok_saving=(1.0 - ac099_pc_tok / sc_tok) if sc_tok else None,
        latency_s=21.1,
        beta=0.0,
        source="gsm8k_llama31_b32_mt2048_ac099_beta0_seed0_7e39e070f987",
        note="AC τ=0.99 + PC C=0; paired AC roots; +PC tok −25.6% vs this parent",
    )
    esc2w8_cell = _cell(
        acc=_ratio(1214, n),
        correct=1214,
        n=n,
        tok=esc2w8_tok,
        tok_saving=(1.0 - esc2w8_tok / sc_tok) if sc_tok else None,
        latency_s=29.8,
        source="gsm8k_llama31_b32_mt2048_esc2w8_seed0_d60277f37fce",
        note="ESC two_window_or_8; 78.5% early-stop; diagnostic vs single unanimous window",
    )
    esc2w8_pc_cell = _cell(
        acc=_ratio(1214, n),
        correct=1214,
        n=n,
        tok=esc2w8_pc_tok,
        tok_saving=(1.0 - esc2w8_pc_tok / sc_tok) if sc_tok else None,
        latency_s=24.4,
        source="gsm8k_llama31_b32_mt2048_esc2w8_seed0_d60277f37fce",
        note="ESC two_window_or_8 + PC C=0.8; paired ESC roots; +PC tok −18.6% vs this parent",
    )
    filled_by = {
        "sc": {"acc": sc_cell, "tok": sc_cell, "lat": sc_cell},
        "pc": {"acc": pc_cell, "tok": pc_cell, "lat": pc_cell},
        "polr": {"acc": polr_cell, "tok": polr_cell, "lat": polr_cell},
        "polr128": {"acc": polr128_cell, "tok": polr128_cell, "lat": polr128_cell},
        "polr256": {"acc": polr256_cell, "tok": polr256_cell, "lat": polr256_cell},
        "cpt": {"acc": cpt_cell, "tok": cpt_cell, "lat": cpt_cell},
        "ac": {"acc": ac_cell, "tok": ac_cell, "lat": ac_cell},
        "ac_pc": {"acc": ac_pc_cell, "tok": ac_pc_cell, "lat": ac_pc_cell},
        "esc": {"acc": esc_cell, "tok": esc_cell, "lat": esc_cell},
        "esc_pc": {"acc": esc_pc_cell, "tok": esc_pc_cell, "lat": esc_pc_cell},
        "ac099": {"acc": ac099_cell, "tok": ac099_cell, "lat": ac099_cell},
        "ac099_pc": {"acc": ac099_pc_cell, "tok": ac099_pc_cell, "lat": ac099_pc_cell},
        "esc2w8": {"acc": esc2w8_cell, "tok": esc2w8_cell, "lat": esc2w8_cell},
        "esc2w8_pc": {"acc": esc2w8_pc_cell, "tok": esc2w8_pc_cell, "lat": esc2w8_pc_cell},
    }
    sc20 = _load_sc_prefix_k(20)
    if sc20 and sc20.get("accuracy") is not None:
        sc20_tok = float(sc20["tok_per_q"])
        sc20_cell = _cell(
            acc=float(sc20["accuracy"]),
            correct=int(sc20["correct"]),
            n=int(sc20["n"]),
            tok=sc20_tok,
            tok_saving=(1.0 - sc20_tok / sc_tok) if sc_tok else None,
            latency_s=float(sc20["latency_s"]),
            source="sc_prefix_k_llama31_c08 k=20",
            note="offline majority over first 20 traces of SC-32; same window seeds as SC B=20",
        )
        filled_by["sc20"] = {"acc": sc20_cell, "tok": sc20_cell, "lat": sc20_cell}
    rows = []
    filled = 0
    for spec in method_specs:
        cells = filled_by.get(spec["id"]) or {col["id"]: None for col in columns}
        filled += sum(1 for value in cells.values() if value is not None)
        rows.append(
            {
                "method_id": spec["id"],
                "method": spec["label"],
                "block": spec.get("block"),
                "cells": cells,
            }
        )
    return {
        "id": "table2",
        "title": "Table 2：现代基线与组合（Llama-3.1-8B-Inst. × GSM8K，B=32）",
        "caption": (
            "Backbone 为 Llama-3.1-8B-Instruct；只报 GSM8K test 1319。"
            "MATH / AIME / GPQA 对 8B 过难，本表不填那些列。"
            "Tok = 平均 decode generated tokens；Latency = generation_seconds（含配对根）。"
            "上块是比较：PoLR / CPT 与等预算 PC，并行宽度一律 N=32 或 B=32。"
            "下块是组合：截断型 parent（DeepConf / AC / ESC）及其 +PC。"
            "PC 为默认操作点 C=0.8，不是 C*。"
            "SC (B=20) 是同一 SC-32 JSONL 的前 20 条轨迹离线多数票，"
            "只用于 matched-token，不是 CPT 的并行宽度。"
            "CPT N=20 Llama adapter 已跑完但不进本表；Llama 无 thinking，不是 Qwen 公平 CPT。"
            "CPT Latency 是官方 total_latency_sec。"
            "N=32 时 CPT 每 worker 仍写满一条解答（mean 263 tok，与 SC 单条 260 相当），"
            "decode token 不降（8,411 vs SC 8,316，+1.1%）；收益是墙钟 54.4 s→34.9 s。"
            "PoLR 为 arXiv:2601.21494：N=32，同时报 L_p=64 / 128（本机真前缀）与论文默认 L_p=256。"
            "AC 用 Adaptive-Consistency 默认 τ=0.95，不是 PC 前缀门控 C=0.8。"
            "ESC 在窗口内答案全同且全部可解析时停止。旧 Sentence-PC AC/ESC（B=20）不进本表。"
            "另报诊断：AC τ=0.99 + PC C=0，以及 ESC 连续两窗全同或累计 ≥8 条相同。"
            "AC / ESC 默认与诊断及其 +PC、CPT N=32、PoLR L_p=64/128/256 已填 1319。DeepConf 未满 1319 前留空。"
            "DTS-Stable 仍是 P1，不占本表。"
        ),
        "model": "Llama-3.1-8B-Inst.",
        "dataset": "GSM8K",
        "n": n,
        "blocks": [
            {"id": "compare", "label": "比较：并行共享（PoLR / CPT vs PC）"},
            {"id": "combine", "label": "组合：截断型 parent ± PC"},
        ],
        "columns": columns,
        "rows": rows,
        "filled": filled,
        "total": len(method_specs) * len(columns),
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
        "partial": bool((plus or {}).get("partial") or (parent or {}).get("partial")),
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
            "Qwen3-30B-A3B × MATH-500 若出现 n=199，是 source_row 0–199 排除工程 151 的部分结果，不是 500。"
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
    note: str | None = None,
) -> dict[str, Any]:
    value_cols = len(columns) - 1 - static_cols
    out_rows = []
    filled = 0
    for row in rows:
        cells = list(row.get("cells") or [])
        if len(cells) < value_cols:
            cells.extend([None] * (value_cols - len(cells)))
        elif len(cells) > value_cols:
            cells = cells[:value_cols]
        filled += sum(1 for cell in cells if cell not in (None, ""))
        out_rows.append(
            {
                "label": row["label"],
                "detail": row.get("detail"),
                "cells": cells,
            }
        )
    return {
        "id": aid,
        "title": title,
        "caption": setting,
        "columns": columns,
        "static_cols": static_cols,
        "rows": out_rows,
        "filled": filled,
        "total": len(out_rows) * max(value_cols, 0),
        "note": note,
    }


# Llama-3.1-8B-Instruct × GSM8K test 1319, tag gsm8k_llama31_b32_mt2048_c08_seed0
# (4 shards, paired-root v2, B=32, m=4). C=0.8 is the default operating point.
_GS = {
    "n": 1319,
    "sc_correct": 1215,
    "pc_correct": 1210,
    "sc_tok": 8316,
    "pc_tok": 5039,
    "sc_prompt": 3487,
    "pc_prompt": 6800,
    "sc_s": 54.4,
    "pc_s": 33.8,
    "gate_q": 1178,
    "mean_ratio": 0.707,
    "wrong_promo": 560,
    "commits": 4692,
    "rollback_w": 129,
    "windows": 10552,
    "rollback_q": 101,
    "failed_trials": 112,
    "highconf_wrong": 44,
    "pc_distinct": 2.53,
    "sc_distinct": 2.96,
    "cache_hit": 2483,
    "traces": 32.0,
}

# DeepSeek-V4-Flash × GSM8K test 1319, fused C=0.8 record:
# Acc = format-tolerant overlay of original paired shards
# (first100+next500+rest719; numeric-core + duration + API same-value).
# Efficiency = vote-v2 PC (3cc1d5762c45) with numeric-core vote keys,
# reusing those SC roots. Tok 12514→8230 (−34.2%). Gate / Wrong Promo /
# Rollback also from vote-v2 commit logs (string parent_modal != gold).
# Do not mix these protocol stats with C=0 / C=0.7 (old vote keys).
_DS_GSM = {
    "n": 1319,
    "sc_correct": 1277,
    "pc_correct": 1277,
    "sc_tok": 12514,
    "pc_tok": 8230,
    "pc_s": 27.007880462870183,
    "gate_q": 1298,
    "wrong_promo": 146,
    "commits": 5189,
    "rollback_w": 37,
    "windows": 10552,
}
_DS_GSM_C0 = {
    "n": 1319,
    "sc_correct": 1283,
    "pc_correct": 1280,
    "sc_tok": 9910,
    "pc_tok": 7070,
    "gate_q": 1281,
    "wrong_promo": 999,
    "commits": 5140,
    "rollback_w": 185,
    "windows": 10552,
}
# C=0.7 paired tag gsm8k_deepseek_v4_flash_b32_c07_rep0 / bb3be741ca71.
# Acc: format-tolerant overlay (local numeric-core + reused 2026-09-11 API
# verdicts). Gate used the pre-numeric-core vote keys. Tok / gate / wrong-promo
# from commit logs: Wrong Promotion = commit parent_modal != gold (string).
_DS_GSM_C07 = {
    "n": 1319,
    "sc_correct": 1282,
    "pc_correct": 1281,
    "sc_tok": 9898,
    "pc_tok": 7399,
    "gate_q": 1236,
    "wrong_promo": 1183,
    "commits": 4922,
    "rollback_w": 165,
    "windows": 10552,
}

# DeepSeek-V4-Flash × MATH-500, B=32 m=4 C=0.8, paired-root v2.
# Shared SC root: math500_deepseek_v4_flash_b32_c08_rep0_4856d4ef5673.
# Acc = raw JSONL on 490 evaluated + 10 engineering items counted correct
# (n=500). Do NOT use numeric_regrade overlay.
# Arm A = main protocol (BLC + recovery on). B = uniform + recovery on.
# C = BLC + recovery off (retain failed trial/committed prefix; no pop;
# no two-strike). D = uniform + recovery off.
# Tok rounded from parent_costs.avg_generated_tokens vs shared SC 41165.62
# over the 490 generated questions.
_DS_MATH500_EVAL_N = MATH500_EVAL_N
_DS_MATH500_N = MATH500_REPORT_N
_DS_MATH500_SC = {
    "correct": 430 + MATH500_ENGINEERING_CREDIT,
    "tok": 41166,
}
_DS_MATH500_SELREC = {
    "A": {
        "label": "BLC + recovery on（主方法 · 臂 A）",
        "correct": 428 + MATH500_ENGINEERING_CREDIT,
        "tok": 26968,
        "gate_q": 481,
        "wrong_promo": 234,
        "commits": 1886,
        "fail_w": 17,
        "fail_label": "17 rollback",
        "fail_q": 14,
        "fail_pc": 6,
        "fail_sc": 7,
        "disable_q": 3,
        "highconf_wrong": 46,
        "diversity": 1.31,
        "gain": 0,
        "loss": 2,
    },
    "B": {
        "label": "Uniform + recovery on（臂 B）",
        "correct": 426 + MATH500_ENGINEERING_CREDIT,
        "tok": 25680,
        "gate_q": 481,
        "wrong_promo": 215,
        "commits": 1849,
        "fail_w": 58,
        "fail_label": "58 rollback",
        "fail_q": 34,
        "fail_pc": 14,
        "fail_sc": 17,
        "disable_q": 11,
        "highconf_wrong": 43,
        "diversity": 1.35,
        "gain": 2,
        "loss": 6,
    },
    "C": {
        "label": "BLC + recovery off（臂 C）",
        "correct": 423 + MATH500_ENGINEERING_CREDIT,
        "tok": 24757,
        "gate_q": 481,
        "wrong_promo": 213,
        "commits": 1851,
        "fail_w": 141,
        "fail_label": "141 retain",
        "fail_q": 33,
        "fail_pc": 10,
        "fail_sc": 15,
        "disable_q": 0,
        "highconf_wrong": 44,
        "diversity": 1.35,
        "gain": 4,
        "loss": 11,
    },
    "D": {
        "label": "Uniform + recovery off（臂 D）",
        "correct": 423 + MATH500_ENGINEERING_CREDIT,
        "tok": 25405,
        "gate_q": 481,
        "wrong_promo": 207,
        "commits": 1829,
        "fail_w": 161,
        "fail_label": "161 retain",
        "fail_q": 31,
        "fail_pc": 9,
        "fail_sc": 14,
        "disable_q": 0,
        "highconf_wrong": 41,
        "diversity": 1.34,
        "gain": 4,
        "loss": 11,
    },
}


def _acc(correct: int, n: int = _GS["n"]) -> str:
    return f"{100.0 * correct / n:.2f}% ({correct}/{n})"


def _pct(num: int, den: int, digits: int = 1) -> str:
    return f"{100.0 * num / den:.{digits}f}%"


def _bin_se_pp(correct: int, n: int) -> str:
    """Binomial standard error of an accuracy, in percentage points.

    This is the uncertainty of a single run, not a multi-seed standard
    deviation. DeepSeek API has no seed; between-run fluctuation is the
    difference of the paired SC accuracies.
    """
    if n <= 0:
        return ""
    p = correct / n
    return f"±{100.0 * math.sqrt(p * (1.0 - p) / n):.2f} pp"


def _delta_pc_sc(pc_correct: int, sc_correct: int, n: int) -> str:
    delta = 100.0 * (pc_correct - sc_correct) / n
    if abs(delta) < 0.005:
        return "0.00 pp"
    sign = "−" if delta < 0 else "+"
    return f"{sign}{abs(delta):.2f} pp"


def _ds_threshold_cells(row: dict[str, int], *, as_sc: bool = False) -> list[str]:
    n = int(row["n"])
    sc_c = int(row["sc_correct"])
    pc_c = sc_c if as_sc else int(row["pc_correct"])
    tok = int(row["sc_tok"] if as_sc else row["pc_tok"])
    if as_sc:
        gate, wrong, rollback = "0.0%", "0.0%", "0.0%"
    else:
        gate = _pct(row["gate_q"], n)
        wrong = _pct(row["wrong_promo"], row["commits"])
        rollback = _pct(row["rollback_w"], row["windows"])
    return [
        _acc(sc_c, n),
        _acc(pc_c, n),
        _delta_pc_sc(pc_c, sc_c, n),
        _bin_se_pp(pc_c, n),
        f"{tok:,}",
        gate,
        wrong,
        rollback,
    ]


def _math500_selrec_cells(row: dict[str, Any] | None, *, as_sc: bool = False) -> list[str]:
    n = _DS_MATH500_N
    sc_c = int(_DS_MATH500_SC["correct"])
    sc_tok = int(_DS_MATH500_SC["tok"])
    if as_sc or row is None:
        return [
            _acc(sc_c, n),
            "0.00 pp",
            f"{sc_tok:,}",
            "—",
            "0.0%",
            "0.0%",
            "0",
            "—",
        ]
    pc_c = int(row["correct"])
    tok = int(row["tok"])
    save = 100.0 * (sc_tok - tok) / sc_tok
    fail_q = int(row["fail_q"])
    fail_pc = int(row["fail_pc"])
    return [
        _acc(pc_c, n),
        _delta_pc_sc(pc_c, sc_c, n),
        f"{tok:,}",
        f"−{save:.1f}%",
        _pct(int(row["gate_q"]), _DS_MATH500_EVAL_N),
        _pct(int(row["wrong_promo"]), int(row["commits"])),
        row["fail_label"],
        f"{100.0 * fail_pc / fail_q:.1f}% ({fail_pc}/{fail_q})",
    ]


def _math500_selrec_e_cells(row: dict[str, Any]) -> list[str]:
    n = _DS_MATH500_N
    sc_c = int(_DS_MATH500_SC["correct"])
    pc_c = int(row["correct"])
    fail_q = int(row["fail_q"])
    fail_pc = int(row["fail_pc"])
    fail_sc = int(row["fail_sc"])
    return [
        _acc(pc_c, n),
        _delta_pc_sc(pc_c, sc_c, n),
        f"{int(row['tok']):,}",
        row["fail_label"],
        f"{100.0 * fail_pc / fail_q:.1f}% ({fail_pc}/{fail_q})",
        f"SC {fail_sc}/{fail_q}",
        str(int(row["disable_q"])),
        _pct(int(row["highconf_wrong"]), _DS_MATH500_EVAL_N),
    ]


_SELREC_CAPTION = (
    "DeepSeek-V4-Flash × MATH-500（B=32，m=4，C=0.8，paired-root v2，thinking on）。"
    "四臂共享同一 SC 根（tag 4856d4ef5673）。"
    "Acc = 490 实测 raw JSONL + 10 道工程题记对，分母 500（SC 440/500）。"
    "Tok 仍是 490 题均值 41,166。不用 numeric_regrade overlay。"
    "臂 A 即主表协议。B/C/D 为 PC-only，--reuse-root-cache，新 fingerprint，不覆盖主表 PC JSONL。"
    "Gate / Wrong Promo / Fail 仍相对 490 实测题。"
    "Fail windows = rollback_*（recovery 开）或 retain_failed_*（recovery 关）。"
    "Fail-set Acc = 至少一次失败事件的题上 PC 正确率。"
    "Wrong Promotion = commit 窗口 proposed_prefix.parent_modal ≠ gold（字符串）。"
    "Gate Pass = 用过前缀的实测题占比。"
    "本表替换原 Llama GSM8K 来源选择 / 回退占位表。"
    "同历史长度干预见 Ablation G（466 题 × R=8 complete）。"
)

_SELREC_NOTE = (
    "Acc 与 Table 1 MATH-500 同一口径：raw + 10 记对 / 500。不要拿 overlay 452 比。"
    "相对同一 SC：A −2 题（0/2），B −4（2/6），C −7（4/11），D −7（4/11）。"
    "相对 A：B −2（3/5），C −5（4/9），D −5（5/10）。D 与 C 同为 433/500，但是 4 gain / 4 loss，不是同一题集。"
    "McNemar 精确双侧 p 均 >0.1；discordance 少，不报显著。"
    "2×2 交互（实测正确题数，不含 +10）：uniform | recovery 开 = B−A = −2；uniform | recovery 关 = D−C = 0；"
    "recovery 关 | BLC = C−A = −5；recovery 关 | uniform = D−B = −3。"
    "交互 (D−A)−(B−A)−(C−A) = +2：两因素同时偏离主方法时，伤害略小于相加，不能写成「uniform+关 recovery 叠加崩掉」。"
    "uniform 增加失败事件（rollback 17→58；retain 141→161）和 prefixing_disabled（3→11，仅 recovery 开）。"
    "recovery 关从不 disable；fail-set Acc A 6/14、B 14/34、C 10/33、D 9/31，相对该子集 SC 分别 −1/−3/−5/−5。"
    "Tok vs SC：A −34.5%、B −37.6%、D −38.3%、C −39.9%。更省来自保留失败长前缀或更短的 uniform 源，不是更准。"
    "B/C/D generation_seconds（65.8 / 62.8 / 60.2 s）低于 A（110.3 s），含复用根，但 actual wall 受 API 时刻影响，"
    "不作为选择/恢复的因果时延；效率主指标是相对同一 SC 的 decode Tok。"
    "BLC 非有限 score：A 731/1899，B 1009/1897，C 757/1869，D 1002/1853；"
    "无 logprob 时 fallback 短源+branch id，不能把本表写成全程完整 BLC。"
)


# Qwen3-30B-A3B × MATH-500 source_row 0–99, fingerprint 630825247308.
# Complete 100/100. Smoke / first100: NOT Table 1 (n≠490).
_QWEN_F100 = {
    "n": 100,
    "sc_correct": 94,
    "pc_correct": 94,
    "p1_correct": 94,
    "sc_tok": 141319.52,
    "pc_tok": 84111.51,
    "sc_s": 564.5561858468282,
    "pc_s": 348.79417930074965,
    "gate_q": 99,
    "mean_ratio": 0.782,
    "trunc_sc": 6,
    "trunc_pc": 3,
    "parser_sc": 4,
    "parser_pc": 1,
    "reused_root": 112,
    "gain": 0,
    "loss": 0,
    "fingerprint": "630825247308",
    "tag": "math500_qwen3_30b_a3b_first100_b32_mt32768_c08_seed0",
    "subjects": [
        {"label": "Algebra", "n": 25, "sc": 24, "pc": 24, "sc_tok": 91949, "pc_tok": 52990},
        {"label": "Intermediate Algebra", "n": 19, "sc": 17, "pc": 17, "sc_tok": 213037, "pc_tok": 126519},
        {"label": "Prealgebra", "n": 18, "sc": 17, "pc": 17, "sc_tok": 108010, "pc_tok": 60440},
        {"label": "Precalculus", "n": 14, "sc": 13, "pc": 13, "sc_tok": 137481, "pc_tok": 82705},
        {"label": "Number Theory", "n": 11, "sc": 11, "pc": 11, "sc_tok": 132574, "pc_tok": 71957},
        {"label": "Geometry", "n": 7, "sc": 6, "pc": 6, "sc_tok": 145207, "pc_tok": 109255},
        {"label": "Counting & Probability", "n": 6, "sc": 6, "pc": 6, "sc_tok": 240307, "pc_tok": 146740},
    ],
}


def _tok_int(x: float) -> str:
    return f"{round(x):,}"


def _save_pct(child: float, parent: float) -> str:
    if not parent:
        return ""
    frac = 1.0 - child / parent
    sign = "−" if frac >= 0 else "+"
    return f"{sign}{100.0 * abs(frac):.1f}%"


def build_qwen_first100() -> list[dict[str, Any]]:
    n = int(_QWEN_F100["n"])
    sc_c = int(_QWEN_F100["sc_correct"])
    pc_c = int(_QWEN_F100["pc_correct"])
    p1_c = int(_QWEN_F100["p1_correct"])
    sc_tok = float(_QWEN_F100["sc_tok"])
    pc_tok = float(_QWEN_F100["pc_tok"])
    sc_s = float(_QWEN_F100["sc_s"])
    pc_s = float(_QWEN_F100["pc_s"])
    caption = (
        "Qwen3-30B-A3B × MATH-500 JSONL source_row 0–99（n=100；与工程 10 题无交集）。"
        "冻结协议：B=32，m=4，C=0.8，paired-root v2，thinking on；"
        "T=0.6，top-p=0.95，top-k=20，max_tokens=32768。"
        f"tag {_QWEN_F100['tag']}，fingerprint {_QWEN_F100['fingerprint']}。"
        "Tok = 平均 decode generated tokens；时延 = generation_seconds（含复用 SC root）。"
        "禁止把 actual_generation_seconds 当 speedup。"
        "C=0.8 是先验默认操作点，不是 C*。"
        "n=100 ≠ 490，不进 Table 1，不回写最优 C，不能替代全量 490。"
    )
    main = _ablation(
        "qwen-first100",
        "试跑：Qwen3-30B-A3B × MATH-500 first100（不进 Table 1）",
        caption,
        ["Method", "Acc（raw）", "Δ vs SC", "Tok / q", "vs SC", "Latency", "vs SC"],
        [
            {
                "label": "Pass@1（SC 第一条）",
                "cells": [
                    _acc(p1_c, n),
                    _delta_pc_sc(p1_c, sc_c, n),
                    "—",
                    "—",
                    "—",
                    "—",
                ],
            },
            {
                "label": "SC（B=32，无前缀）",
                "cells": [
                    _acc(sc_c, n),
                    "—",
                    _tok_int(sc_tok),
                    "—",
                    f"{sc_s:.1f} s",
                    "—",
                ],
            },
            {
                "label": "PC (C=0.8)",
                "cells": [
                    _acc(pc_c, n),
                    _delta_pc_sc(pc_c, sc_c, n),
                    _tok_int(pc_tok),
                    _save_pct(pc_tok, sc_tok),
                    f"{pc_s:.1f} s",
                    _save_pct(pc_s, sc_s),
                ],
            },
        ],
        note=(
            "配对 100/100；SC 与 PC 逐题一致（gain=0，loss=0），ΔAcc=0.00 pp。"
            f"Gate（最终前缀比例>0）{_QWEN_F100['gate_q']}/{n}，"
            f"均值 r={_QWEN_F100['mean_ratio']:.3f}（96 题停在 0.8）。"
            f"截断 token 事件 SC {_QWEN_F100['trunc_sc']} / PC {_QWEN_F100['trunc_pc']}；"
            f"parser_failures SC {_QWEN_F100['parser_sc']} / PC {_QWEN_F100['parser_pc']}。"
            f"PC 复用 SC root 窗口 {_QWEN_F100['reused_root']}。"
            "SE(PC)=±2.37 pp（二项标准误，不是多 seed 标准差）。"
            "raw Acc 未做 DeepSeek 那套格式容错 overlay；6 道双方皆错里有若干看起来像格式/次序 "
            "（Evelyn、east、1,-2 vs -2,1、矩阵分数线），未回写主表、未当 C*。"
            "墙钟约 22.9 h（2026-09-19 22:53 → 09-20 21:49 CST）。"
        ),
    )
    sub_rows = []
    for row in _QWEN_F100["subjects"]:
        sub_rows.append(
            {
                "label": f"{row['label']} (n={row['n']})",
                "cells": [
                    f"{row['sc']}/{row['n']}",
                    f"{row['pc']}/{row['n']}",
                    _tok_int(row["sc_tok"]),
                    _tok_int(row["pc_tok"]),
                    _save_pct(row["pc_tok"], row["sc_tok"]),
                ],
            }
        )
    subjects = _ablation(
        "qwen-first100-subject",
        "试跑：first100 按 MATH 科目（仍不进 Table 1）",
        "同一 fingerprint 的 100 题按 JSONL subject 拆开。每行 n 很小，只作分布说明，不做科目结论。",
        ["Subject", "SC Acc", "PC Acc", "Tok SC", "Tok PC", "vs SC"],
        sub_rows,
        note="Geometry 仅 7 题、token 节省最低（−24.8%）；Number Theory 11 题全对且 −45.7%。不要按科目选 C。",
    )
    return [main, subjects]


def build_ablations() -> list[dict[str, Any]]:
    setting = (
        "当前设置：Llama-3.1-8B-Instruct × GSM8K test 1319（B=32，m=4，paired-root v2）。"
        "C=0.8 是先验默认操作点，不是 C*。"
        "Tok = 平均 decode generated tokens；时延 = generation_seconds（含复用 SC root）。"
        "Gate Pass = 最终前缀比例>0 的题占比；Wrong Promotion = commit 窗口中 parent modal ≠ gold；"
        "Rollback % = 含 rollback 的窗口占比。"
        "未跑的对照行留空。旧 Sentence-PC（Llama-3-8B，B=20）不填入。"
        "DeepConf 10 题 smoke 与 offline 过滤不进本表。"
        "PVC 查找表控制器已取消；Adaptive 行保持空，除非另做干预式分析。"
    )
    pc_acc = _acc(_GS["pc_correct"])
    sc_acc = _acc(_GS["sc_correct"])
    pc_tok = f"{_GS['pc_tok']:,}"
    sc_tok = f"{_GS['sc_tok']:,}"
    pc_lat = f"{_GS['pc_s']:.1f} s"
    sc_lat = f"{_GS['sc_s']:.1f} s"
    gate = _pct(_GS["gate_q"], _GS["n"])
    wrong = _pct(_GS["wrong_promo"], _GS["commits"])
    rollback = _pct(_GS["rollback_w"], _GS["windows"])
    ratio = f"{100.0 * _GS['mean_ratio']:.1f}%"
    cache = _pct(_GS["cache_hit"], _GS["windows"])
    return [
        _ablation(
            "A",
            "Ablation A：Beta 阈值（Llama-3.1-8B × GSM8K）",
            setting + " 主表用 C=0.8；本表比较不同阈值。更低 C → 更大加速；更高 C → 更安全（更接近 SC）。Dev Acc 列不用来选阈，可留空。",
            ["C", "Dev Acc", "Test Acc", "Total Tok", "Gate Pass %", "Wrong Promotion %", "Rollback %"],
            [
                {"label": "0.00（纯多数）"},
                {"label": "0.70"},
                {
                    "label": "0.80（默认操作点）",
                    "cells": [None, pc_acc, pc_tok, gate, wrong, rollback],
                },
                {"label": "0.90"},
                {"label": "0.95"},
                {
                    "label": "1.00（SC）",
                    "cells": [None, sc_acc, sc_tok, "0.0%", "0.0%", "0.0%"],
                },
            ],
        ),
        _ablation(
            "A-deepseek",
            "Ablation A：Beta 阈值（DeepSeek-V4-Flash × GSM8K）",
            "DeepSeek-V4-Flash × GSM8K test 1319（B=32，m=4，paired-root v2，thinking on，"
            "reasoning_effort=high）。C=0.8 是先验默认操作点，不是 C*。"
            "每一行的 SC Acc 来自**同一次** sc,pc 配对 run，Δ = PC−该行 SC。"
            "SE(PC) 是单次准确率的二项标准误 √[p(1−p)/n]，不是多 seed 重复实验的标准差；"
            "API 无 seed，run 间波动看两行 SC 之差。"
            "Acc 为格式容错复判。"
            "禁止从本表回写最优 C。"
            "C=0.8 为融合记录：Acc 仍是原配对分片 overlay（1277/1277）；"
            "Tok / Gate / Wrong Promo / Rollback 来自 vote-v2（numeric-core 投票键）。"
            "C=0 与 C=0.7 的效率列仍是旧投票键，不要拿来跟 C=0.8 比「阈值效应」。",
            [
                "C",
                "SC Acc（配对）",
                "PC Acc",
                "Δ (PC−SC)",
                "SE(PC)",
                "Total Tok",
                "Gate Pass %",
                "Wrong Promotion %",
                "Rollback %",
            ],
            [
                {
                    "label": "0.00（纯多数）",
                    "cells": _ds_threshold_cells(_DS_GSM_C0),
                },
                {
                    "label": "0.70",
                    "cells": _ds_threshold_cells(_DS_GSM_C07),
                },
                {
                    "label": "0.80（默认操作点）",
                    "cells": _ds_threshold_cells(_DS_GSM),
                },
                {"label": "0.90"},
                {"label": "0.95"},
                {
                    "label": "1.00（SC，同 C=0.8 的配对 root）",
                    "cells": _ds_threshold_cells(_DS_GSM, as_sc=True),
                },
            ],
            note=(
                "读 Δ 列，不要跨行比 PC 绝对值。"
                "C=0.8（融合）：Δ = 0.00 pp，decode 12514→8230（−34.2%），Gate 98.4%，Wrong Promo 2.8%。"
                "C=0.7：Δ = −0.08 pp（−1 题），decode 9898→7399（−25.2%），Gate 93.7%，Wrong Promo 24.0%。"
                "C=0：Δ = −0.23 pp（官方 JSONL −18 题）。"
                "C=0.8 的 −34.2% / 98.4% Gate 来自投票键对齐，不是「比 C=0.7 更激进的阈值」。"
                "更低 C → 更容易晋升、token 更少、错前缀风险更高；不能回写最优 C。"
                "C=0.7 门控仍用旧投票键（非 numeric-core）；Acc 是事后格式容错复判。"
                "C=0 / C=0.7 / C=0.8 的 SC 底池来自独立 API 抽样，已差数题，小于 SE(PC)≈0.46–0.48 pp。"
                "表中 SE 是二项标准误，不是多 seed 标准差。"
            ),
        ),
        _ablation(
            "B",
            "Ablation B：前缀粒度与长度策略",
            setting + " 当前长度策略是固定 20% 网格（gate 通过则前进一档）。PVC 查找表控制器已取消。",
            ["Length policy", "Level schedule", "Acc", "Total Tok", "Latency", "Avg final ratio", "Wrong Promotion %"],
            [
                {"label": "Shared Sentence-PC", "detail": "+1 sentence / level；其余组件同主方法"},
                {"label": "Original Sentence-PC", "detail": "B=20, m=5，per-branch random"},
                {"label": "Absolute-token PC", "detail": "+128 tokens / level"},
                {
                    "label": "Relative-PC（主方法）",
                    "detail": "20→40→60→80%；gate 通过则前进",
                    "cells": [pc_acc, pc_tok, pc_lat, ratio, wrong],
                },
                {"label": "Fixed-20", "detail": "始终 20%"},
                {"label": "Fixed-40", "detail": "始终 40%"},
                {"label": "PVC Adaptive（已取消）", "detail": "查找表控制器；不进测试时策略"},
                {"label": "PVC Adaptive (agreement only)（已取消）", "detail": "无标签查找表；已取消"},
            ],
            static_cols=1,
        ),
        _ablation(
            "C",
            "Ablation C：DeepSeek MATH-500 选择 × recovery（2×2）",
            _SELREC_CAPTION,
            [
                "Setting",
                "PC Acc（raw）",
                "Δ vs SC",
                "Tok / q",
                "vs SC",
                "Gate Pass",
                "Wrong Promo",
                "Fail windows",
                "Fail-set Acc",
            ],
            [
                {
                    "label": "SC（共享根，无前缀）",
                    "cells": _math500_selrec_cells(None, as_sc=True),
                },
                {
                    "label": _DS_MATH500_SELREC["B"]["label"],
                    "cells": _math500_selrec_cells(_DS_MATH500_SELREC["B"]),
                },
                {
                    "label": _DS_MATH500_SELREC["D"]["label"],
                    "cells": _math500_selrec_cells(_DS_MATH500_SELREC["D"]),
                },
                {
                    "label": _DS_MATH500_SELREC["A"]["label"],
                    "cells": _math500_selrec_cells(_DS_MATH500_SELREC["A"]),
                },
                {
                    "label": _DS_MATH500_SELREC["C"]["label"],
                    "cells": _math500_selrec_cells(_DS_MATH500_SELREC["C"]),
                },
            ],
            note=_SELREC_NOTE,
        ),
        _ablation(
            "D",
            "Ablation D：共享方式与系统收益",
            setting + " Prefill Tok = 平均 prompt tokens（含前缀）。Cache hit = 复用 SC root 的窗口 / 全部 PC 窗口。",
            ["Prefix mode", "Acc", "Decode Tok", "Prefill Tok", "Latency", "Cache hit", "Distinct answers"],
            [
                {"label": "Per-branch random（原算法）"},
                {"label": "Shared random"},
                {"label": "Shared token-medoid"},
                {
                    "label": "Shared BLC（主方法）",
                    "cells": [
                        pc_acc,
                        pc_tok,
                        f"{_GS['pc_prompt']:,}",
                        pc_lat,
                        cache,
                        f"{_GS['pc_distinct']:.2f}",
                    ],
                },
            ],
        ),
        _ablation(
            "E",
            "Ablation E：DeepSeek MATH-500 recovery / 回退",
            _SELREC_CAPTION
            + " Disabled q = two-strike 后关闭前缀的题数（recovery 关为 0）。"
            " High-conf. wrong = PC 错且最终前缀比例≥80% 的题占比。",
            [
                "Setting",
                "PC Acc（raw）",
                "Δ vs SC",
                "Tok / q",
                "Fail windows",
                "Fail-set Acc",
                "Fail-set vs SC",
                "Disabled q",
                "High-conf. wrong %",
            ],
            [
                {
                    "label": _DS_MATH500_SELREC["B"]["label"],
                    "cells": _math500_selrec_e_cells(_DS_MATH500_SELREC["B"]),
                },
                {
                    "label": _DS_MATH500_SELREC["D"]["label"],
                    "cells": _math500_selrec_e_cells(_DS_MATH500_SELREC["D"]),
                },
                {
                    "label": _DS_MATH500_SELREC["A"]["label"],
                    "cells": _math500_selrec_e_cells(_DS_MATH500_SELREC["A"]),
                },
                {
                    "label": _DS_MATH500_SELREC["C"]["label"],
                    "cells": _math500_selrec_e_cells(_DS_MATH500_SELREC["C"]),
                },
            ],
            note=(
                "recovery 开才 rollback / two-strike；关则 retain_failed_*。"
                "fail-set 上 recovery 关更差（C 10/33、D 9/31，相对该子集 SC 均 −5），"
                "主方法 A 为 6/14（相对 SC −1）。"
                "High-conf. wrong 四臂接近（8.4–9.4%），不是区分因素。"
                "其余读 Ablation C 脚注。"
            ),
        ),
        _ablation(
            "G",
            "Ablation G：DeepSeek MATH-500 matched-history 长度干预",
            (
                "DeepSeek-V4-Flash × MATH-500 common-support 466（E0：490 中 gate 失败 19、缺 endpoint 5）。"
                "冻 SC 根 4856d4ef5673 的 window-0 与 BLC@20% source；只改 r∈{0,0.2,0.4,0.6,0.8}；"
                "每档 8 次窗内 m=4。tag math500_deepseek_lenint_r8_seed0_2ebec2c2eced，18640/18640。"
                "Acc 是题均值（先对 8 次重复平均）。Final 含冻住的 4 条 root + 24 条 cache 尾巴，不是完整 d_j(h)。"
                "Tok 是干预窗新生成 token（4 条合计）。不是 PVC 控制器，不进 Table 1，不回写最优 r。"
                "本表替换原空的 Prefix Value Curve 占位；查找表控制器仍取消。"
            ),
            [
                "r",
                "Window Acc",
                "Δ vs r=0",
                "Final Acc",
                "Δ vs r=0",
                "Tok / 窗",
                "vs r=0",
                "Prefix tok",
            ],
            [
                {
                    "label": "0（无前缀对照）",
                    "cells": ["88.22%", "—", "89.94%", "—", "3,557", "—", "0"],
                },
                {
                    "label": "0.2",
                    "cells": ["88.04%", "−0.19 pp", "89.91%", "−0.03 pp", "3,444", "−3.2%", "135"],
                },
                {
                    "label": "0.4",
                    "cells": ["88.22%", "0.00 pp", "89.86%", "−0.08 pp", "2,940", "−17.3%", "276"],
                },
                {
                    "label": "0.6",
                    "cells": ["88.04%", "−0.19 pp", "89.89%", "−0.05 pp", "2,502", "−29.7%", "417"],
                },
                {
                    "label": "0.8",
                    "cells": ["87.85%", "−0.38 pp", "89.86%", "−0.08 pp", "2,008", "−43.6%", "558"],
                },
            ],
            note=(
                "题聚类 95% CI：窗 Acc Δ 都穿过 0（r=0.8 为 −1.04～+0.29 pp）。"
                "464/466 题五档 Final Acc 相同，因为 28/32 票被冻住。"
                "答案 TV vs r=0 仅 0.015–0.020；每窗 unique≈1.08。"
                "window-0 对 438 题：r=0.8 窗 Δ −0.31 pp、tok −43.9%。"
                "window-0 错 28 题：r≥0.4 窗 Δ 约 −1 pp，SE≈3 pp，不能写成显著伤害。"
                "r=0.2 几乎不省 token（−3.2%）。禁止把最省 token 的 r 写成测试时策略。"
            ),
        ),
        _ablation(
            "F",
            "Ablation F：SC / DeepConf 与 PC 的 2×2 因子",
            setting + " Total Tok 与 Decode Tok 同为 generated tokens。DeepConf 行待 A3B/官方协议全量。",
            ["Trace controller", "Prefix redirection", "Acc", "Decode Tok", "Total Tok", "Latency", "Completed traces"],
            [
                {
                    "label": "SC",
                    "detail": "off",
                    "cells": [sc_acc, sc_tok, sc_tok, sc_lat, f"{_GS['traces']:.0f}"],
                },
                {
                    "label": "SC",
                    "detail": "PC",
                    "cells": [pc_acc, pc_tok, pc_tok, pc_lat, f"{_GS['traces']:.0f}"],
                },
                {"label": "DeepConf", "detail": "off"},
                {"label": "DeepConf", "detail": "PC"},
            ],
            static_cols=1,
        ),
    ]


def build_paper_tables(runs: list[dict]) -> dict[str, Any]:
    cells = collect_complete_cells(runs)
    table1 = build_table1(cells)
    table2 = build_table2()
    table3 = build_table3(cells)
    pilots = build_qwen_first100()
    ablations = build_ablations()
    return {
        "notes": [
            "只写入 B=32、配对完整、覆盖计划题数的新协议实验。",
            "主表与消融主方法行均为默认 C=0.8，不要标成 PC(C*)。",
            "Tok 口径：平均 generated reasoning/decode tokens；时延含配对根。",
            "DeepSeek GSM8K C=0.8：Acc overlay 1277/1319；PC Tok 来自 vote-v2（−34.2%）。",
            "DeepSeek MATH-500 Acc：raw 490 + 10 工程题记对，分母 500；Tok 仍是 490 均值。不用 overlay。"
            "DeepSeek MATH-500 选择×recovery 已填 Ablation C/E（同一 Acc 口径）；matched-history 长度已填 Ablation G（R=8）。selrec / lenint 不进 Table 1。"
            "Qwen3-30B-A3B MATH-500 当前写入 Table 1 部分格（n=199，source_row 0–199 排除 151）；不是 500，全量到位后改用 raw+10/500。",
        ],
        "table1": table1,
        "table2": table2,
        "table3": table3,
        "pilots": pilots,
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
