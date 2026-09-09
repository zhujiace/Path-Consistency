#!/usr/bin/env python3
"""Local experiment console for the Path-Consistency ICLR 2027 project.

    /home/26xyw/.conda/envs/PC/bin/python console/serve.py --port 8765
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from paper_tables import build_paper_tables


REPO = Path(__file__).resolve().parent.parent
GSM8K_DIR = REPO / "results"
MATH_DIR = Path("/home/26xyw/data/path-consistency-results/math500")
GSM8K_PC_DIR = Path("/home/26xyw/data/path-consistency-results/gsm8k")
AIME_DIR = Path("/home/26xyw/data/path-consistency-results/aime")
GPQA_DIR = Path("/home/26xyw/data/path-consistency-results/gpqa")
PROTOCOL_DIRS = [MATH_DIR, GSM8K_PC_DIR, AIME_DIR, GPQA_DIR]
DATASETS = Path("/home/26xyw/data/Datasets")
TZ = timezone(timedelta(hours=8))
ABSTRACT_DEADLINE = datetime(2026, 9, 18, 23, 59, tzinfo=timezone.utc)
PAPER_DEADLINE = datetime(2026, 9, 25, 23, 59, tzinfo=timezone.utc)

CATALOG = [
    {
        "id": "gsm8k-llama3-sentence-pc",
        "wave": "附录连续性",
        "priority": "P2",
        "title": "GSM8K Sentence-PC 复现（Llama-3-8B）",
        "model": "Llama-3-8B",
        "dataset": "GSM8K test · 1319",
        "methods": "SC / PC C∈{0,0.7,0.8,0.9} / AC / ESC / margin",
        "note": "旧协议 B=20。主文不再用这些数字对 B=32 算节省率。",
        "match": "gsm8k",
    },
    {
        "id": "llama31-math-pilot10-v2",
        "wave": "9/6–9/8 协议",
        "priority": "P0",
        "title": "Llama-3.1-8B MATH500 工程 10 题 v2",
        "model": "Llama-3.1-8B-Instruct",
        "dataset": "MATH-500 engineering split",
        "methods": "SC + PC 配对根",
        "note": "已查看，不得用于选 C*/PVC；主表排除这 10 题。",
        "match": "pilot10_v2",
    },
    {
        "id": "llama31-math-initial50",
        "wave": "9/6–9/8 协议",
        "priority": "P0",
        "title": "Llama-3.1-8B MATH500 初始 50 题",
        "model": "Llama-3.1-8B-Instruct",
        "dataset": "MATH-500 held-out 50",
        "methods": "SC + PC 配对根 · B=16 · C=0.8",
        "note": "仍不作超参选择。MATH-500 对 8B 过难，全量 490 暂停，改 GSM8K。",
        "match": "initial50",
    },
    {
        "id": "llama31-gsm8k-pc",
        "wave": "9/7–9/8 P0",
        "priority": "P0",
        "title": "Llama-3.1-8B GSM8K SC + 新协议 PC",
        "model": "Llama-3.1-8B-Instruct",
        "dataset": "GSM8K test · 1319",
        "methods": "Pass@1 · SC · PC B=32 m=4 C=0.8",
        "note": "完整新协议：配对根、BLC、trial/commit/rollback。C=0.8 仍非 C*；PVC 查找表尚未启用。",
        "match": "gsm8k_llama31",
    },
    {
        "id": "data-aime-gpqa-devmix",
        "wave": "9/6–9/8 数据",
        "priority": "P0",
        "title": "准备 AIME24/25、GPQA-D、Dev-Mix JSONL",
        "model": "—",
        "dataset": "AIME24 · AIME25 · GPQA-D · Dev-Mix 200",
        "methods": "数据转换",
        "note": "CUDA 前转 JSONL。GPQA 需用户接受 gated 条款。",
        "match": "datasets",
    },
    {
        "id": "qwen32-smoke-tp4",
        "wave": "9/6–9/8 接口",
        "priority": "P2",
        "title": "Qwen3-32B vLLM TP=4 smoke（已归档）",
        "model": "Qwen3-32B",
        "dataset": "MATH-500 1 题",
        "methods": "SC/PC",
        "note": "2026-09-08 起本地 Qwen 主模型改为 30B-A3B。32B smoke/eager 490 已暂停归档。",
        "match": "qwen_smoke",
    },
    {
        "id": "deepseek-prefix-logprob",
        "wave": "9/6–9/8 接口",
        "priority": "P0",
        "title": "DeepSeek-V4-Flash prefix + logprob smoke",
        "model": "deepseek-v4-flash",
        "dataset": "接口探测",
        "methods": "prefix completion",
        "note": "不通则 DeepSeek 主结果降级。",
        "match": "deepseek_smoke",
    },
    {
        "id": "deepconf-cpt-minirepro",
        "wave": "9/6–9/8 接口",
        "priority": "P0",
        "title": "隔离环境复现 DeepConf / CPT 最小官方例子",
        "model": "Llama-3.1-8B-Instruct",
        "dataset": "GSM8K smoke 10",
        "methods": "DeepConf · CPT",
        "note": "单卡 smoke。官方 DeepConf 语义 + vLLM 0.8.5 适配；CPT 若无法忠实跑 Llama 则只记缺口。环境在 ~/data，不升级 PC conda。",
        "match": "deepconf_cpt_smoke",
    },
    {
        "id": "pvc-pilot",
        "wave": "9/8–9/10 冻结",
        "priority": "P0-C",
        "title": "干预式 Prefix Value Curve pilot",
        "model": "Llama-3.1 先，Qwen 后",
        "dataset": "Dev-Mix / MATH subset",
        "methods": "r∈{0,0.2,0.4,0.6,0.8}",
        "note": "独立 discovery/evaluation；拆 correct/wrong source。",
        "match": "none",
    },
    {
        "id": "devmix-cstar",
        "wave": "9/8–9/10 冻结",
        "priority": "P0-C",
        "title": "Dev-Mix 选择每模型 C*",
        "model": "三模型各一次",
        "dataset": "Dev-Mix 200",
        "methods": "C∈{0.70,0.80,0.90,0.95,1.00}",
        "note": "非劣约束下最少 token；测试前冻结。",
        "match": "none",
    },
    {
        "id": "selector-ablation-pilot",
        "wave": "9/8–9/10 冻结",
        "priority": "P0-C",
        "title": "BLC vs random / medoid / mean-logprob",
        "model": "Llama-3.1 或 Qwen",
        "dataset": "50–100 题",
        "methods": "selector 消融",
        "note": "G1：BLC 不得明显差于 shared-random。",
        "match": "none",
    },
    {
        "id": "p0a-llama-math500",
        "wave": "9/9 主表",
        "priority": "P0-A",
        "title": "Llama-3.1-8B × MATH-500 全量 SC/PC",
        "model": "Llama-3.1-8B-Instruct",
        "dataset": "MATH-500 490（排除工程 10 题）",
        "methods": "Pass@1 · SC · PC B=32 m=4 C=0.8",
        "note": "V1 CUDA-graph TP=1 GPU1；max_tokens=8192。C=0.8 仍非 C*。",
        "match": "llama31_math500",
    },
    {
        "id": "p0a-llama-aime",
        "wave": "9/9 主表",
        "priority": "P0-A",
        "title": "Llama-3.1-8B × AIME24/25 SC/PC",
        "model": "Llama-3.1-8B-Instruct",
        "dataset": "AIME 2024 · AIME 2025",
        "methods": "Pass@1 · SC · PC B=32 m=4 C=0.8",
        "note": "V1 CUDA-graph TP=1 GPU3 顺序跑两年。C=0.8 仍非 C*。",
        "match": "llama31_aime",
    },
    {
        "id": "p0a-llama-gpqa",
        "wave": "9/9 主表",
        "priority": "P0-A",
        "title": "Llama-3.1-8B × GPQA-Diamond SC/PC",
        "model": "Llama-3.1-8B-Instruct",
        "dataset": "GPQA-D · 198",
        "methods": "Pass@1 · SC · PC B=32 m=4 C=0.8",
        "note": "V1 CUDA-graph TP=1 GPU2。C=0.8 仍非 C*。",
        "match": "llama31_gpqa",
    },
    {
        "id": "p0a-qwen-math500",
        "wave": "9/8 P0-A",
        "priority": "P0-A",
        "title": "Qwen3-30B-A3B × MATH-500 490 SC/PC",
        "model": "Qwen3-30B-A3B",
        "dataset": "MATH-500 490（排除工程 10 题）",
        "methods": "Pass@1 · SC · PC B=32 m=4 C=0.8",
        "note": "V1 CUDA-graph TP=4 + CUDA-12.2 NCCL；thinking on；物理窗口 4。C=0.8 仍非 C*。",
        "match": "qwen30_a3b_math500",
    },
    {
        "id": "p0a-qwen-aime-gpqa",
        "wave": "9/10–9/14 主表",
        "priority": "P0-A",
        "title": "Qwen3-30B-A3B × AIME/GPQA SC/PC",
        "model": "Qwen3-30B-A3B",
        "dataset": "AIME24 · AIME25 · GPQA-D",
        "methods": "Pass@1 · SC · PC",
        "note": "MATH-500 之后。JSONL 仍缺。",
        "match": "none",
    },
    {
        "id": "p0a-deepseek",
        "wave": "9/10–9/14 主表",
        "priority": "P0-A",
        "title": "DeepSeek-V4-Flash × 四主数据集 SC/PC",
        "model": "deepseek-v4-flash",
        "dataset": "MATH · AIME · GPQA",
        "methods": "SC · PC-medoid fallback 若无 logprob",
        "note": "MATH-500 490 已完成。AIME/GPQA 为独立 tag。C=0.8 仍非 C*。",
        "match": "deepseek_math500",
    },
    {
        "id": "p0a-deepseek-aime",
        "wave": "9/8–9/10 主表",
        "priority": "P0-A",
        "title": "DeepSeek-V4-Flash × AIME24/25 SC/PC",
        "model": "deepseek-v4-flash",
        "dataset": "AIME 2024 · AIME 2025",
        "methods": "Pass@1 · SC · PC B=32 m=4 C=0.8",
        "note": "配对根 v2；BLC；整数 0–999。C=0.8 仍非 C*。",
        "match": "deepseek_aime",
    },
    {
        "id": "p0a-deepseek-gpqa",
        "wave": "9/8–9/10 主表",
        "priority": "P0-A",
        "title": "DeepSeek-V4-Flash × GPQA-Diamond SC/PC",
        "model": "deepseek-v4-flash",
        "dataset": "GPQA-D · 198",
        "methods": "Pass@1 · SC · PC B=32 m=4 C=0.8",
        "note": "选项 seed=2027 写入 JSONL。C=0.8 仍非 C*。",
        "match": "deepseek_gpqa",
    },
    {
        "id": "p0b-deepconf-pc",
        "wave": "9/10–9/14 主表",
        "priority": "P0-B",
        "title": "Qwen3-30B-A3B DeepConf 与 DeepConf+PC",
        "model": "Qwen3-30B-A3B",
        "dataset": "AIME24+25 · MATH-500 · GPQA",
        "methods": "2×2 parent × prefix",
        "note": "不可用 AC/ESC 代替。",
        "match": "none",
    },
    {
        "id": "p0b-cpt",
        "wave": "9/10–9/14 主表",
        "priority": "P0-B",
        "title": "CPT matched-budget 比较",
        "model": "Qwen3-30B-A3B",
        "dataset": "AIME24/25 · MATH-500",
        "methods": "CPT",
        "note": "官方实现适配失败则不得写公平比较完成。",
        "match": "none",
    },
    {
        "id": "p0c-ablations",
        "wave": "9/14–9/17 消融",
        "priority": "P0-C",
        "title": "Beta / 长度 / selector / rollback / PVC 消融",
        "model": "Qwen3-30B-A3B 优先",
        "dataset": "MATH 100 + AIME 60",
        "methods": "Ablation A–G",
        "note": "representative subset 即可。",
        "match": "none",
    },
    {
        "id": "p0d-repeats",
        "wave": "9/14–9/17 统计",
        "priority": "P0-D",
        "title": "主表核心行 seed/repetition 1/2 与 CI",
        "model": "主方法",
        "dataset": "主表行",
        "methods": "重复",
        "note": "9/16 冻结数字。",
        "match": "none",
    },
    {
        "id": "p1-dts",
        "wave": "P1 缓冲",
        "priority": "P1",
        "title": "DTS-Stable smoke（一天不通则降级）",
        "model": "Qwen3-30B-A3B",
        "dataset": "AIME24/25",
        "methods": "DTS-Stable",
        "note": "资源允许进主表。",
        "match": "none",
    },
    {
        "id": "writing-abstract",
        "wave": "9/15–9/18 写作",
        "priority": "P0",
        "title": "摘要与主体改写",
        "model": "—",
        "dataset": "—",
        "methods": "写作",
        "note": "只写已冻结可复算数字。摘要 9/18 23:59 AOE。",
        "match": "none",
    },
]


def now_cst() -> datetime:
    return datetime.now(TZ)


def iso_cst(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(TZ).strftime("%Y-%m-%d %H:%M:%S CST")


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def run_cmd(args: list[str]) -> str:
    try:
        out = subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True, timeout=8)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return out


def load_json_summary_only(path: Path) -> dict | None:
    """Read GSM8K result JSON without loading per_question."""
    try:
        with path.open() as handle:
            raw = handle.read(65536)
        marker = '"per_question"'
        index = raw.find(marker)
        if index > 0:
            raw = raw[:index].rstrip().rstrip(",") + "\n}\n"
        data = json.loads(raw)
        return data.get("summary", data)
    except (OSError, json.JSONDecodeError):
        return None


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open() as handle:
        for line in handle:
            if line.strip():
                n += 1
    return n


def last_jsonl_row(path: Path) -> dict | None:
    if not path.exists():
        return None
    last = None
    with path.open() as handle:
        for line in handle:
            if line.strip():
                last = line
    if not last:
        return None
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return None


def gpus() -> list[dict]:
    text = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        used = float(parts[2])
        total = float(parts[4]) or 1.0
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "used_mib": used,
                "free_mib": float(parts[3]),
                "total_mib": total,
                "util": float(parts[5]),
                "busy": used > 4000 or float(parts[5]) > 20,
            }
        )
    return rows


def running_processes() -> list[dict]:
    text = run_cmd(["ps", "-eo", "pid,etime,pcpu,rss,cmd"])
    rows = []
    for line in text.splitlines()[1:]:
        if not re.search(r"run_math\.py|run_math_deepseek\.py|run_gsm8k\.py|EngineCore", line):
            continue
        if "console/serve.py" in line:
            continue
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        cmd = parts[4]
        if "grep" in cmd:
            continue
        rows.append(
            {
                "pid": int(parts[0]),
                "etime": parts[1],
                "cpu": parts[2],
                "rss_mb": round(int(parts[3]) / 1024, 1),
                "cmd": cmd[:240],
            }
        )
    return rows


def gsm8k_results() -> list[dict]:
    order = [
        ("SC_seed0", "SC"),
        ("PC_C0_seed0", "PC C=0"),
        ("PC_C0.7_seed0", "PC C=0.7"),
        ("PC_C0.8_seed0", "PC C=0.8"),
        ("PC_C0.9_seed0", "PC C=0.9"),
        ("AC_seed0", "AC"),
        ("ESC_seed0", "ESC"),
        ("MARGIN_seed0", "margin（非 MARS）"),
    ]
    sc_tokens = None
    rows = []
    for tag, label in order:
        path = GSM8K_DIR / f"{tag}.json"
        summary = load_json_summary_only(path) if path.exists() else None
        if not summary:
            continue
        tokens = summary.get("avg_tokens_per_question")
        if tag == "SC_seed0":
            sc_tokens = tokens
        saving = None
        if sc_tokens and tokens is not None:
            saving = 1 - tokens / sc_tokens
        rows.append(
            {
                "tag": tag,
                "label": label,
                "status": "complete"
                if summary.get("questions_done") == summary.get("num_questions")
                else "partial",
                "done": summary.get("questions_done"),
                "total": summary.get("num_questions"),
                "accuracy": summary.get("accuracy"),
                "avg_tokens": tokens,
                "avg_latency_s": summary.get("avg_latency_per_question"),
                "token_saving_vs_sc": saving,
            }
        )
    return rows


def classify_math_run(stem: str) -> str:
    name = stem.lower()
    if "diagnostic" in name or "nothink" in name:
        return "debug"
    if "smoke" in name or "pairedroot" in name:
        return "smoke"
    if "pilot10" in name and "_v2_" not in name and "v2" not in name:
        return "superseded"
    if (
        "pilot10" in name
        or "initial50" in name
        or "deepseek" in name
        or "llama31" in name
    ):
        return "primary"
    if "qwen3_30b_a3b" in name and "smoke" not in name:
        return "primary"
    if "qwen3_32b" in name and "smoke" not in name:
        return "other"
    return "other"


def math_runs() -> list[dict]:
    runs = []
    for directory in PROTOCOL_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.run.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                record = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            config = record.get("config") or {}
            fingerprint = record.get("fingerprint") or path.name.split("_")[-1].replace(".run.json", "")
            result_paths = record.get("result_paths") or {}
            sc_path = Path(result_paths.get("sc") or "")
            pc_path = Path(result_paths.get("pc") or "")
            sc_n = count_jsonl(sc_path) if sc_path.name else 0
            pc_n = count_jsonl(pc_path) if pc_path.name else 0
            planned = len(config.get("selected_question_ids") or [])
            pid = record.get("pid")
            status = record.get("status") or "unknown"
            alive = pid_alive(pid)
            if (
                status
                not in {
                    "complete",
                    "failed",
                    "terminated_protocol_invalid",
                    "paused",
                }
                and pid
                and not alive
            ):
                status = "stale"
            summary_path = path.with_name(path.name.replace(".run.json", ".summary.json"))
            summary = {}
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text())
                except json.JSONDecodeError:
                    summary = {}
            methods = (summary.get("methods") or {}) if summary else {}
            costs = (summary.get("parent_costs") or {}) if summary else {}
            sc_method = methods.get("SC") or {}
            pc_method = methods.get("PC") or {}
            pass1_method = methods.get("Pass@1(SC)") or methods.get("Pass@1") or {}
            sc_cost = costs.get("sc") or {}
            pc_cost = costs.get("pc") or {}
            sc_acc = sc_method.get("accuracy")
            pc_acc = pc_method.get("accuracy")
            sc_tok = sc_cost.get("avg_generated_tokens")
            pc_tok = pc_cost.get("avg_generated_tokens")
            last_pc = last_jsonl_row(pc_path) if pc_path.name else None
            last_sc = last_jsonl_row(sc_path) if sc_path.name else None
            last = last_pc or last_sc
            tag = path.name.replace(".run.json", "")
            elapsed = summary.get("run_elapsed_seconds")
            if elapsed is None and record.get("started_at_utc"):
                try:
                    started = datetime.fromisoformat(str(record["started_at_utc"]))
                    if started.tzinfo is None:
                        started = started.replace(tzinfo=timezone.utc)
                    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
                except (TypeError, ValueError):
                    elapsed = None
            eta = None
            paired = min(sc_n, pc_n)
            if status in {"running"} and paired and planned and elapsed:
                rate = elapsed / max(paired, 1)
                eta = rate * max(planned - paired, 0)
            runs.append(
                {
                    "file": str(path),
                    "tag": tag,
                    "fingerprint": fingerprint,
                    "kind": classify_math_run(path.name),
                    "status": status,
                    "pid": pid,
                    "pid_alive": alive,
                    "gpus": config.get("gpus"),
                    "model": Path(str(config.get("model") or "")).name or str(config.get("model")),
                    "model_path": config.get("model"),
                    "dataset": config.get("dataset"),
                    "model_type": (record.get("engine_times") or {}).get("model_type")
                    or (summary.get("engine_times") or {}).get("model_type"),
                    "protocol": config.get("protocol_version"),
                    "planned": planned,
                    "sc_done": sc_n,
                    "pc_done": pc_n,
                    "paired": paired,
                    "max_branches": config.get("max_branches"),
                    "beta_threshold": config.get("beta_threshold"),
                    "max_tokens": config.get("max_tokens"),
                    "sc_accuracy": sc_acc,
                    "pc_accuracy": pc_acc,
                    "sc_correct": sc_method.get("correct"),
                    "sc_questions": sc_method.get("questions"),
                    "pc_correct": pc_method.get("correct"),
                    "pc_questions": pc_method.get("questions"),
                    "sc_avg_tokens": sc_tok,
                    "pc_avg_tokens": pc_tok,
                    "sc_total_generated": sc_cost.get("total_generated_tokens"),
                    "pc_total_generated": pc_cost.get("total_generated_tokens"),
                    "sc_total_generation_s": sc_cost.get("total_generation_seconds"),
                    "pc_total_generation_s": pc_cost.get("total_generation_seconds"),
                    "sc_avg_generation_s": sc_cost.get("avg_generation_seconds"),
                    "pc_avg_generation_s": pc_cost.get("avg_generation_seconds"),
                    "pass1_accuracy": pass1_method.get("accuracy"),
                    "pass1_correct": pass1_method.get("correct"),
                    "pass1_questions": pass1_method.get("questions"),
                    "decode_saving": (1 - pc_tok / sc_tok)
                    if sc_tok and pc_tok
                    else None,
                    "truncations_sc": sc_cost.get("truncations"),
                    "started": record.get("started_at_utc"),
                    "finished": record.get("finished_at_utc"),
                    "elapsed_s": elapsed,
                    "eta_s": eta,
                    "last_question": None
                    if not last
                    else {
                        "id": last.get("question_id"),
                        "method": last.get("method"),
                        "sc": (last.get("predictions") or {}).get("SC")
                        or (last.get("predictions") or {}).get("PC"),
                        "tokens": last.get("total_generated_tokens"),
                        "ratio": last.get("final_prefix_ratio"),
                        "truncations": last.get("truncations"),
                    },
                }
            )
    return runs


def dataset_status() -> list[dict]:
    specs = [
        ("GSM8K test", DATASETS / "gsm8k_test.jsonl", True),
        ("MATH-500 test", DATASETS / "math500_test.jsonl", True),
        ("AIME 2024", DATASETS / "aime_2024.jsonl", False),
        ("AIME 2025", DATASETS / "aime_2025.jsonl", False),
        ("GPQA-Diamond", DATASETS / "gpqa_diamond.jsonl", False),
        ("Dev-Mix", DATASETS / "devmix_2027.jsonl", False),
    ]
    rows = []
    for name, path, required in specs:
        n = count_jsonl(path) if path.exists() else 0
        rows.append(
            {
                "name": name,
                "path": str(path),
                "exists": path.exists(),
                "rows": n,
                "required": required,
                "status": "ready" if path.exists() and n else "missing",
            }
        )
    return rows


def catalog_status(math: list[dict], gsm8k: list[dict], datasets: list[dict]) -> list[dict]:
    math_by = {row["kind"] + ":" + row["tag"]: row for row in math}
    has_pilot10_v2 = any(
        row["kind"] == "primary" and "pilot10" in row["tag"] and row["status"] == "complete"
        for row in math
    )
    initial50 = next((row for row in math if "initial50" in row["tag"]), None)
    qwen_smoke = next(
        (
            row
            for row in math
            if "qwen3_32b" in row["tag"].lower() and "smoke" in row["tag"].lower()
        ),
        None,
    )
    gsm8k_complete = bool(gsm8k) and all(row["status"] == "complete" or (row["done"] or 0) >= 1316 for row in gsm8k)
    aime_ready = all(
        row["status"] == "ready" for row in datasets if row["name"].startswith("AIME")
    )
    gpqa_ready = any(row["name"].startswith("GPQA") and row["status"] == "ready" for row in datasets)
    devmix_ready = any(row["name"].startswith("Dev-Mix") and row["status"] == "ready" for row in datasets)

    out = []
    for item in CATALOG:
        key = item["match"]
        status = "queued"
        detail = item.get("note") or ""
        progress = None
        link = None
        if key == "gsm8k":
            status = "complete" if gsm8k_complete else "queued"
            if gsm8k:
                sc = next((r for r in gsm8k if r["tag"] == "SC_seed0"), None)
                pc = next((r for r in gsm8k if r["tag"] == "PC_C0.8_seed0"), None)
                if sc and pc:
                    detail = (
                        f"SC {100 * sc['accuracy']:.1f}% / {sc['avg_tokens']:.0f} tok；"
                        f"PC C=0.8 {100 * pc['accuracy']:.1f}% / {pc['avg_tokens']:.0f} tok"
                    )
        elif key == "pilot10_v2":
            status = "complete" if has_pilot10_v2 else "queued"
            hit = next((r for r in math if "pilot10" in r["tag"] and "v2" in r["tag"] and r["status"] == "complete"), None)
            if hit:
                progress = {"done": hit["paired"], "total": hit["planned"]}
                if hit["sc_accuracy"] is not None:
                    detail = (
                        f"SC={100 * hit['sc_accuracy']:.0f}% PC={100 * (hit['pc_accuracy'] or 0):.0f}%；"
                        f"decode 节省 {100 * (hit['decode_saving'] or 0):.1f}%"
                    )
        elif key == "initial50":
            if initial50:
                status = initial50["status"]
                progress = {"done": initial50["paired"], "total": initial50["planned"]}
                if initial50["sc_accuracy"] is not None:
                    detail = (
                        f"已配对 {initial50['paired']}/{initial50['planned']}；"
                        f"SC {100 * initial50['sc_accuracy']:.0f}% / "
                        f"PC {100 * (initial50['pc_accuracy'] or 0):.0f}%"
                    )
                link = initial50["tag"]
            else:
                status = "queued"
        elif key == "datasets":
            missing = [row["name"] for row in datasets if row["status"] != "ready" and not row["name"].startswith("GSM8K") and not row["name"].startswith("MATH")]
            if not missing:
                status = "complete"
            elif aime_ready or gpqa_ready or devmix_ready:
                status = "partial"
                detail = "仍缺：" + "、".join(missing)
            else:
                status = "queued"
                detail = "已有 GSM8K + MATH-500。仍缺：" + "、".join(missing)
        elif key == "qwen_smoke":
            if qwen_smoke:
                status = "partial"
                detail = (
                    f"存在 run `{qwen_smoke['tag']}` 状态 {qwen_smoke['status']}。"
                    "TP=4 CUDA-graph 仍失败，不可批量。"
                )
            else:
                status = "queued"
        elif key == "gsm8k_llama31":
            hits = [
                row
                for row in math
                if "gsm8k_llama31" in row["tag"] and "smoke" not in row["tag"]
            ]
            if not hits:
                status = "queued"
            else:
                paired = sum(row["paired"] or 0 for row in hits)
                planned = sum(row["planned"] or 0 for row in hits) or 1319
                progress = {"done": paired, "total": planned}
                link = hits[0]["tag"]
                if any(row["status"] in {"running", "loading_engine"} for row in hits):
                    status = "running"
                elif all(row["status"] == "complete" for row in hits):
                    status = "complete"
                elif any(row["status"] in {"failed", "stale"} for row in hits):
                    status = "stale" if any(row["status"] == "stale" for row in hits) else "failed"
                else:
                    status = "partial"
                sc_c = sum(row.get("sc_correct") or 0 for row in hits)
                sc_n = sum(row.get("sc_questions") or 0 for row in hits)
                pc_c = sum(row.get("pc_correct") or 0 for row in hits)
                pc_n = sum(row.get("pc_questions") or 0 for row in hits)
                if sc_n and pc_n:
                    detail = (
                        f"已配对 {paired}/{planned}；"
                        f"SC {100 * sc_c / sc_n:.0f}% / PC {100 * pc_c / pc_n:.0f}%"
                    )
                else:
                    detail = f"已配对 {paired}/{planned}；{len(hits)} 个 shard"
        elif key == "qwen30_a3b_math500":
            hits = [
                row
                for row in math
                if (
                    "qwen3_30b_a3b" in row["tag"].lower()
                    or "qwen30_a3b" in row["tag"].lower()
                )
                and "smoke" not in row["tag"].lower()
                and "probe" not in row["tag"].lower()
                and "math500" in row["tag"].lower()
            ]
            if not hits:
                status = "queued"
            else:
                hit = hits[0]
                status = hit["status"]
                progress = {"done": hit["paired"], "total": hit["planned"]}
                link = hit["tag"]
                if status == "paused":
                    detail = "已暂停。"
                elif hit["sc_accuracy"] is not None:
                    detail = (
                        f"已配对 {hit['paired']}/{hit['planned']}；"
                        f"SC {100 * hit['sc_accuracy']:.0f}% / "
                        f"PC {100 * (hit['pc_accuracy'] or 0):.0f}%"
                    )
                else:
                    detail = f"已配对 {hit['paired']}/{hit['planned']}"
        elif key == "deepconf_cpt_smoke":
            smoke_file = Path("/home/26xyw/data/path-consistency-results/deepconf-cpt/smoke_llama31_gsm8k.json")
            if smoke_file.exists():
                try:
                    smoke = json.loads(smoke_file.read_text())
                except json.JSONDecodeError:
                    smoke = {}
                dc = smoke.get("deepconf") or {}
                cpt = smoke.get("cpt") or {}
                dc_ok = dc.get("status") == "complete"
                cpt_status = cpt.get("status") or "blocked"
                if dc_ok and cpt_status == "complete":
                    status = "complete"
                elif dc_ok:
                    status = "partial"
                else:
                    status = "queued"
                dc_label = dc.get("label") or "DeepConf"
                n = dc.get("n_questions")
                acc = dc.get("final_accuracy")
                acc_s = f"{100 * acc:.0f}%" if isinstance(acc, (int, float)) else "—"
                detail = (
                    f"{dc_label} n={n} acc={acc_s}；"
                    f"CPT {cpt_status}（非公平 Llama 比较）"
                )
            else:
                status = "queued"
        elif key == "deepseek_smoke":
            smoke_file = MATH_DIR / "reports" / "deepseek_smoke.json"
            if smoke_file.exists():
                try:
                    smoke = json.loads(smoke_file.read_text())
                except json.JSONDecodeError:
                    smoke = {}
                protocol = smoke.get("protocol") or {}
                status = "complete" if protocol.get("ready_for_math500_pc") else "failed"
                detail = (
                    f"selector={protocol.get('selector')}；"
                    f"prefix={protocol.get('prefix_variant')}；"
                    f"logprobs={'yes' if protocol.get('logprobs_ok') else 'no'}"
                )
            else:
                status = "queued"
        elif key == "llama31_math500":
            hit = next(
                (
                    row
                    for row in math
                    if "llama31" in row["tag"].lower()
                    and "math500" in row["tag"].lower()
                    and "initial50" not in row["tag"].lower()
                    and "pilot" not in row["tag"].lower()
                ),
                None,
            )
            if hit:
                status = hit["status"]
                progress = {"done": hit["paired"], "total": hit["planned"]}
                link = hit["tag"]
                if hit["sc_accuracy"] is not None:
                    detail = (
                        f"已配对 {hit['paired']}/{hit['planned']}；"
                        f"SC {100 * hit['sc_accuracy']:.0f}% / "
                        f"PC {100 * (hit['pc_accuracy'] or 0):.0f}%"
                    )
                else:
                    detail = f"{hit['paired']}/{hit['planned']} {hit['status']}"
            else:
                status = "queued"
        elif key == "llama31_aime":
            hits = [
                row
                for row in math
                if "llama31" in row["tag"].lower()
                and ("aime_2024" in row["tag"].lower() or "aime_2025" in row["tag"].lower())
            ]
            if hits:
                paired = sum(row["paired"] or 0 for row in hits)
                planned = sum(row["planned"] or 0 for row in hits)
                progress = {"done": paired, "total": planned or 60}
                if any(row["status"] in {"running", "loading_engine"} for row in hits):
                    status = "running"
                elif all(row["status"] == "complete" for row in hits) and len(hits) >= 2:
                    status = "complete"
                elif any(row["status"] == "failed" for row in hits):
                    status = "failed"
                else:
                    status = "partial"
                link = hits[0]["tag"]
                bits = []
                for row in sorted(hits, key=lambda item: item["tag"]):
                    year = "24" if "2024" in row["tag"] else "25"
                    sc = row["sc_accuracy"]
                    pc = row["pc_accuracy"]
                    if sc is not None:
                        bits.append(
                            f"AIME{year} SC {100 * sc:.0f}% / PC {100 * (pc or 0):.0f}%"
                        )
                    else:
                        bits.append(f"AIME{year} {row['paired']}/{row['planned']}")
                detail = "；".join(bits)
            else:
                status = "queued"
        elif key == "llama31_gpqa":
            hit = next(
                (
                    row
                    for row in math
                    if "llama31" in row["tag"].lower() and "gpqa" in row["tag"].lower()
                ),
                None,
            )
            if hit:
                status = hit["status"]
                progress = {"done": hit["paired"], "total": hit["planned"]}
                link = hit["tag"]
                if hit["sc_accuracy"] is not None:
                    detail = (
                        f"已配对 {hit['paired']}/{hit['planned']}；"
                        f"SC {100 * hit['sc_accuracy']:.0f}% / "
                        f"PC {100 * (hit['pc_accuracy'] or 0):.0f}%"
                    )
                else:
                    detail = f"{hit['paired']}/{hit['planned']} {hit['status']}"
            else:
                status = "queued"
        elif key == "deepseek_math500":
            hit = next((row for row in math if "deepseek" in row["tag"].lower() and "math500" in row["tag"].lower()), None)
            if hit:
                status = hit["status"]
                progress = {"done": hit["paired"], "total": hit["planned"]}
                link = hit["tag"]
                if hit["sc_accuracy"] is not None:
                    detail = (
                        f"已配对 {hit['paired']}/{hit['planned']}；"
                        f"SC {100 * hit['sc_accuracy']:.0f}% / "
                        f"PC {100 * (hit['pc_accuracy'] or 0):.0f}%"
                    )
            else:
                status = "queued"
        elif key == "deepseek_aime":
            hits = [
                row
                for row in math
                if "deepseek" in row["tag"].lower()
                and ("aime_2024" in row["tag"].lower() or "aime_2025" in row["tag"].lower())
            ]
            if hits:
                paired = sum(row["paired"] or 0 for row in hits)
                planned = sum(row["planned"] or 0 for row in hits)
                progress = {"done": paired, "total": planned or 60}
                if any(row["status"] == "running" for row in hits):
                    status = "running"
                elif all(row["status"] == "complete" for row in hits) and len(hits) >= 2:
                    status = "complete"
                elif any(row["status"] == "failed" for row in hits):
                    status = "failed"
                else:
                    status = "partial"
                link = hits[0]["tag"]
                bits = []
                for row in sorted(hits, key=lambda item: item["tag"]):
                    year = "24" if "2024" in row["tag"] else "25"
                    sc = row["sc_accuracy"]
                    pc = row["pc_accuracy"]
                    if sc is not None:
                        bits.append(
                            f"AIME{year} SC {100 * sc:.0f}% / PC {100 * (pc or 0):.0f}%"
                        )
                    else:
                        bits.append(f"AIME{year} {row['paired']}/{row['planned']}")
                detail = "；".join(bits)
            else:
                status = "queued"
        elif key == "deepseek_gpqa":
            hit = next(
                (
                    row
                    for row in math
                    if "deepseek" in row["tag"].lower() and "gpqa" in row["tag"].lower()
                ),
                None,
            )
            if hit:
                status = hit["status"]
                progress = {"done": hit["paired"], "total": hit["planned"]}
                link = hit["tag"]
                if hit["sc_accuracy"] is not None:
                    detail = (
                        f"已配对 {hit['paired']}/{hit['planned']}；"
                        f"SC {100 * hit['sc_accuracy']:.0f}% / "
                        f"PC {100 * (hit['pc_accuracy'] or 0):.0f}%"
                    )
                else:
                    detail = f"{hit['paired']}/{hit['planned']} {hit['status']}"
            else:
                status = "queued"
        else:
            status = "queued"
        out.append({**item, "status": status, "detail": detail, "progress": progress, "run_tag": link})
    return out


def collect_state() -> dict:
    gpu_rows = gpus()
    procs = running_processes()
    gsm8k = gsm8k_results()
    math = math_runs()
    datasets = dataset_status()
    catalog = catalog_status(math, gsm8k, datasets)
    counts = {"complete": 0, "running": 0, "partial": 0, "queued": 0, "stale": 0, "failed": 0}
    for item in catalog:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    running = [row for row in math if row["status"] in {"running", "loading_engine"}]
    running.sort(key=lambda row: row.get("started") or "", reverse=True)
    stale = [row for row in math if row["status"] == "stale"]
    return {
        "generated_at": iso_cst(now_cst()),
        "deadlines": {
            "abstract_iso": ABSTRACT_DEADLINE.isoformat(),
            "paper_iso": PAPER_DEADLINE.isoformat(),
            "abstract": "2026-09-18 23:59 AOE",
            "paper": "2026-09-25 23:59 AOE",
        },
        "counts": counts,
        "running_math": len(running),
        "gpus": gpu_rows,
        "processes": procs,
        "running": running,
        "stale": stale,
        "gsm8k": gsm8k,
        "math": math,
        "datasets": datasets,
        "catalog": catalog,
        "paper": build_paper_tables(math),
    }


PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>Path-Consistency 实验控制台</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root {
    --bg: #111114;
    --bg2: #1a1a1f;
    --bg3: #222228;
    --line: #2e2e36;
    --text: #ececec;
    --muted: #9a9aa6;
    --dim: #6f6f7a;
    --accent: #7aa2f7;
    --ok: #7dcfff;
    --warn: #e0af68;
    --bad: #f7768e;
    --run: #9ece6a;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; background: var(--bg); color: var(--text);
    font: 13px/1.45 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
  .wrap { max-width: 1440px; margin: 0 auto; padding: 28px 24px 64px; }
  h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; letter-spacing: -0.02em; }
  h2 { font-size: 13px; font-weight: 600; color: var(--muted); text-transform: uppercase;
    letter-spacing: 0.08em; margin: 28px 0 10px; }
  h3 { font-size: 15px; font-weight: 600; margin: 22px 0 6px; letter-spacing: -0.01em; }
  .caption { color: var(--muted); font-size: 12px; margin: 0 0 10px; max-width: 110ch; }
  .paper-scroll { overflow-x: auto; border: 1px solid var(--line); background: var(--bg2); }
  table.paper { width: max-content; min-width: 100%; font-size: 12px; }
  table.paper th, table.paper td { padding: 6px 8px; white-space: nowrap; }
  table.paper th { text-align: center; }
  table.paper th.left, table.paper td.left { text-align: left; }
  table.paper td.num { text-align: right; font-variant-numeric: tabular-nums; }
  table.paper td.filled { color: var(--text); font-weight: 600; background: #1e2430; }
  table.paper td.empty { color: transparent; }
  table.paper .cont { color: var(--warn); }
  table.paper tr.method-pc td.left { color: var(--ok); }
  table.paper .stub { color: var(--dim); font-size: 11px; }
  table.paper .stub.save { color: var(--run); }
  table.paper .stub.cost { color: var(--bad); }
  .fill-meta { color: var(--muted); font-size: 12px; margin-bottom: 8px; }
  .sub { color: var(--muted); margin-bottom: 18px; }
  .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: stretch; }
  .stat { background: var(--bg2); border: 1px solid var(--line); padding: 12px 14px; min-width: 140px; flex: 1; }
  .stat .v { font-size: 20px; font-variant-numeric: tabular-nums; font-weight: 600; white-space: nowrap; }
  .stat .l { color: var(--muted); font-size: 12px; margin-top: 2px; }
  .ok { color: var(--run); } .warn { color: var(--warn); } .bad { color: var(--bad); } .info { color: var(--accent); }
  table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
  th { text-align: left; color: var(--dim); font-weight: 500; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.06em; padding: 8px 8px; border-bottom: 1px solid var(--line); }
  td { padding: 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
  tr:hover td { background: var(--bg2); }
  .pill { display: inline-block; font-size: 11px; padding: 1px 7px; border: 1px solid var(--line);
    color: var(--muted); white-space: nowrap; }
  .pill.run { color: var(--run); border-color: #3d5a2c; }
  .pill.complete { color: var(--ok); border-color: #2a4a5a; }
  .pill.queued { color: var(--dim); }
  .pill.partial { color: var(--warn); border-color: #5a4a28; }
  .pill.stale, .pill.failed { color: var(--bad); border-color: #5a2a36; }
  .pill.paused { color: var(--warn); border-color: #5a4a28; }
  .bar { height: 6px; background: var(--bg3); margin-top: 6px; }
  .bar > span { display: block; height: 100%; background: var(--accent); }
  .gpu { flex: 1; min-width: 160px; background: var(--bg2); border: 1px solid var(--line); padding: 10px 12px; }
  .gpu .mem { color: var(--muted); font-size: 12px; }
  .tabs button { background: transparent; color: var(--muted); border: 1px solid var(--line);
    padding: 4px 10px; cursor: pointer; }
  .tabs button.on { color: var(--text); border-color: var(--accent); }
  .note { color: var(--muted); max-width: 72ch; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
  .dim { color: var(--dim); }
  .live { color: var(--run); }
  a { color: var(--accent); text-decoration: none; }
  .empty { color: var(--dim); padding: 12px 0; }
  .call { border: 1px solid var(--line); padding: 12px 14px; background: var(--bg2); margin: 8px 0 0; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Path-Consistency 实验控制台</h1>
  <div class="sub">ICLR 2027 改稿 · 数据每 8 秒从磁盘与 <span class="mono">nvidia-smi</span> 刷新 · <span id="stamp"></span></div>

  <div class="row" id="kpis"></div>
  <div class="call" id="headline"></div>

  <h2>GPU</h2>
  <div class="row" id="gpus"></div>

  <h2>正在进行</h2>
  <div id="running"></div>

  <h2>论文表格</h2>
  <div class="note" id="paper-notes"></div>
  <div id="table1"></div>
  <div id="table2"></div>
  <div id="table3"></div>
  <h2>消融实验</h2>
  <div id="ablations"></div>

  <h2>实验清单</h2>
  <div class="tabs" id="tabs"></div>
  <div id="catalog"></div>

  <h2>GSM8K 已完成结果（Llama-3-8B，旧 Sentence-PC）</h2>
  <div id="gsm8k"></div>

  <h2>MATH-500 运行记录</h2>
  <div id="math"></div>

  <h2>数据集</h2>
  <div id="datasets"></div>
</div>
<script>
const FILTERS = ["全部","进行中","已完成","未开始","部分/卡住"];
let filter = "全部";
let state = null;

function pct(x, d=1){ if(x==null||Number.isNaN(x)) return "—"; return (100*x).toFixed(d)+"%"; }
function num(x, d=1){ if(x==null||Number.isNaN(x)) return "—"; return Number(x).toFixed(d); }
function pill(status){
  const map = {complete:"完成", running:"进行中", queued:"未开始", partial:"部分", stale:"进程丢失", failed:"失败", paused:"已暂停", terminated_protocol_invalid:"已作废", missing:"缺失", ready:"就绪", loading_engine:"加载中"};
  return `<span class="pill ${status}">${map[status]||status}</span>`;
}
function eta(s){
  if(s==null) return "—";
  if(s<60) return Math.round(s)+"s";
  if(s<3600) return Math.round(s/60)+" min";
  return (s/3600).toFixed(1)+" h";
}
function bar(done, total){
  const p = total ? Math.min(100, 100*done/total) : 0;
  return `<div class="bar"><span style="width:${p}%"></span></div>`;
}

function acc(run, which){
  const p = which==="sc" ? run.sc_accuracy : run.pc_accuracy;
  const c = which==="sc" ? run.sc_correct : run.pc_correct;
  const n = which==="sc" ? run.sc_questions : run.pc_questions;
  if(p==null) return "—";
  if(c==null || n==null) return pct(p,0);
  return pct(p,0) + " ("+c+"/"+n+")";
}
function where(run){
  if(run.gpus==="api" || (run.model_type||"").indexOf("deepseek")>=0) return "API";
  return "GPU " + (run.gpus==null ? "—" : run.gpus);
}
function fmtAcc(cell){
  if(!cell || cell.acc==null) return "";
  const pctv = (100*cell.acc).toFixed(2);
  if(cell.correct==null || cell.n==null) return pctv;
  return pctv + '<div class="stub">' + cell.correct + "/" + cell.n + "</div>";
}
function fmtTok(cell){
  if(!cell || cell.tok==null) return "";
  const main = Math.round(cell.tok).toLocaleString("en-US");
  if(cell.tok_saving==null) return main;
  const p = 100 * cell.tok_saving;
  const sign = p >= 0 ? "−" : "+";
  const cls = p >= 0 ? "save" : "cost";
  return main + `<div class="stub ${cls}">${sign}${Math.abs(p).toFixed(1)}%</div>`;
}
function fmtNum(x, d, suffix){
  if(x==null || Number.isNaN(x)) return "";
  const t = Number(x).toFixed(d);
  return suffix ? t+suffix : t;
}
function tdCell(html, filled){
  return `<td class="num ${filled?"filled":"empty"}">${html||""}</td>`;
}
function renderTable1(t){
  if(!t) return "";
  const ds = t.datasets;
  const group = ds.map(d =>
    `<th colspan="2" class="${d.kind==="continuity"?"cont":""}">${d.label}<div class="stub">n=${d.n}${d.kind==="continuity"?" · 连续性":""}</div></th>`
  ).join("");
  const sub = ds.map(() => `<th>Acc</th><th>Tok</th>`).join("");
  const nMethods = t.methods.length;
  const body = t.rows.map((row, i) => {
    const showModel = i % nMethods === 0;
    const modelCell = showModel
      ? `<td class="left" rowspan="${nMethods}">${row.model}</td>`
      : "";
    const cells = ds.map(d => {
      const c = row.cells[d.id];
      return tdCell(fmtAcc(c), !!c) + tdCell(fmtTok(c), !!(c && c.tok!=null));
    }).join("");
    return `<tr class="method-${row.method_id}">${modelCell}<td class="left">${row.method}</td>${cells}</tr>`;
  }).join("");
  return `
    <h3>${t.title}</h3>
    <div class="fill-meta">已填 ${t.filled} / ${t.total} 格</div>
    <div class="caption">${t.caption}</div>
    <div class="paper-scroll"><table class="paper">
      <thead>
        <tr><th class="left" rowspan="2">模型</th><th class="left" rowspan="2">方法</th>${group}</tr>
        <tr>${sub}</tr>
      </thead>
      <tbody>${body}</tbody>
    </table></div>`;
}
function renderTable2(t){
  if(!t) return "";
  const head = t.columns.map(c => `<th>${c.label}</th>`).join("");
  const body = t.rows.map(row => {
    const cells = t.columns.map(c => tdCell("", false)).join("");
    return `<tr><td class="left">${row.method}</td>${cells}</tr>`;
  }).join("");
  return `
    <h3>${t.title}</h3>
    <div class="fill-meta">已填 ${t.filled} / ${t.total} 格</div>
    <div class="caption">${t.caption}</div>
    <div class="paper-scroll"><table class="paper">
      <thead><tr><th class="left">Method</th>${head}</tr></thead>
      <tbody>${body}</tbody>
    </table></div>`;
}
function renderTable3(t){
  if(!t) return "";
  const body = t.rows.map(row => {
    const filled = row.filled;
    const cls = filled ? "filled" : "empty";
    const accP = row.parent_acc==null ? "" : (100*row.parent_acc).toFixed(2);
    const accC = row.plus_acc==null ? "" : (100*row.plus_acc).toFixed(2);
    const dAcc = row.delta_acc_pp==null ? "" : ((row.delta_acc_pp>=0?"+":"")+row.delta_acc_pp.toFixed(2)+" pp");
    const tokP = row.parent_tok==null ? "" : Math.round(row.parent_tok).toLocaleString("en-US");
    const tokC = row.plus_tok==null ? "" : Math.round(row.plus_tok).toLocaleString("en-US");
    const sav = row.saving==null ? "" : ((row.saving>=0?"−":"+")+(100*Math.abs(row.saving)).toFixed(1)+"%");
    const tokCHtml = tokC && row.saving!=null
      ? tokC + `<div class="stub ${row.saving>=0?"save":"cost"}">${sav}</div>`
      : tokC;
    const dLat = row.delta_latency_s==null ? "" : ((row.delta_latency_s>=0?"+":"")+row.delta_latency_s.toFixed(1)+" s");
    const n = row.n ? `<div class="stub">n=${row.n}${row.planned?"":" · 已完成配对"}</div>` : "";
    return `<tr>
      <td class="left">${row.model} / ${row.dataset}${n}</td>
      <td class="left">${row.parent}</td>
      <td class="left">${row.plus}</td>
      <td class="num ${cls}">${accP}</td>
      <td class="num ${cls}">${accC}</td>
      <td class="num ${cls}">${dAcc}</td>
      <td class="num ${cls}">${tokP}</td>
      <td class="num ${cls}">${tokCHtml}</td>
      <td class="num ${cls}">${sav}</td>
      <td class="num ${cls}">${dLat}</td>
    </tr>`;
  }).join("");
  return `
    <h3>${t.title}</h3>
    <div class="fill-meta">已填 ${t.filled} / ${t.total} 行</div>
    <div class="caption">${t.caption}</div>
    <div class="paper-scroll"><table class="paper">
      <thead><tr>
        <th class="left">Model / Dataset</th><th class="left">Parent</th><th class="left">+PC</th>
        <th>Parent Acc</th><th>+PC Acc</th><th>ΔAcc</th>
        <th>Parent Tok</th><th>+PC Tok</th><th>Saving</th><th>ΔLatency</th>
      </tr></thead>
      <tbody>${body}</tbody>
    </table></div>`;
}
function renderAblation(t){
  const staticCols = t.static_cols || 0;
  const head = t.columns.map(c => `<th class="left">${c}</th>`).join("");
  const body = t.rows.map(row => {
    const detail = staticCols ? `<td class="left">${row.detail||""}</td>` : "";
    const cells = (row.cells||[]).map(() => tdCell("", false)).join("");
    return `<tr><td class="left">${row.label}</td>${detail}${cells}</tr>`;
  }).join("");
  return `
    <h3>${t.title}</h3>
    <div class="fill-meta">已填 ${t.filled} / ${t.total} 格</div>
    <div class="caption">${t.caption}</div>
    <div class="paper-scroll"><table class="paper">
      <thead><tr>${head}</tr></thead>
      <tbody>${body}</tbody>
    </table></div>`;
}
function renderPaper(paper){
  if(!paper){
    document.getElementById("paper-notes").textContent = "论文表格尚未生成。";
    return;
  }
  document.getElementById("paper-notes").innerHTML =
    (paper.notes||[]).map(x => `<div>${x}</div>`).join("");
  document.getElementById("table1").innerHTML = renderTable1(paper.table1);
  document.getElementById("table2").innerHTML = renderTable2(paper.table2);
  document.getElementById("table3").innerHTML = renderTable3(paper.table3);
  document.getElementById("ablations").innerHTML =
    (paper.ablations||[]).map(renderAblation).join("");
}

function remainClock(iso){
  const ms = new Date(iso).getTime() - Date.now();
  if(!(ms > 0)) return { overdue: true, days: 0, text: "已截止" };
  let s = Math.floor(ms / 1000);
  const days = Math.floor(s / 86400); s -= days * 86400;
  const h = Math.floor(s / 3600); s -= h * 3600;
  const m = Math.floor(s / 60); s -= m * 60;
  const pad = n => String(n).padStart(2, "0");
  return { overdue: false, days, text: pad(days) + ":" + pad(h) + ":" + pad(m) + ":" + pad(s) };
}
function toneDays(days, overdue){
  if(overdue) return "bad";
  if(days < 8) return "bad";
  if(days < 12) return "warn";
  return "ok";
}
function renderKpis(){
  if(!state) return;
  const d = state.deadlines || {};
  const abs = remainClock(d.abstract_iso);
  const pap = remainClock(d.paper_iso);
  document.getElementById("kpis").innerHTML = `
    <div class="stat"><div class="v ${toneDays(abs.days, abs.overdue)}">${abs.text}</div><div class="l">距摘要 ${d.abstract||""}</div></div>
    <div class="stat"><div class="v ${toneDays(pap.days, pap.overdue)}">${pap.text}</div><div class="l">距全文 ${d.paper||""}</div></div>
    <div class="stat"><div class="v live">${state.running.length}</div><div class="l">正在跑的 MATH</div></div>
    <div class="stat"><div class="v">${state.counts.complete}</div><div class="l">清单已完成</div></div>
    <div class="stat"><div class="v warn">${state.counts.queued}</div><div class="l">尚未开始</div></div>`;
}
function render(){
  if(!state) return;
  document.getElementById("stamp").textContent = state.generated_at;
  try { renderPaper(state.paper); } catch (err) { console.error(err); }
  renderKpis();

  const stale = state.stale[0];
  let head = "没有正在跑的 MATH 任务。";
  if(state.running.length){
    head = state.running.map(run => `
      <div style="margin:0 0 12px">
        <b class="live">进行中</b> ${run.model} · ${where(run)} · 配对进度 ${run.paired}/${run.planned}
        ${bar(run.paired, run.planned)}
        <div class="note">PID ${run.pid} · 已用 ${eta(run.elapsed_s)} · 预计剩余 ${eta(run.eta_s)}
        · 已完成题准确率 SC ${acc(run,"sc")} / PC ${acc(run,"pc")}
        ${run.last_question ? " · 最近题 "+run.last_question.id : ""}</div>
      </div>`).join("");
  } else if(stale){
    head = `<b class="bad">进程丢失</b> ${stale.tag} 标记为 running 但 PID ${stale.pid} 已不在。检查 orphan EngineCore 后决定是否续跑。`;
  }
  document.getElementById("headline").innerHTML = head;

  document.getElementById("gpus").innerHTML = state.gpus.map(g => `
    <div class="gpu">
      <div>GPU ${g.index} ${g.busy ? '<span class="pill run">占用</span>' : '<span class="pill queued">空闲</span>'}</div>
      <div class="mem">${num(g.used_mib,0)} / ${num(g.total_mib,0)} MiB · util ${num(g.util,0)}%</div>
      ${bar(g.used_mib, g.total_mib)}
    </div>`).join("") || '<div class="empty">无法读取 nvidia-smi</div>';

  if(!state.running.length && !state.processes.length){
    document.getElementById("running").innerHTML = '<div class="empty">无 run_math / run_math_deepseek / run_gsm8k / EngineCore 进程。</div>';
  } else {
    const procRows = state.processes.map(p =>
      `<tr><td class="mono">${p.pid}</td><td>${p.etime}</td><td>${p.cpu}%</td><td>${p.rss_mb} MB</td><td class="mono">${p.cmd}</td></tr>`
    ).join("");
    const runRows = state.running.map(r =>
      `<tr><td>${pill(r.status)}</td><td class="mono">${r.model}<div class="note">${r.tag}</div></td><td>${where(r)}</td>
       <td>${r.paired}/${r.planned}</td><td>SC ${acc(r,"sc")}<br>PC ${acc(r,"pc")}</td>
       <td>${eta(r.eta_s)}</td></tr>`
    ).join("");
    document.getElementById("running").innerHTML = `
      <table><thead><tr><th>状态</th><th>Run</th><th>资源</th><th>配对进度</th><th>已完成准确率</th><th>ETA</th></tr></thead>
      <tbody>${runRows || '<tr><td colspan="6" class="empty">无 MATH run.json 处于 running</td></tr>'}</tbody></table>
      <table style="margin-top:12px"><thead><tr><th>PID</th><th>已运行</th><th>CPU</th><th>RSS</th><th>命令</th></tr></thead>
      <tbody>${procRows}</tbody></table>`;
  }

  document.getElementById("tabs").innerHTML = FILTERS.map(f =>
    `<button class="${f===filter?"on":""}" data-f="${f}">${f}</button>`
  ).join("");
  document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => { filter=b.dataset.f; render(); });

  const want = {
    "全部": () => true,
    "进行中": x => x.status==="running",
    "已完成": x => x.status==="complete",
    "未开始": x => x.status==="queued",
    "部分/卡住": x => ["partial","stale","failed"].includes(x.status),
  }[filter];
  const items = state.catalog.filter(want);
  document.getElementById("catalog").innerHTML = `
    <table>
      <thead><tr><th></th><th>实验</th><th>波次 / 优先级</th><th>模型 · 数据</th><th>进度</th><th>说明</th></tr></thead>
      <tbody>${items.map(it => `
        <tr>
          <td>${pill(it.status)}</td>
          <td>${it.title}<div class="mono dim">${it.id}</div></td>
          <td>${it.wave}<div class="note">${it.priority}</div></td>
          <td>${it.model}<div class="note">${it.dataset}</div></td>
          <td>${it.progress ? it.progress.done+" / "+it.progress.total + bar(it.progress.done, it.progress.total) : "—"}</td>
          <td class="note">${it.detail||it.note||""}</td>
        </tr>`).join("")}</tbody>
    </table>`;

  document.getElementById("gsm8k").innerHTML = `
    <table>
      <thead><tr><th>方法</th><th>题数</th><th>Accuracy</th><th>Decode tok/题</th><th>相对 SC</th><th>时延 s/题</th></tr></thead>
      <tbody>${state.gsm8k.map(r => `
        <tr>
          <td>${r.label}</td><td>${r.done}/${r.total}</td>
          <td>${pct(r.accuracy)}</td><td>${num(r.avg_tokens,1)}</td>
          <td>${r.token_saving_vs_sc==null?"—":((r.token_saving_vs_sc>=0?"−":"+")+pct(Math.abs(r.token_saving_vs_sc)))}</td>
          <td>${num(r.avg_latency_s,2)}</td>
        </tr>`).join("")}</tbody>
    </table>
    <div class="note">相对 SC 的 token 列为减少量。这些是旧协议数字，不能与 MATH B=16/32 混算节省率。</div>`;

  const mathRows = state.math.map(r => `
    <tr>
      <td>${pill(r.status)}</td>
      <td><span class="pill">${r.kind}</span></td>
      <td class="mono">${r.fingerprint}<div class="note">${r.model_type||""} · GPU ${r.gpus||"—"} · B=${r.max_branches} C=${r.beta_threshold} max_tok=${r.max_tokens}</div></td>
      <td>${r.paired}/${r.planned}</td>
      <td>${pct(r.sc_accuracy,0)} (${r.sc_correct??"?"}/${r.sc_questions??"?"}) / ${pct(r.pc_accuracy,0)} (${r.pc_correct??"?"}/${r.pc_questions??"?"})</td>
      <td>${r.decode_saving==null?"—":pct(r.decode_saving)}</td>
      <td>${r.truncations_sc??"—"}</td>
    </tr>`).join("");
  document.getElementById("math").innerHTML = `<table>
    <thead><tr><th>状态</th><th>类型</th><th>Run</th><th>配对</th><th>已完成准确率 SC / PC</th><th>Decode 节省</th><th>SC 截断</th></tr></thead>
    <tbody>${mathRows}</tbody></table>`;

  document.getElementById("datasets").innerHTML = `<table>
    <thead><tr><th>数据集</th><th>状态</th><th>行数</th><th>路径</th></tr></thead>
    <tbody>${state.datasets.map(r => `
      <tr><td>${r.name}</td><td>${pill(r.status==="ready"?"complete":"queued")}</td>
      <td>${r.rows||0}</td><td class="mono">${r.path}</td></tr>`).join("")}</tbody></table>`;
}

async function tick(){
  try {
    const res = await fetch("/api/state", {cache:"no-store"});
    state = await res.json();
    render();
  } catch(e) {
    document.getElementById("stamp").textContent = "无法刷新 /api/state";
  }
}
tick();
setInterval(tick, 8000);
setInterval(renderKpis, 1000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        stamp = now_cst().strftime("%H:%M:%S")
        sys_stderr = __import__("sys").stderr
        sys_stderr.write(f"[{stamp}] " + (fmt % args) + "\n")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            return
        if path == "/api/state":
            payload = json.dumps(collect_state(), ensure_ascii=False, default=str)
            self._send(200, payload.encode(), "application/json; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"Experiment console at http://{args.host}:{args.port}/  ({iso_cst(now_cst())})",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)


if __name__ == "__main__":
    main()
