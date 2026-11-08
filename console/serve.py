#!/usr/bin/env python3
"""Local experiment console for the Path-Consistency ICLR 2027 project.

    /home/26xyw/.conda/envs/PC/bin/python console/serve.py --host 0.0.0.0 --port 28787
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
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
CPT_DIR = Path("/home/26xyw/data/path-consistency-results/deepconf-cpt")
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
        "note": "已查看，不得用于选阈或 PVC；主表排除这 10 题。",
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
        "note": "完整新协议：配对根、BLC、trial/commit/rollback。主表默认 C=0.8。",
        "match": "gsm8k_llama31_c08",
    },
    {
        "id": "llama31-gsm8k-cpt",
        "wave": "9/13 Table 2",
        "priority": "P0",
        "title": "Llama-3.1-8B GSM8K 官方 CPT（N=32）",
        "model": "Llama-3.1-8B-Instruct",
        "dataset": "GSM8K test · 1319",
        "methods": "CPT qwen/math.py N=32",
        "note": "GPU 4。N=32 对齐 PC B=32 / PoLR N=32。1319 已完成，已进 Table 2。N=20 不进本表。Llama 无 thinking；不是 Qwen 公平 CPT。",
        "match": "cpt_llama31_gsm8k",
    },
    {
        "id": "llama31-gsm8k-polr",
        "wave": "9/12 Table 2",
        "priority": "P0",
        "title": "Llama-3.1-8B GSM8K PoLR（TF-IDF）",
        "model": "Llama-3.1-8B-Instruct",
        "dataset": "GSM8K test · 1319",
        "methods": "PoLR N=32 L_p=64",
        "note": "GPU 2。L_p=64：Llama GSM8K SC 轨迹 mean 260 / p10 160，64 是真前缀。1319 完成，已进 Table 2。",
        "match": "polr_llama31_gsm8k",
    },
    {
        "id": "llama31-gsm8k-polr-lp256",
        "wave": "9/13 论文默认 L_p",
        "priority": "P0",
        "title": "Llama-3.1-8B GSM8K PoLR（L_p=256）",
        "model": "Llama-3.1-8B-Instruct",
        "dataset": "GSM8K test · 1319",
        "methods": "PoLR N=32 L_p=256",
        "note": "论文默认前缀长度。本机 SC 轨迹 mean 260，256 对多数题接近全文。1319 已完成，已进 Table 2。主对照仍是 L_p=64。",
        "match": "polr_llama31_gsm8k_lp256",
    },
    {
        "id": "llama31-gsm8k-polr-lp128",
        "wave": "9/14 L_p 插值",
        "priority": "P0",
        "title": "Llama-3.1-8B GSM8K PoLR（L_p=128）",
        "model": "Llama-3.1-8B-Instruct",
        "dataset": "GSM8K test · 1319",
        "methods": "PoLR N=32 L_p=128",
        "note": "插在 L_p=64 与 256 之间。本机 SC mean 260 / p10 160，128 仍是真前缀。N=32。1319 已完成，已进 Table 2。",
        "match": "polr_llama31_gsm8k_lp128",
    },
    {
        "id": "llama31-gsm8k-ac",
        "wave": "9/13 Table 2 组合",
        "priority": "P0",
        "title": "Llama-3.1-8B GSM8K AC 与 AC+PC",
        "model": "Llama-3.1-8B-Instruct",
        "dataset": "GSM8K test · 1319",
        "methods": "AC τ=0.95 · AC+PC B=32 m=4 C=0.8",
        "note": "GPU 1。新协议截断组合，不是旧 Sentence-PC B=20。AC 停止阈值 0.95，PC 前缀门控仍是 C=0.8。1319 已完成，已进 Table 2。",
        "match": "gsm8k_llama31_ac",
    },
    {
        "id": "llama31-gsm8k-esc",
        "wave": "9/13 Table 2 组合",
        "priority": "P0",
        "title": "Llama-3.1-8B GSM8K ESC 与 ESC+PC",
        "model": "Llama-3.1-8B-Instruct",
        "dataset": "GSM8K test · 1319",
        "methods": "ESC · ESC+PC B=32 m=4 C=0.8",
        "note": "GPU 3。窗口 4 条答案全同且可解析则停。配对根复用 ESC 无前缀窗口。1319 已完成，已进 Table 2。",
        "match": "gsm8k_llama31_esc",
    },
    {
        "id": "llama31-gsm8k-ac099-c00",
        "wave": "9/14 诊断",
        "priority": "P1",
        "title": "Llama-3.1-8B GSM8K AC τ=0.99 + PC C=0",
        "model": "Llama-3.1-8B-Instruct",
        "dataset": "GSM8K test · 1319",
        "methods": "AC τ=0.99 · AC+PC C=0",
        "note": "诊断：4/4 Beta≈0.969 不再早停。PC C=0。1319 已完成，已进 Table 2 组合块（不覆盖 τ=0.95）。",
        "match": "gsm8k_llama31_ac099",
    },
    {
        "id": "llama31-gsm8k-esc2w8",
        "wave": "9/14 诊断",
        "priority": "P1",
        "title": "Llama-3.1-8B GSM8K ESC 延迟停（两窗或满 8）",
        "model": "Llama-3.1-8B-Instruct",
        "dataset": "GSM8K test · 1319",
        "methods": "ESC two_window_or_8 · ESC+PC C=0.8",
        "note": "诊断：连续两窗全同且同答案，或累计 ≥8 条相同。1319 已完成，已进 Table 2 组合块（Table 2 默认仍是单窗 4 条）。",
        "match": "gsm8k_llama31_esc2w8",
    },
    {
        "id": "llama31-gsm8k-pc-c00",
        "wave": "9/10 消融 A",
        "priority": "P0-C",
        "title": "Llama-3.1-8B GSM8K PC C=0 消融",
        "model": "Llama-3.1-8B-Instruct",
        "dataset": "GSM8K test · 1319",
        "methods": "Pass@1 · SC · PC B=32 m=4 C=0",
        "note": "纯多数晋升对照，不进 Table 1，不是 C*。GPU 3/4，2 shard。采样对齐 C=0.8 GSM8K（mt=2048）。",
        "match": "gsm8k_llama31_c00",
    },
    {
        "id": "deepseek-gsm8k-first100",
        "wave": "9/9 DeepSeek",
        "priority": "P0",
        "title": "DeepSeek-V4-Flash × GSM8K 前 100 题 SC/PC",
        "model": "deepseek-v4-flash",
        "dataset": "GSM8K test · source_row 0–99",
        "methods": "Pass@1 · SC · PC B=32 m=4 C=0.8",
        "note": "JSONL 前 100 行，不是 sample_seed 随机子集。thinking on，默认 C=0.8。不进 Table 1 全量格。",
        "match": "deepseek_gsm8k_first100",
    },
    {
        "id": "deepseek-gsm8k-next500",
        "wave": "9/9 DeepSeek",
        "priority": "P0",
        "title": "DeepSeek-V4-Flash × GSM8K 随后 500 题 SC/PC",
        "model": "deepseek-v4-flash",
        "dataset": "GSM8K test · source_row 100–599",
        "methods": "Pass@1 · SC · PC B=32 m=4 C=0.8",
        "note": "紧接前 100 题的下 500 行。同一协议。不进 Table 1 全量格。",
        "match": "deepseek_gsm8k_next500",
    },
    {
        "id": "deepseek-gsm8k-rest719",
        "wave": "9/10 DeepSeek",
        "priority": "P0",
        "title": "DeepSeek-V4-Flash × GSM8K 剩余 719 题 SC/PC",
        "model": "deepseek-v4-flash",
        "dataset": "GSM8K test · source_row 600–1318",
        "methods": "Pass@1 · SC · PC B=32 m=4 C=0.8",
        "note": "补完 0280 之后启动。结束后 merge 成 1319。",
        "match": "deepseek_gsm8k_rest719",
    },
    {
        "id": "deepseek-gsm8k-full1319",
        "wave": "9/10 DeepSeek",
        "priority": "P0",
        "title": "DeepSeek-V4-Flash × GSM8K 1319 合并",
        "model": "deepseek-v4-flash",
        "dataset": "GSM8K test · 1319",
        "methods": "merged SC/PC",
        "note": "first100 + next500 + rest719。满 1319。Table 1 Acc 用本合并 overlay；PC Tok 见 vote-v2 融合。",
        "match": "deepseek_gsm8k_full1319",
    },
    {
        "id": "deepseek-gsm8k-pc-vote-v2",
        "wave": "9/12 PC 复跑",
        "priority": "P0",
        "title": "DeepSeek-V4-Flash × GSM8K PC C=0.8（投票键对齐复判）",
        "model": "deepseek-v4-flash",
        "dataset": "GSM8K test · 1319",
        "methods": "PC B=32 m=4 C=0.8 · numeric vote keys",
        "note": "已融合进 Table 1 / Ablation A：Acc 仍用原分片 overlay 1277/1319；PC Tok / 时延用本 run（decode −34.2%）。",
        "match": "deepseek_gsm8k_pc_vote_v2",
    },
    {
        "id": "deepseek-gsm8k-pc-c00",
        "wave": "9/10 消融 A",
        "priority": "P0-C",
        "title": "DeepSeek-V4-Flash × GSM8K PC C=0 消融",
        "model": "deepseek-v4-flash",
        "dataset": "GSM8K test · 1319",
        "methods": "Pass@1 · SC · PC B=32 m=4 C=0",
        "note": "Ablation A 纯多数晋升对照。不进 Table 1，不是 C*。",
        "match": "deepseek_gsm8k_c00",
    },
    {
        "id": "deepseek-gsm8k-pc-c07",
        "wave": "9/12 消融 A",
        "priority": "P0-C",
        "title": "DeepSeek-V4-Flash × GSM8K SC/PC C=0.7 消融",
        "model": "deepseek-v4-flash",
        "dataset": "GSM8K test · 1319",
        "methods": "Pass@1 · SC · PC B=32 m=4 C=0.7",
        "note": "Ablation A。配对 sc,pc，1319 完成。不进 Table 1，不是 C*。",
        "match": "deepseek_gsm8k_c07",
    },
    {
        "id": "data-aime-gpqa-devmix",
        "wave": "9/6–9/8 数据",
        "priority": "P0",
        "title": "准备 AIME24/25、GPQA-D JSONL",
        "model": "—",
        "dataset": "AIME24 · AIME25 · GPQA-D",
        "methods": "数据转换",
        "note": "CUDA 前转 JSONL。Dev-Mix 不再用于选 C*，不阻塞。GPQA 需用户接受 gated 条款。",
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
        "id": "protocol-v4",
        "wave": "9/9 冻结",
        "priority": "P0",
        "title": "协议 v4：默认 C=0.8，取消 C* 与 PVC 查找表",
        "model": "—",
        "dataset": "—",
        "methods": "决策",
        "note": "主表 PC (C=0.8)；sweep 只进消融；长度固定 20% 网格。见 plan v4 / 报告 §12.14。",
        "match": "protocol_v4",
    },
    {
        "id": "pvc-pilot",
        "wave": "可选分析",
        "priority": "P1",
        "title": "干预式 Prefix Value Curve（可选分析）",
        "model": "Llama-3.1 先，Qwen 后",
        "dataset": "子集即可",
        "methods": "r∈{0,0.2,0.4,0.6,0.8}",
        "note": "只做分析图，不做测试时查找表控制器（2026-09-09 已取消查找表）。",
        "match": "none",
    },
    {
        "id": "devmix-cstar",
        "wave": "已取消",
        "priority": "—",
        "title": "Dev-Mix 选 C*（已取消）",
        "model": "—",
        "dataset": "—",
        "methods": "—",
        "note": "2026-09-09：C=0.8 为默认操作点；不再用 Dev-Mix 选 C*。",
        "match": "cancelled",
    },
    {
        "id": "selector-ablation-pilot",
        "wave": "9/14–9/17 消融",
        "priority": "P0-C",
        "title": "BLC vs random / medoid / mean-logprob",
        "model": "Llama-3.1 或 Qwen",
        "dataset": "50–100 题",
        "methods": "selector 消融",
        "note": "不再作为冻结 PC 的前提；representative subset 即可。",
        "match": "none",
    },
    {
        "id": "p0a-llama-math500",
        "wave": "9/9 主表",
        "priority": "P0-A",
        "title": "Llama-3.1-8B × MATH-500 全量 SC/PC",
        "model": "Llama-3.1-8B-Instruct",
        "dataset": "MATH-500（Acc：490 raw + 10 记对 / 500）",
        "methods": "Pass@1 · SC · PC B=32 m=4 C=0.8",
        "note": "V1 CUDA-graph TP=1 GPU1；max_tokens=8192。默认 C=0.8。",
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
        "note": "V1 CUDA-graph TP=1 GPU3 顺序跑两年。默认 C=0.8。",
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
        "note": "V1 CUDA-graph TP=1 GPU2。默认 C=0.8。",
        "match": "llama31_gpqa",
    },
    {
        "id": "p0a-qwen-math500",
        "wave": "9/8 P0-A",
        "priority": "P0-A",
        "title": "Qwen3-30B-A3B × MATH-500 490 SC/PC",
        "model": "Qwen3-30B-A3B",
        "dataset": "MATH-500（Acc：490 raw + 10 记对 / 500）",
        "methods": "Pass@1 · SC · PC B=32 m=4 C=0.8",
        "note": "V1 CUDA-graph TP=4 + CUDA-12.2 NCCL；thinking on；物理窗口 4。默认 C=0.8。",
        "match": "qwen30_a3b_math500",
    },
    {
        "id": "qwen30-a3b-math500-first100",
        "wave": "9/19 试跑",
        "priority": "P0-A",
        "title": "Qwen3-30B-A3B × MATH-500 前 100 题 SC/PC",
        "model": "Qwen3-30B-A3B",
        "dataset": "MATH-500 JSONL source_row 0–99（无工程排除）",
        "methods": "SC · PC B=32 m=4 C=0.8 thinking on",
        "note": "2026-09-20 complete。SC=PC 94/100。与 next100 合并后 Table 1 部分格 n=199。",
        "match": "qwen30_a3b_math500_first100",
    },
    {
        "id": "qwen30-a3b-math500-next100",
        "wave": "9/21 分片",
        "priority": "P0-A",
        "title": "Qwen3-30B-A3B × MATH-500 第 101–200 题 SC/PC",
        "model": "Qwen3-30B-A3B",
        "dataset": "MATH-500 JSONL source_row 100–199（排除工程行 151 → planned 99）",
        "methods": "SC · PC B=32 m=4 C=0.8 thinking on",
        "note": "2026-09-22 complete。SC=PC 94/99；decode −42.0%。与 first100 合并后 Table 1 部分格 n=199。",
        "match": "qwen30_a3b_math500_next100",
    },
    {
        "id": "p0a-qwen-aime-gpqa",
        "wave": "9/10–9/14 主表",
        "priority": "P0-A",
        "title": "Qwen3-30B-A3B × AIME/GPQA SC/PC",
        "model": "Qwen3-30B-A3B",
        "dataset": "AIME24 · AIME25 · GPQA-D",
        "methods": "Pass@1 · SC · PC",
        "note": "MATH-500 之后。本机 JSONL 已有；新服务器需同样路径。默认 C=0.8，thinking on。",
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
        "note": "MATH-500 490 已完成。AIME/GPQA 为独立 tag。默认 C=0.8。",
        "match": "deepseek_math500",
    },
    {
        "id": "deepseek-math500-selrec-b",
        "wave": "9/16 消融",
        "priority": "P1",
        "title": "DeepSeek MATH-500 选择×恢复 臂 B（uniform + recovery 开）",
        "model": "deepseek-v4-flash",
        "dataset": "MATH-500（Acc：490 raw + 10 记对 / 500）",
        "methods": "PC uniform C=0.8 recovery on",
        "note": "已进消融表 Ablation C/E。raw PC 426+10=436/500。复用主表 SC 根。不进 Table 1。",
        "match": "deepseek_selrec_rand_on",
    },
    {
        "id": "deepseek-math500-selrec-c",
        "wave": "9/16 消融",
        "priority": "P1",
        "title": "DeepSeek MATH-500 选择×恢复 臂 C（BLC + recovery 关）",
        "model": "deepseek-v4-flash",
        "dataset": "MATH-500（Acc：490 raw + 10 记对 / 500）",
        "methods": "PC BLC C=0.8 recovery off",
        "note": "已进消融表 Ablation C/E。raw PC 423+10=433/500。失败前缀保留。不进 Table 1。",
        "match": "deepseek_selrec_blc_off",
    },
    {
        "id": "deepseek-math500-selrec-d",
        "wave": "9/16 消融",
        "priority": "P1",
        "title": "DeepSeek MATH-500 选择×恢复 臂 D（uniform + recovery 关）",
        "model": "deepseek-v4-flash",
        "dataset": "MATH-500（Acc：490 raw + 10 记对 / 500）",
        "methods": "PC uniform C=0.8 recovery off",
        "note": "已进消融表 Ablation C/E。raw PC 423+10=433/500。uniform + recovery 关。不进 Table 1。",
        "match": "deepseek_selrec_rand_off",
    },
    {
        "id": "deepseek-math500-lenint",
        "wave": "9/17 消融",
        "priority": "P1",
        "title": "DeepSeek MATH-500 matched-history 长度干预",
        "model": "deepseek-v4-flash",
        "dataset": "MATH-500 490（排除工程 10 题；common-support）",
        "methods": "r∈{0,0.2,0.4,0.6,0.8} × R=8；冻 window-0 BLC source；尾巴无前缀 SC cache",
        "note": "Ablation G 已填。466×5×R=8 complete。不是 PVC。不进 Table 1。",
        "match": "deepseek_lenint",
    },
    {
        "id": "p0a-deepseek-aime",
        "wave": "9/8–9/10 主表",
        "priority": "P0-A",
        "title": "DeepSeek-V4-Flash × AIME24/25 SC/PC",
        "model": "deepseek-v4-flash",
        "dataset": "AIME 2024 · AIME 2025",
        "methods": "Pass@1 · SC · PC B=32 m=4 C=0.8",
        "note": "配对根 v2；BLC；整数 0–999。默认 C=0.8。",
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
        "note": "选项 seed=2027 写入 JSONL。默认 C=0.8。",
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
        "id": "p0c-threshold-sweep",
        "wave": "9/14–9/17 消融",
        "priority": "P0-C",
        "title": "Beta 阈值 sweep（消融 A）",
        "model": "Llama-3.1-8B 或 Qwen A3B",
        "dataset": "GSM8K 或 representative subset",
        "methods": "C∈{0, 0.7, 0.8, 0.9, 0.95, 1.0}",
        "note": "只进消融，不从测试集挑最优 C。C=0.8 与 SC 已有；C=0 在 GPU 3/4 跑。{0.7,0.9,0.95} 仍缺。",
        "match": "none",
    },
    {
        "id": "p0c-ablations",
        "wave": "9/14–9/17 消融",
        "priority": "P0-C",
        "title": "Beta 阈值 sweep / 长度 / selector / rollback 消融",
        "model": "Llama-3.1-8B-Instruct",
        "dataset": "GSM8K test · 1319",
        "methods": "Ablation A–F",
        "note": "C=0.8 默认操作点。Llama 主方法 / SC 已填入。DeepSeek GSM8K C=0.8/C=0/C=0.7 已填。DeepSeek MATH-500 选择×recovery A/B/C/D 已替换 Ablation C/E。{0.9,0.95} 与 PVC 查找表不做。",
        "match": "ablations_llama31_gsm8k",
    },
    {
        "id": "p0d-repeats",
        "wave": "9/14–9/17 统计",
        "priority": "P0-D",
        "title": "主表核心行 seed/repetition 1/2 与 CI",
        "model": "主方法",
        "dataset": "主表行",
        "methods": "重复",
        "note": "9/16 冻结数字。不靠测试集挑 C 补救主表。",
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
        "note": "只写已冻结可复算数字。摘要 9/18 23:59 AOE。不要写 PC(C*) 或 adaptive PVC。",
        "match": "none",
    },
]


REBUTTAL = {
    "updated": "2026-11-08",
    "source": "handoff_rebuttal.md",
    "headline": (
        "四项补实验按显存分机器。Qwen3-30B-A3B 的 BF16 权重大约 60GB，"
        "只在 4 张 A40 上跑。C=0.8 是先验操作点，不从这些重复里重选 C，"
        "也不把结果最好的一次换进表。"
    ),
    "protocol": "B=32，m=4，Beta top-1 vs all-rest，thinking on。实验 1 报墙钟和吞吐；其余时间只报 generation_seconds。",
    "platforms": [
        {
            "name": "A40 ×4",
            "spec": "每张 48GB",
            "role": "全部 Qwen3-30B-A3B",
            "lines": [
                "卡 0、1 · TP=2：实验 1，32 路墙钟。QWEN_GPU_BLOCKS 留空。",
                "卡 2、3 · TP=2：实验 2，然后 Qwen AIME 的 seed 1/2，再跑 50 题搜索。",
                "不要把四张卡合成 TP=4 来做实验 1，否则 batch 32 会放得太松。",
            ],
        },
        {
            "name": "5090 ×1",
            "spec": "32GB",
            "role": "只跑 Llama-3.1-8B",
            "lines": [
                "实验 4：GSM8K 1319，ToT、beam、MCTS 依次跑。",
                "预算用已有 PC 的逐题生成 token，PC 本身不重跑。",
                "这张卡装不下 A3B，不跑 Qwen。",
            ],
        },
        {
            "name": "4090 实验室 ×5",
            "spec": "每张 24GB · 10.10.60.106",
            "role": "DeepSeek API，不占卡",
            "lines": [
                "实验 3：AIME 2024/2025 与 MATH-500 的 rep1、rep2。",
                "单张或两张 4090 都装不下 A3B，本机不启动 Qwen。",
                "已有 JSONL 的翻转和多样性统计也在这里做。",
            ],
        },
    ],
    "groups": [
        {
            "id": "1",
            "title": "实验 1 · 显存受限的 32 路墙钟",
            "reviews": "R1-W2 · R1-Q1 · R3-W2",
            "platform": "A40 卡 0、1 · TP=2 · BF16",
            "model": "Qwen3-30B-A3B · thinking on",
            "data": "MATH-500 source_row 0–31，32 题",
            "note": "SC 用 max_num_seqs=32，不按窗口串行。PC 为 8 窗 × 4。长序列 OOM 时记下峰值显存和实际驻留 batch，墙钟按真实调度。回复写明是两张 A40 的显存受限结果。",
            "runs": [
                {"what": "SC，一次批 32 条", "tag": "latency_qwen3_30b_a3b_math500_n32_sc_batch32_tp2_a40"},
                {"what": "PC，8 窗 × 4", "tag": "latency_qwen3_30b_a3b_math500_n32_pc_serial8x4_tp2_a40"},
            ],
        },
        {
            "id": "2",
            "title": "实验 2 · 长链上的 DeepConf、CPT、PoLR",
            "reviews": "R1-W1 · R1-Q3 · R2-W1",
            "platform": "A40 卡 2、3 · TP=2",
            "model": "Qwen3-30B-A3B · thinking on · C=0.8",
            "data": "AIME 2024、AIME 2025、MATH-500 490",
            "note": "不用 Llama 上的 CPT adapter。DeepConf 只做 offline（N_init=16，B=32）。PoLR 接不上 thinking 就停，N=32，L_p=128。SC/PC 已有完整 JSONL 的不重跑。",
            "runs": [
                {"what": "AIME 2024 · SC/PC", "tag": "aime_2024_qwen3_30b_a3b_b32_c08_seed0"},
                {"what": "AIME 2024 · DeepConf-offline", "tag": "aime_2024_qwen3_30b_a3b_deepconf_offline_b32"},
                {"what": "AIME 2024 · DeepConf+PC", "tag": "aime_2024_qwen3_30b_a3b_deepconf_pc_b32"},
                {"what": "AIME 2024 · 官方 CPT", "tag": "aime_2024_qwen3_30b_a3b_cpt_official_n32"},
                {"what": "AIME 2024 · PoLR，能接才跑", "tag": "aime_2024_qwen3_30b_a3b_polr_n32_lp128"},
                {"what": "AIME 2025 · SC/PC", "tag": "aime_2025_qwen3_30b_a3b_b32_c08_seed0"},
                {"what": "AIME 2025 · DeepConf-offline", "tag": "aime_2025_qwen3_30b_a3b_deepconf_offline_b32"},
                {"what": "AIME 2025 · DeepConf+PC", "tag": "aime_2025_qwen3_30b_a3b_deepconf_pc_b32"},
                {"what": "AIME 2025 · 官方 CPT", "tag": "aime_2025_qwen3_30b_a3b_cpt_official_n32"},
                {"what": "AIME 2025 · PoLR，能接才跑", "tag": "aime_2025_qwen3_30b_a3b_polr_n32_lp128"},
                {"what": "MATH-500 · SC/PC，已有分片不删", "tag": "math500_qwen3_30b_a3b_b32_mt32768_c08_seed0"},
                {"what": "MATH-500 · DeepConf-offline", "tag": "math500_qwen3_30b_a3b_deepconf_offline_b32"},
                {"what": "MATH-500 · DeepConf+PC", "tag": "math500_qwen3_30b_a3b_deepconf_pc_b32"},
                {"what": "MATH-500 · 官方 CPT", "tag": "math500_qwen3_30b_a3b_cpt_official_n32"},
                {"what": "MATH-500 · PoLR，能接才跑", "tag": "math500_qwen3_30b_a3b_polr_n32_lp128"},
            ],
        },
        {
            "id": "3",
            "title": "实验 3 · 三次重复，报 mean±std",
            "reviews": "R1-W3 · R1-Q2",
            "platform": "DeepSeek 在 4090 实验室，不占卡；Qwen 在 A40 卡 2、3",
            "model": "DeepSeek-V4-Flash reasoning_effort=high；Qwen3-30B-A3B",
            "data": "两边都做 AIME 2024/2025。DeepSeek 再加 MATH-500。Qwen 的 MATH-500、GSM8K、GPQA 不做 3 次",
            "note": "论文那一次保留。再跑 2 个独立重复。DeepSeek 没有可依赖的 seed，rep1/rep2 是新请求，PC 复用这一次的 SC root。Qwen 的 seed 1/2 才改变抽样。不挑最好的一次。",
            "runs": [
                {"what": "DeepSeek AIME 2024 · rep1 / rep2", "tag": "aime_2024_deepseek_v4_flash_b32_c08_rep1 · rep2"},
                {"what": "DeepSeek AIME 2025 · rep1 / rep2", "tag": "aime_2025_deepseek_v4_flash_b32_c08_rep1 · rep2"},
                {"what": "DeepSeek MATH-500 · rep1 / rep2", "tag": "math500_deepseek_v4_flash_b32_c08_rep1 · rep2"},
                {"what": "Qwen AIME 2024 · seed1 / seed2", "tag": "aime_2024_qwen3_30b_a3b_b32_c08_seed1 · seed2"},
                {"what": "Qwen AIME 2025 · seed1 / seed2", "tag": "aime_2025_qwen3_30b_a3b_b32_c08_seed1 · seed2"},
            ],
        },
        {
            "id": "4",
            "title": "实验 4 · 相同 token 的 ToT、beam、MCTS",
            "reviews": "R3-W1 · R3-Q1",
            "platform": "GSM8K 在单张 5090；MATH-500 的 50 题在 A40 卡 2、3",
            "model": "Llama-3.1-8B-Instruct；Qwen3-30B-A3B",
            "data": "GSM8K 1319 全量；MATH-500 source_row 0–49",
            "note": "仓库还没有这三个入口，先写脚本。每题生成 token 对齐该题 PC。搜索更高只说明任务不同，不写成 PC 失败，也不写成 PC 赢过搜索。",
            "runs": [
                {"what": "Llama GSM8K · ToT", "tag": "gsm8k_llama31_tot_matched_pc_seed0"},
                {"what": "Llama GSM8K · step beam", "tag": "gsm8k_llama31_beam_matched_pc_seed0"},
                {"what": "Llama GSM8K · rollout MCTS", "tag": "gsm8k_llama31_mcts_matched_pc_seed0"},
                {"what": "Qwen 50 题 · PC，没有才跑", "tag": "math500_qwen3_30b_a3b_totcmp_n50_b32_c08_seed0"},
                {"what": "Qwen 50 题 · ToT", "tag": "math500_qwen3_30b_a3b_tot_matched_pc_n50_seed0"},
                {"what": "Qwen 50 题 · step beam", "tag": "math500_qwen3_30b_a3b_beam_matched_pc_n50_seed0"},
                {"what": "Qwen 50 题 · rollout MCTS", "tag": "math500_qwen3_30b_a3b_mcts_matched_pc_n50_seed0"},
            ],
        },
    ],
    "skip": [
        "R1-Q4 配对翻转用已有 JSONL，不新跑。DeepSeek MATH-500 是 2 题由对变错、0 题由错变对；AIME 与 GPQA 翻转数为 0。",
        "R2-W2 / R2-Q1 用已有轨迹算 unique answers。DeepSeek AIME 2025 上 SC 比 Pass@1 高 13.3 个百分点，PC 与 SC 同为 100%。",
        "AC / ESC 已在 Llama/GSM8K 做过，不单开。",
        "Qwen MATH-500 已完成分片不重跑：630825247308（0–99）与 664997212910（100–199，跳过 151）。",
        "DeepSeek 已发表的 rep0 不重跑：e4cd6dcd6672、771c0b2351c6、4856d4ef5673。",
    ],
}


PLAN = {
    "version": "v4",
    "frozen_at": "2026-09-09",
    "source": "paper/plan_2weeks_ICLR2027.md · report §12.14",
    "headline": (
        "主表一律 PC (C=0.8)：先验默认操作点，不是 Dev-Mix C*。"
        "长度固定 20% 网格（gate 通过则前进一档）。"
        "不做 PVC 查找表控制器；干预式 PVC 只当分析图。"
    ),
    "decisions": [
        {
            "id": "threshold",
            "title": "阈值 C=0.8",
            "status": "frozen",
            "text": "沿用原论文 / Adaptive-Consistency。禁止从任何测试集回写「最优 C」。已完成的 C=0.8 配对不必重跑。",
        },
        {
            "id": "sweep",
            "title": "阈值 sweep",
            "status": "ablation",
            "text": "{0.7, 0.8, 0.9, 0.95, 1.0} 只进消融。更低 C → 更大加速、更高错前缀风险；更高 C → 更接近 SC、更安全。C=1 即 SC。",
        },
        {
            "id": "length",
            "title": "长度策略",
            "status": "frozen",
            "text": "Relative-PC：20%→40%→60%→80%。gate 通过则前进一档。不要写成 adaptive PVC。",
        },
        {
            "id": "core",
            "title": "算法核心",
            "status": "frozen",
            "text": "Beta top-1 vs all-rest；每窗口一条共享前缀；BLC；trial/commit；单层回退；两击重置。B=32，m=4，paired-root v2。",
        },
        {
            "id": "pvc",
            "title": "PVC",
            "status": "optional",
            "text": "查找表控制器已取消。干预式曲线若做只进分析图，不是方法主贡献，也不作为 go/no-go。",
        },
        {
            "id": "devmix",
            "title": "Dev-Mix",
            "status": "cancelled",
            "text": "不再用来选 C* 或建查找表。MATH-500 工程 10 题仍排除出主统计。",
        },
    ],
    "go_nogo": [
        {"must": True, "text": "PC 相对同预算 SC：至少两模型、两现代数据集准确率非劣（ε=1.0 pp），并有稳定 token 或 latency 收益。"},
        {"must": True, "text": "DeepConf+PC 相对同一 DeepConf parent：至少在 Qwen3-30B-A3B 的两个设置上有增量。"},
        {"must": True, "text": "matched-budget / matched-latency 下 PC 不劣于最接近的 CPT。"},
        {"must": True, "text": "代码、公式、门控、回退和表格口径完全一致。speedup 只报 generation_seconds（含复用 root）。"},
        {"must": False, "text": "干预式 PVC 分析图与 Dev-Mix 选 C*：不再作为投稿 go/no-go。"},
    ],
    "schedule": [
        {
            "when": "9/6–9/8",
            "status": "done",
            "title": "协议与接口",
            "items": "新协议 runner、Llama GSM8K 1319、DeepSeek MATH-500 490、AIME/GPQA JSONL。",
        },
        {
            "when": "9/9",
            "status": "running",
            "title": "本机主表",
            "items": "Llama MATH-500 490 / GPQA-D / AIME24→25（GPU 1–3，不要杀）。DeepSeek AIME/GPQA API。",
        },
        {
            "when": "立即",
            "status": "partial",
            "title": "新服务器 Qwen",
            "items": "4090 first100（source_row 0–99）已完成，不进 Table 1。全量 490 仍走新服务器；不要把 n=100 填进主表。",
        },
        {
            "when": "9/10–9/14",
            "status": "queued",
            "title": "P0-B 现代对照",
            "items": "A3B 上 DeepConf、DeepConf+PC、官方 CPT。不可用 Llama CPT adapter 填主表。",
        },
        {
            "when": "9/14–9/17",
            "status": "partial",
            "title": "消融与重复",
            "items": "阈值 {0.9,0.95} 仍空。DeepSeek MATH-500 选择×recovery A/B/C/D 已替换 Ablation C/E。主表 seed 1/2。PVC 查找表不做。",
        },
        {
            "when": "9/15–9/18",
            "status": "queued",
            "title": "摘要",
            "items": "只写已冻结可复算数字。不要写 PC(C*) 或 adaptive PVC。摘要 9/18 23:59 AOE。",
        },
        {
            "when": "9/19–9/25",
            "status": "queued",
            "title": "全文缓冲",
            "items": "P1、limitations、模板、审计。全文 9/25 23:59 AOE。",
        },
    ],
    "cancelled": [
        "Dev-Mix 选 C*",
        "PVC 查找表控制器（测试时 adaptive PVC）",
        "从测试集挑最优 C",
        "Qwen3-32B 主表 / 续跑 paused eager 490",
    ],
}


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


def _pct_acc(label: str, acc, correct, questions) -> str | None:
    if acc is None:
        return None
    if correct is not None and questions:
        return f"{label} {100 * acc:.0f}% ({correct}/{questions})"
    return f"{label} {100 * acc:.0f}%"


def _first_prediction(last_pred: dict | None):
    if not last_pred:
        return None
    for key in ("AC", "AC+PC", "ESC", "ESC+PC", "SC", "PC", "PoLR", "CPT"):
        if last_pred.get(key):
            return last_pred[key]
    return next(iter(last_pred.values()), None)


def _accuracy_summary(
    family: str,
    *,
    sc_method: dict | None = None,
    pc_method: dict | None = None,
    polr_method: dict | None = None,
    ac_method: dict | None = None,
    ac_pc_method: dict | None = None,
    esc_method: dict | None = None,
    esc_pc_method: dict | None = None,
    cpt_acc=None,
    cpt_correct=None,
    cpt_n=None,
) -> str:
    sc_method = sc_method or {}
    pc_method = pc_method or {}
    polr_method = polr_method or {}
    ac_method = ac_method or {}
    ac_pc_method = ac_pc_method or {}
    esc_method = esc_method or {}
    esc_pc_method = esc_pc_method or {}
    if family == "polr":
        return (
            _pct_acc(
                "PoLR",
                polr_method.get("accuracy"),
                polr_method.get("correct"),
                polr_method.get("questions"),
            )
            or "—"
        )
    if family == "cpt":
        return _pct_acc("CPT", cpt_acc, cpt_correct, cpt_n) or "—"
    if family == "acesc":
        bits = [
            _pct_acc(
                "AC",
                ac_method.get("accuracy"),
                ac_method.get("correct"),
                ac_method.get("questions"),
            ),
            _pct_acc(
                "AC+PC",
                ac_pc_method.get("accuracy"),
                ac_pc_method.get("correct"),
                ac_pc_method.get("questions"),
            ),
            _pct_acc(
                "ESC",
                esc_method.get("accuracy"),
                esc_method.get("correct"),
                esc_method.get("questions"),
            ),
            _pct_acc(
                "ESC+PC",
                esc_pc_method.get("accuracy"),
                esc_pc_method.get("correct"),
                esc_pc_method.get("questions"),
            ),
        ]
        bits = [bit for bit in bits if bit]
        return " / ".join(bits) if bits else "—"
    if family == "lenint":
        return "—"
    if family == "pc":
        return (
            _pct_acc(
                "PC",
                pc_method.get("accuracy"),
                pc_method.get("correct"),
                pc_method.get("questions"),
            )
            or "—"
        )
    bits = [
        _pct_acc(
            "SC",
            sc_method.get("accuracy"),
            sc_method.get("correct"),
            sc_method.get("questions"),
        ),
        _pct_acc(
            "PC",
            pc_method.get("accuracy"),
            pc_method.get("correct"),
            pc_method.get("questions"),
        ),
    ]
    bits = [bit for bit in bits if bit]
    return " / ".join(bits) if bits else "—"


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


def gpu_short_name(name: str) -> str:
    text = name or ""
    upper = text.upper()
    rtx = re.search(r"RTX\s*(\d{3,4}\s*(?:TI|SUPER)?)", upper)
    if rtx:
        return re.sub(r"\s+", "", rtx.group(1))
    chip = re.search(r"\b(B200|H200|H100|A100|A800|A6000|L40S|L40|V100|T4)\b", upper)
    if chip:
        return chip.group(1)
    return re.sub(r"^(NVIDIA|GeForce)\s+", "", text, flags=re.I).strip() or text


def host_info(gpu_rows: list[dict]) -> dict:
    hostname = socket.gethostname()
    shorts = [row.get("short") or "" for row in gpu_rows]
    counts: dict[str, int] = {}
    for short in shorts:
        if short:
            counts[short] = counts.get(short, 0) + 1
    if len(counts) == 1:
        chip, n = next(iter(counts.items()))
        gpu_summary = f"{n}× {chip}"
    elif counts:
        gpu_summary = " / ".join(f"{n}× {chip}" for chip, n in counts.items())
    else:
        gpu_summary = ""
    return {
        "hostname": hostname,
        "label": os.environ.get("PC_CONSOLE_HOST_LABEL", "").strip(),
        "gpu_summary": gpu_summary,
    }


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
                "short": gpu_short_name(parts[1]),
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
        if not re.search(
            r"run_math\.py|run_math_deepseek\.py|run_math_deepseek_lenint\.py|run_gsm8k\.py|run_polr\.py|run_cpt_llama_adapter\.py|EngineCore",
            line,
        ):
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
    if "polr" in name:
        return "polr"
    if "lenint" in name:
        return "ablation"
    if "selrec" in name:
        return "ablation"
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
    if "first100" in name or "next100" in name:
        return "smoke"
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
            counts = {
                key: count_jsonl(Path(path)) if path else 0
                for key, path in result_paths.items()
            }
            sc_path = Path(result_paths.get("sc") or "")
            pc_path = Path(result_paths.get("pc") or "")
            polr_path = Path(result_paths.get("polr") or "")
            ac_path = Path(result_paths.get("ac") or "")
            esc_path = Path(result_paths.get("esc") or "")
            pc_ac_path = Path(result_paths.get("pc_ac") or "")
            pc_esc_path = Path(result_paths.get("pc_esc") or "")
            sc_n = counts.get("sc") or 0
            pc_n = counts.get("pc") or 0
            polr_n = counts.get("polr") or 0
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
                    "cancelled",
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
            regrade_path = path.with_name(
                path.name.replace(".run.json", ".numeric_regrade.json")
            )
            regrade = {}
            if regrade_path.exists():
                try:
                    regrade = json.loads(regrade_path.read_text())
                except json.JSONDecodeError:
                    regrade = {}
            dataset_blob = " ".join(
                str(item or "")
                for item in (
                    path.name,
                    config.get("dataset"),
                    config.get("dataset_task"),
                    config.get("tag"),
                )
            ).lower()
            is_math500 = "math500" in dataset_blob or "math-500" in dataset_blob
            if regrade.get("methods") and not is_math500:
                merged_methods = dict(methods)
                merged_methods.update(regrade["methods"])
                methods = merged_methods
            sc_method = methods.get("SC") or {}
            pc_method = methods.get("PC") or {}
            polr_method = methods.get("PoLR") or {}
            ac_method = methods.get("AC") or {}
            esc_method = methods.get("ESC") or {}
            ac_pc_method = methods.get("AC+PC") or {}
            esc_pc_method = methods.get("ESC+PC") or {}
            pass1_method = methods.get("Pass@1(SC)") or methods.get("Pass@1") or {}
            sc_cost = costs.get("sc") or {}
            pc_cost = costs.get("pc") or {}
            polr_cost = costs.get("polr") or {}
            is_lenint = "lenint" in path.name.lower() or "lenint" in result_paths
            is_polr = bool(
                polr_n
                or config.get("method") == "polr"
                or str(config.get("protocol_version") or "").startswith("polr")
            )
            cfg_methods = config.get("methods") or []
            is_pc_only = (not is_polr) and (
                cfg_methods == ["pc"] or (pc_n > 0 and sc_n == 0)
            )
            sc_acc = sc_method.get("accuracy")
            pc_acc = pc_method.get("accuracy")
            sc_tok = sc_cost.get("avg_generated_tokens")
            pc_tok = pc_cost.get("avg_generated_tokens")
            last_pc = last_jsonl_row(pc_path) if pc_path.name else None
            last_sc = last_jsonl_row(sc_path) if sc_path.name else None
            last_polr = last_jsonl_row(polr_path) if polr_path.name else None
            last_ac = last_jsonl_row(ac_path) if ac_path.name else None
            last_esc = last_jsonl_row(esc_path) if esc_path.name else None
            last_pc_ac = last_jsonl_row(pc_ac_path) if pc_ac_path.name else None
            last_pc_esc = last_jsonl_row(pc_esc_path) if pc_esc_path.name else None
            last = last_pc or last_sc or last_polr or last_pc_ac or last_pc_esc or last_ac or last_esc
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
            cfg_methods = [str(item).lower() for item in (config.get("methods") or [])]
            method_names = {
                "sc": "SC",
                "pc": "PC",
                "ac": "AC",
                "esc": "ESC",
                "pc_ac": "AC+PC",
                "pc_esc": "ESC+PC",
                "polr": "PoLR",
            }
            if is_lenint:
                progress_n = counts.get("lenint") or 0
                planned = int(record.get("planned") or config.get("planned_units") or 0)
                family = "lenint"
                method_label = "LenInt"
            elif is_polr:
                progress_n = polr_n
                family = "polr"
                method_label = "PoLR"
            elif cfg_methods:
                progress_counts = [counts.get(name) or 0 for name in cfg_methods]
                progress_n = min(progress_counts) if progress_counts else 0
                family = "acesc" if any(name in {"ac", "esc", "pc_ac", "pc_esc"} for name in cfg_methods) else (
                    "pc" if is_pc_only else "scpc"
                )
                method_label = " / ".join(method_names.get(name, name.upper()) for name in cfg_methods)
            elif is_pc_only:
                progress_n = pc_n
                family = "pc"
                method_label = "PC"
            else:
                progress_n = min(sc_n, pc_n)
                family = "scpc"
                method_label = "SC / PC"
            paired = progress_n
            if status in {"running"} and paired and planned and elapsed:
                rate = elapsed / max(paired, 1)
                eta = rate * max(planned - paired, 0)
            polr_tok = polr_cost.get("avg_generated_tokens")
            last_pred = (last.get("predictions") or {}) if last else {}
            tau = config.get("early_stop_threshold")
            esc_rule = config.get("esc_stop_rule")
            if family == "acesc":
                extra = ""
                if esc_rule and esc_rule != "window":
                    extra = f" esc={esc_rule}"
                display_meta = (
                    f"{(record.get('engine_times') or summary.get('engine_times') or {}).get('model_type') or ''} · "
                    f"GPU {config.get('gpus')} · B={config.get('max_branches')} "
                    f"C={config.get('beta_threshold')} τ={tau}{extra} max_tok={config.get('max_tokens')}"
                ).strip(" ·")
                progress_label = "配对进度"
            elif family == "lenint":
                display_meta = None
                progress_label = "进度"
            elif family == "polr":
                display_meta = None
                progress_label = "进度"
            elif family == "pc":
                display_meta = None
                progress_label = "进度"
            else:
                display_meta = None
                progress_label = "配对进度"
            accuracy_summary = _accuracy_summary(
                family,
                sc_method=sc_method,
                pc_method=pc_method,
                polr_method=polr_method,
                ac_method=ac_method,
                ac_pc_method=ac_pc_method,
                esc_method=esc_method,
                esc_pc_method=esc_pc_method,
            )
            runs.append(
                {
                    "file": str(path),
                    "tag": tag,
                    "fingerprint": fingerprint,
                    "kind": classify_math_run(path.name),
                    "family": family,
                    "method_label": method_label,
                    "accuracy_summary": accuracy_summary,
                    "progress_label": progress_label,
                    "display_meta": display_meta,
                    "early_stop_threshold": tau,
                    "esc_stop_rule": config.get("esc_stop_rule"),
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
                    "polr_done": polr_n,
                    "paired": paired,
                    "max_branches": config.get("n_prefixes") if is_polr else config.get("max_branches"),
                    "beta_threshold": config.get("beta_threshold"),
                    "prefix_len": config.get("prefix_len"),
                    "n_prefixes": config.get("n_prefixes"),
                    "max_tokens": config.get("max_tokens"),
                    "sc_accuracy": sc_acc,
                    "pc_accuracy": pc_acc,
                    "polr_accuracy": polr_method.get("accuracy"),
                    "sc_correct": sc_method.get("correct"),
                    "sc_questions": sc_method.get("questions"),
                    "pc_correct": pc_method.get("correct"),
                    "pc_questions": pc_method.get("questions"),
                    "polr_correct": polr_method.get("correct"),
                    "polr_questions": polr_method.get("questions"),
                    "ac_accuracy": ac_method.get("accuracy"),
                    "esc_accuracy": esc_method.get("accuracy"),
                    "ac_pc_accuracy": ac_pc_method.get("accuracy"),
                    "esc_pc_accuracy": esc_pc_method.get("accuracy"),
                    "ac_correct": ac_method.get("correct"),
                    "esc_correct": esc_method.get("correct"),
                    "ac_questions": ac_method.get("questions"),
                    "esc_questions": esc_method.get("questions"),
                    "ac_avg_tokens": (costs.get("ac") or {}).get("avg_generated_tokens"),
                    "esc_avg_tokens": (costs.get("esc") or {}).get("avg_generated_tokens"),
                    "sc_avg_tokens": sc_tok,
                    "pc_avg_tokens": pc_tok,
                    "polr_avg_tokens": polr_tok,
                    "sc_total_generated": sc_cost.get("total_generated_tokens"),
                    "pc_total_generated": pc_cost.get("total_generated_tokens"),
                    "sc_total_generation_s": sc_cost.get("total_generation_seconds"),
                    "pc_total_generation_s": pc_cost.get("total_generation_seconds"),
                    "sc_avg_generation_s": sc_cost.get("avg_generation_seconds"),
                    "pc_avg_generation_s": pc_cost.get("avg_generation_seconds"),
                    "polr_avg_generation_s": polr_cost.get("avg_generation_seconds"),
                    "pass1_accuracy": pass1_method.get("accuracy"),
                    "pass1_correct": pass1_method.get("correct"),
                    "pass1_questions": pass1_method.get("questions"),
                    "accuracy_regraded": bool(regrade.get("methods")) and not is_math500,
                    "regrade_rescued": regrade.get("rescued"),
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
                        "method": last.get("method") or method_label,
                        "sc": _first_prediction(last_pred),
                        "tokens": last.get("total_generated_tokens"),
                        "ratio": last.get("final_prefix_ratio") or last.get("kappa"),
                        "truncations": last.get("truncations"),
                    },
                }
            )
    return runs


def cpt_runs() -> list[dict]:
    """CPT writes per-question JSON, not paired-root run.json."""
    rows = []
    logs = CPT_DIR / "logs"
    if not CPT_DIR.exists():
        return rows
    pid_files = sorted(logs.glob("cpt_llama31_gsm8k_n*.pid")) if logs.exists() else []
    for pid_path in pid_files:
        tag = pid_path.stem
        if "smoke" in tag:
            continue
        results = CPT_DIR / tag / "results"
        summary_path = CPT_DIR / f"{tag}.summary.json"
        done = (
            len([path for path in results.glob("*.json") if path.stem.isdigit()])
            if results.exists()
            else 0
        )
        try:
            pid = int(pid_path.read_text().strip())
        except ValueError:
            pid = None
        alive = bool(pid) and pid_alive(pid)
        if done >= 1319:
            status = "complete"
        elif alive:
            status = "running"
        elif done:
            status = "stale" if pid and not alive else "partial"
        else:
            continue
        summary = {}
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
            except json.JSONDecodeError:
                summary = {}
        started = datetime.fromtimestamp(pid_path.stat().st_mtime, TZ)
        elapsed = (datetime.now(TZ) - started).total_seconds()
        eta = None
        if status == "running" and done and elapsed:
            eta = (elapsed / done) * max(1319 - done, 0)
        last_idx = None
        if results.exists():
            numeric = [int(path.stem) for path in results.glob("*.json") if path.stem.isdigit()]
            last_idx = max(numeric) if numeric else None
        n_workers = None
        if "_n" in tag:
            try:
                n_workers = int(tag.rsplit("_n", 1)[-1])
            except ValueError:
                n_workers = None
        gpu = "4" if tag.endswith("_n32") or tag.endswith("_n20") else None
        accuracy_summary = _accuracy_summary(
            "cpt",
            cpt_acc=summary.get("accuracy"),
            cpt_correct=summary.get("correct"),
            cpt_n=summary.get("n_questions") or done,
        )
        rows.append(
            {
                "file": str(pid_path),
                "tag": tag,
                "fingerprint": tag,
                "kind": "cpt",
                "family": "cpt",
                "method_label": f"CPT (N={n_workers})" if n_workers else "CPT",
                "accuracy_summary": accuracy_summary,
                "progress_label": "进度",
                "display_meta": (
                    f"llama · GPU {gpu or '—'} · CPT N={n_workers or '—'} "
                    f"max_tok=2048 · Llama adapter"
                ),
                "status": status,
                "pid": pid,
                "pid_alive": alive,
                "gpus": gpu,
                "model": "Llama-3.1-8B-Instruct",
                "model_path": None,
                "dataset": "/home/26xyw/data/Datasets/gsm8k_test_cpt.jsonl",
                "model_type": "llama",
                "protocol": "cpt-llama-adapter",
                "planned": 1319,
                "paired": done,
                "max_branches": n_workers,
                "n_prefixes": n_workers,
                "beta_threshold": None,
                "max_tokens": 2048,
                "started": started.astimezone(timezone.utc).isoformat(),
                "finished": (
                    datetime.fromtimestamp(
                        max(path.stat().st_mtime for path in results.glob("*.json") if path.stem.isdigit()),
                        timezone.utc,
                    ).isoformat()
                    if status == "complete" and results.exists()
                    and any(path.stem.isdigit() for path in results.glob("*.json"))
                    else None
                ),
                "elapsed_s": elapsed,
                "eta_s": eta,
                "last_question": None
                if last_idx is None
                else {
                    "id": f"gsm8k_{last_idx:04d}",
                    "method": "CPT",
                    "sc": None,
                    "tokens": None,
                    "ratio": None,
                    "truncations": None,
                },
            }
        )
    return rows


def dataset_status() -> list[dict]:
    specs = [
        ("GSM8K test", DATASETS / "gsm8k_test.jsonl", True),
        ("MATH-500 test", DATASETS / "math500_test.jsonl", True),
        ("AIME 2024", DATASETS / "aime_2024.jsonl", True),
        ("AIME 2025", DATASETS / "aime_2025.jsonl", True),
        ("GPQA-Diamond", DATASETS / "gpqa_diamond.jsonl", True),
        ("Dev-Mix（已不用于选 C*）", DATASETS / "devmix_2027.jsonl", False),
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
                "status": (
                    "ready" if path.exists() and n
                    else ("optional" if not required else "missing")
                ),
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
            missing = [
                row["name"]
                for row in datasets
                if row.get("required") and row["status"] != "ready"
            ]
            if not missing:
                status = "complete"
            elif aime_ready or gpqa_ready:
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
        elif key == "gsm8k_llama31_c08":
            hits = [
                row
                for row in math
                if "gsm8k_llama31" in row["tag"]
                and "_c08_" in row["tag"]
                and "smoke" not in row["tag"]
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
        elif key == "deepseek_gsm8k_first100":
            hits = [
                row
                for row in math
                if "deepseek" in row["tag"].lower()
                and "gsm8k" in row["tag"].lower()
                and "first100" in row["tag"].lower()
            ]
            if not hits:
                status = "queued"
            else:
                hit = hits[0]
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
                    detail = f"已配对 {hit['paired']}/{hit['planned']}"
        elif key == "deepseek_gsm8k_next500":
            hits = [
                row
                for row in math
                if "deepseek" in row["tag"].lower()
                and "gsm8k" in row["tag"].lower()
                and "next500" in row["tag"].lower()
            ]
            if not hits:
                status = "queued"
            else:
                hit = hits[0]
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
                    detail = f"已配对 {hit['paired']}/{hit['planned']}"
        elif key == "deepseek_gsm8k_rest719":
            hits = [
                row
                for row in math
                if "deepseek" in row["tag"].lower()
                and "gsm8k" in row["tag"].lower()
                and "rest719" in row["tag"].lower()
            ]
            if not hits:
                status = "queued"
            else:
                hit = hits[0]
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
                    detail = f"已配对 {hit['paired']}/{hit['planned']}"
        elif key == "deepseek_gsm8k_full1319":
            hits = [
                row
                for row in math
                if "full1319" in row["tag"].lower() and "deepseek" in row["tag"].lower()
            ]
            if not hits:
                status = "queued"
            else:
                hit = hits[0]
                status = hit["status"]
                progress = {"done": hit["paired"], "total": hit["planned"] or 1319}
                link = hit["tag"]
                detail = f"已配对 {hit['paired']}/{hit['planned'] or 1319}"
        elif key == "deepseek_gsm8k_pc_vote_v2":
            hits = [
                row
                for row in math
                if "pc_vote_v2" in row["tag"].lower() and "deepseek" in row["tag"].lower()
            ]
            if not hits:
                status = "queued"
            else:
                hit = hits[0]
                status = hit["status"]
                progress = {"done": hit["paired"] or 0, "total": hit["planned"] or 1319}
                link = hit["tag"]
                acc = hit.get("pc_accuracy")
                detail = f"PC {progress['done']}/{progress['total']}"
                if acc is not None:
                    detail = (
                        f"PC {progress['done']}/{progress['total']}；"
                        f"{100 * acc:.2f}%"
                    )
        elif key == "cancelled":
            status = "cancelled"
        elif key == "gsm8k_llama31_c00":
            hits = [
                row
                for row in math
                if "gsm8k_llama31" in row["tag"]
                and "_c00_" in row["tag"]
                and "ac099" not in row["tag"]
                and "smoke" not in row["tag"]
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
                    status = (
                        "stale"
                        if any(row["status"] == "stale" for row in hits)
                        else "failed"
                    )
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
        elif key == "deepseek_gsm8k_c00":
            hits = [
                row
                for row in math
                if "deepseek" in row["tag"].lower()
                and "gsm8k" in row["tag"].lower()
                and "_c00_" in row["tag"].lower()
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
                    status = (
                        "stale"
                        if any(row["status"] == "stale" for row in hits)
                        else "failed"
                    )
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
                    detail = f"已配对 {paired}/{planned}"
        elif key == "deepseek_gsm8k_c07":
            hits = [
                row
                for row in math
                if "deepseek" in row["tag"].lower()
                and "gsm8k" in row["tag"].lower()
                and "_c07_" in row["tag"].lower()
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
                    status = (
                        "stale"
                        if any(row["status"] == "stale" for row in hits)
                        else "failed"
                    )
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
                    detail = f"已配对 {paired}/{planned}"
        elif key == "ablations_llama31_gsm8k":
            status = "partial"
            detail = (
                "Llama GSM8K：C=0.8 / SC 已填。DeepSeek GSM8K：C=0.8 融合（Acc overlay 1277，Tok vote-v2 −34.2%）/ C=0 / C=0.7 已填。"
                "DeepSeek MATH-500 选择×recovery：Ablation C/E 已用 A/B/C/D 替换原 Llama 占位表（raw Acc）。"
                "阈值 {0.9,0.95} 与 PVC 查找表不做。"
            )
        elif key == "protocol_v4":
            status = "complete"
            detail = "2026-09-09 冻结。主表 PC (C=0.8)；不做 C* 与 PVC 查找表。"
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
                and "first100" not in row["tag"].lower()
                and "next100" not in row["tag"].lower()
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
        elif key == "qwen30_a3b_math500_first100":
            hit = next(
                (
                    row
                    for row in math
                    if "qwen3_30b_a3b" in row["tag"].lower()
                    and "first100" in row["tag"].lower()
                    and "smoke" not in row["tag"].lower()
                ),
                None,
            )
            if not hit:
                status = "queued"
            else:
                status = hit["status"]
                progress = {"done": hit["paired"] or 0, "total": hit["planned"] or 100}
                link = hit["tag"]
                if hit.get("sc_accuracy") is not None:
                    detail = (
                        f"已配对 {progress['done']}/{progress['total']}；"
                        f"SC {100 * hit['sc_accuracy']:.0f}% / "
                        f"PC {100 * (hit['pc_accuracy'] or 0):.0f}%；Table 1 部分格 n=100"
                    )
                else:
                    detail = (
                        f"已配对 {progress['done']}/{progress['total']}；"
                        "JSONL 前 100 行；Table 1 部分格"
                    )
        elif key == "qwen30_a3b_math500_next100":
            hit = next(
                (
                    row
                    for row in math
                    if "qwen3_30b_a3b" in row["tag"].lower()
                    and "next100" in row["tag"].lower()
                    and "smoke" not in row["tag"].lower()
                ),
                None,
            )
            if not hit:
                status = "queued"
            else:
                status = hit["status"]
                progress = {"done": hit["paired"] or 0, "total": hit["planned"] or 99}
                link = hit["tag"]
                if hit.get("sc_accuracy") is not None:
                    detail = (
                        f"已配对 {progress['done']}/{progress['total']}；"
                        f"SC {100 * hit['sc_accuracy']:.0f}% / "
                        f"PC {100 * (hit['pc_accuracy'] or 0):.0f}%；"
                        "source_row 100–199，排除 151"
                    )
                else:
                    detail = (
                        f"已配对 {progress['done']}/{progress['total']}；"
                        "JSONL 101–200（0-index 100–199，跳过工程行 151）"
                    )
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
        elif key == "cpt_llama31_gsm8k":
            results = Path(
                "/home/26xyw/data/path-consistency-results/deepconf-cpt/"
                "cpt_llama31_gsm8k_n32/results"
            )
            summary_path = Path(
                "/home/26xyw/data/path-consistency-results/deepconf-cpt/"
                "cpt_llama31_gsm8k_n32.summary.json"
            )
            done = (
                len([p for p in results.glob("*.json") if p.stem.isdigit()])
                if results.exists()
                else 0
            )
            progress = {"done": done, "total": 1319}
            pid_path = Path(
                "/home/26xyw/data/path-consistency-results/deepconf-cpt/logs/"
                "cpt_llama31_gsm8k_n32.pid"
            )
            alive = False
            if pid_path.exists():
                try:
                    pid = int(pid_path.read_text().strip())
                except ValueError:
                    pid = None
                alive = bool(pid) and pid_alive(pid)
            if done >= 1319:
                status = "complete"
            elif done or alive:
                status = "running"
            else:
                status = "queued"
            detail = f"N=32 已完成 {done}/1319；N=20 不进 Table 2"
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text())
                except json.JSONDecodeError:
                    summary = {}
                acc = summary.get("accuracy")
                n = summary.get("n_questions")
                if acc is not None and n:
                    detail = (
                        f"N=32 {n}/1319；Acc {100 * acc:.2f}%；"
                        f"tok {summary.get('tok_per_q') or 0:.0f}"
                    )
            link = "cpt_llama31_gsm8k_n32"
        elif key == "polr_llama31_gsm8k":
            hits = [
                row
                for row in math
                if "polr_llama31_gsm8k" in row["tag"]
                and "lp64" in row["tag"]
                and "smoke" not in row["tag"]
            ]
            if not hits:
                status = "queued"
            else:
                hits.sort(
                    key=lambda row: (
                        0 if row["status"] in {"running", "loading_engine"} else 1,
                        0 if "lp64" in row["tag"] else 1,
                        0 if row["status"] != "cancelled" else 1,
                    )
                )
                hit = hits[0]
                status = hit["status"]
                progress = {"done": hit["paired"] or 0, "total": hit["planned"] or 1319}
                link = hit["tag"]
                acc = hit.get("polr_accuracy")
                tok = hit.get("polr_avg_tokens")
                detail = f"已完成 {progress['done']}/{progress['total']}"
                if hit["status"] == "cancelled":
                    detail = (
                        f"已取消 {progress['done']}/{progress['total']}"
                        + (f"；当时 PoLR {100 * acc:.2f}%" if acc is not None else "")
                    )
                elif acc is not None:
                    tok_s = f"；tok {tok:.0f}" if tok else ""
                    lp = (hit.get("prefix_len") or "")
                    lp_s = f" L_p={lp}" if lp else ""
                    detail = (
                        f"{progress['done']}/{progress['total']}；"
                        f"PoLR{lp_s} {100 * acc:.2f}%{tok_s}"
                    )
        elif key == "polr_llama31_gsm8k_lp256":
            hits = [
                row
                for row in math
                if "polr_llama31_gsm8k" in row["tag"]
                and "lp256" in row["tag"]
                and "smoke" not in row["tag"]
            ]
            if not hits:
                status = "queued"
            else:
                hits.sort(
                    key=lambda row: (
                        0 if row["status"] in {"running", "loading_engine"} else 1,
                        0 if row["status"] != "cancelled" else 1,
                    )
                )
                hit = hits[0]
                status = hit["status"]
                progress = {"done": hit["paired"] or 0, "total": hit["planned"] or 1319}
                link = hit["tag"]
                acc = hit.get("polr_accuracy")
                tok = hit.get("polr_avg_tokens")
                detail = f"L_p=256 已完成 {progress['done']}/{progress['total']}"
                if hit["status"] == "cancelled" and progress["done"] < 1319:
                    detail = (
                        f"先前取消 {progress['done']}/{progress['total']}"
                        + (f"；当时 PoLR {100 * acc:.2f}%" if acc is not None else "")
                    )
                elif acc is not None:
                    tok_s = f"；tok {tok:.0f}" if tok else ""
                    detail = (
                        f"{progress['done']}/{progress['total']}；"
                        f"PoLR L_p=256 {100 * acc:.2f}%{tok_s}；已进 Table 2"
                    )
        elif key == "polr_llama31_gsm8k_lp128":
            hits = [
                row
                for row in math
                if "polr_llama31_gsm8k" in row["tag"]
                and "lp128" in row["tag"]
                and "smoke" not in row["tag"]
            ]
            if not hits:
                status = "queued"
            else:
                hits.sort(
                    key=lambda row: (
                        0 if row["status"] in {"running", "loading_engine"} else 1,
                        0 if row["status"] != "cancelled" else 1,
                    )
                )
                hit = hits[0]
                status = hit["status"]
                progress = {"done": hit["paired"] or 0, "total": hit["planned"] or 1319}
                link = hit["tag"]
                acc = hit.get("polr_accuracy")
                tok = hit.get("polr_avg_tokens")
                detail = f"L_p=128 {progress['done']}/{progress['total']}；未满 1319 不进 Table 2"
                if hit["status"] == "cancelled" and progress["done"] < 1319:
                    detail = (
                        f"先前取消 {progress['done']}/{progress['total']}"
                        + (f"；当时 PoLR {100 * acc:.2f}%" if acc is not None else "")
                    )
                elif acc is not None and progress["done"] >= 1319:
                    tok_s = f"；tok {tok:.0f}" if tok else ""
                    detail = (
                        f"{progress['done']}/{progress['total']}；"
                        f"PoLR L_p=128 {100 * acc:.2f}%{tok_s}；已进 Table 2"
                    )
                elif acc is not None:
                    tok_s = f"；tok {tok:.0f}" if tok else ""
                    detail = (
                        f"{progress['done']}/{progress['total']}；"
                        f"PoLR L_p=128 {100 * acc:.2f}%{tok_s}；未满 1319 不进 Table 2"
                    )
        elif key == "gsm8k_llama31_ac":
            hits = [
                row
                for row in math
                if "gsm8k_llama31" in row["tag"]
                and "_ac_" in row["tag"]
                and "smoke" not in row["tag"]
            ]
            if not hits:
                status = "queued"
            else:
                hit = hits[0]
                status = hit["status"]
                progress = {"done": hit["paired"] or 0, "total": hit["planned"] or 1319}
                link = hit["tag"]
                acc = hit.get("ac_accuracy")
                plus = hit.get("ac_pc_accuracy")
                done_n = progress["done"]
                total_n = progress["total"]
                if done_n >= total_n and hit["status"] == "complete":
                    plus_s = f"；AC+PC {100 * plus:.2f}%" if plus is not None else ""
                    acc_s = f"AC {100 * acc:.2f}%" if acc is not None else "AC 完成"
                    detail = f"{done_n}/{total_n}；{acc_s}{plus_s}；已进 Table 2"
                else:
                    detail = f"已配对 {done_n}/{total_n}；未满 1319 不进 Table 2"
                    if acc is not None:
                        plus_s = f"；AC+PC {100 * plus:.2f}%" if plus is not None else ""
                        detail = (
                            f"{done_n}/{total_n}；"
                            f"AC {100 * acc:.2f}%{plus_s}；未满 1319 不进 Table 2"
                        )
        elif key == "gsm8k_llama31_esc":
            hits = [
                row
                for row in math
                if "gsm8k_llama31" in row["tag"]
                and "_esc_" in row["tag"]
                and "smoke" not in row["tag"]
            ]
            if not hits:
                status = "queued"
            else:
                hit = hits[0]
                status = hit["status"]
                progress = {"done": hit["paired"] or 0, "total": hit["planned"] or 1319}
                link = hit["tag"]
                acc = hit.get("esc_accuracy")
                plus = hit.get("esc_pc_accuracy")
                done_n = progress["done"]
                total_n = progress["total"]
                if done_n >= total_n and hit["status"] == "complete":
                    plus_s = f"；ESC+PC {100 * plus:.2f}%" if plus is not None else ""
                    acc_s = f"ESC {100 * acc:.2f}%" if acc is not None else "ESC 完成"
                    detail = f"{done_n}/{total_n}；{acc_s}{plus_s}；已进 Table 2"
                else:
                    detail = f"已配对 {done_n}/{total_n}；未满 1319 不进 Table 2"
                    if acc is not None:
                        plus_s = f"；ESC+PC {100 * plus:.2f}%" if plus is not None else ""
                        detail = (
                            f"{done_n}/{total_n}；"
                            f"ESC {100 * acc:.2f}%{plus_s}；未满 1319 不进 Table 2"
                        )
        elif key == "gsm8k_llama31_ac099":
            hits = [
                row
                for row in math
                if "ac099" in row["tag"] and "smoke" not in row["tag"]
            ]
            if not hits:
                status = "queued"
            else:
                hit = hits[0]
                status = hit["status"]
                progress = {"done": hit["paired"] or 0, "total": hit["planned"] or 1319}
                link = hit["tag"]
                acc = hit.get("ac_accuracy")
                plus = hit.get("ac_pc_accuracy")
                plus_s = f"；AC+PC {100 * plus:.2f}%" if plus is not None else ""
                acc_s = f"AC {100 * acc:.2f}%" if acc is not None else "AC τ=0.99"
                detail = (
                    f"{progress['done']}/{progress['total']}；{acc_s}{plus_s}"
                    "；已进 Table 2（诊断行，不覆盖 τ=0.95）"
                )
        elif key == "gsm8k_llama31_esc2w8":
            hits = [
                row
                for row in math
                if "esc2w8" in row["tag"] and "smoke" not in row["tag"]
            ]
            if not hits:
                status = "queued"
            else:
                hit = hits[0]
                status = hit["status"]
                progress = {"done": hit["paired"] or 0, "total": hit["planned"] or 1319}
                link = hit["tag"]
                acc = hit.get("esc_accuracy")
                plus = hit.get("esc_pc_accuracy")
                plus_s = f"；ESC+PC {100 * plus:.2f}%" if plus is not None else ""
                acc_s = f"ESC {100 * acc:.2f}%" if acc is not None else "ESC 两窗/8"
                detail = (
                    f"{progress['done']}/{progress['total']}；{acc_s}{plus_s}"
                    "；已进 Table 2（诊断行，不覆盖单窗 ESC）"
                )
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
            hit = next(
                (
                    row
                    for row in math
                    if "deepseek" in row["tag"].lower()
                    and "math500" in row["tag"].lower()
                    and "selrec" not in row["tag"].lower()
                    and "lenint" not in row["tag"].lower()
                    and "smoke" not in row["tag"].lower()
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
                status = "queued"
        elif key == "deepseek_selrec_rand_on":
            hit = next(
                (
                    row
                    for row in math
                    if "selrec_rand_on" in row["tag"] and "smoke" not in row["tag"]
                ),
                None,
            )
            if not hit:
                status = "queued"
            else:
                status = hit["status"]
                progress = {"done": hit["paired"] or 0, "total": hit["planned"] or 490}
                link = hit["tag"]
                acc = hit.get("pc_accuracy")
                acc_s = f"；PC {100 * acc:.2f}%" if acc is not None else ""
                detail = (
                    f"{progress['done']}/{progress['total']}{acc_s}"
                    "；臂 B uniform+recovery 开；已进消融表 C/E；不进主表"
                )
        elif key == "deepseek_selrec_blc_off":
            hit = next(
                (
                    row
                    for row in math
                    if "selrec_blc_off" in row["tag"] and "smoke" not in row["tag"]
                ),
                None,
            )
            if not hit:
                status = "queued"
            else:
                status = hit["status"]
                progress = {"done": hit["paired"] or 0, "total": hit["planned"] or 490}
                link = hit["tag"]
                acc = hit.get("pc_accuracy")
                acc_s = f"；PC {100 * acc:.2f}%" if acc is not None else ""
                detail = (
                    f"{progress['done']}/{progress['total']}{acc_s}"
                    "；臂 C BLC+recovery 关；已进消融表 C/E；不进主表"
                )
        elif key == "deepseek_selrec_rand_off":
            hit = next(
                (
                    row
                    for row in math
                    if "selrec_rand_off" in row["tag"] and "smoke" not in row["tag"]
                ),
                None,
            )
            if not hit:
                status = "queued"
            else:
                status = hit["status"]
                progress = {"done": hit["paired"] or 0, "total": hit["planned"] or 490}
                link = hit["tag"]
                acc = hit.get("pc_accuracy")
                acc_s = f"；PC {100 * acc:.2f}%" if acc is not None else ""
                detail = (
                    f"{progress['done']}/{progress['total']}{acc_s}"
                    "；臂 D uniform+recovery 关；已进消融表 C/E；不进主表"
                )
        elif key == "deepseek_lenint":
            hit = next(
                (
                    row
                    for row in math
                    if "lenint" in row["tag"] and "smoke" not in row["tag"]
                ),
                None,
            )
            if not hit:
                status = "queued"
            else:
                status = hit["status"]
                progress = {"done": hit["paired"] or 0, "total": hit["planned"] or 0}
                link = hit["tag"]
                detail = (
                    f"{progress['done']}/{progress['total']}"
                    "；r∈{0,.2,.4,.6,.8} R=8 complete；窗 Acc 88.2%→87.8%，tok −43.6%；已进 Ablation G；不进主表"
                )
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
        finished = None
        if link:
            for row in math:
                if row.get("tag") == link or row.get("fingerprint") == link:
                    finished = row.get("finished")
                    break
        out.append(
            {
                **item,
                "status": status,
                "detail": detail,
                "progress": progress,
                "run_tag": link,
                "finished": finished,
            }
        )
    return out


def collect_state() -> dict:
    gpu_rows = gpus()
    procs = running_processes()
    gsm8k = gsm8k_results()
    math = math_runs() + cpt_runs()
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
        "host": host_info(gpu_rows),
        "gpus": gpu_rows,
        "processes": procs,
        "running": running,
        "stale": stale,
        "gsm8k": gsm8k,
        "math": math,
        "datasets": datasets,
        "catalog": catalog,
        "plan": PLAN,
        "rebuttal": REBUTTAL,
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
  .host { color: var(--text); font-size: 14px; margin: 0 0 6px; }
  .host .chip { color: var(--ok); }
  h2 { font-size: 13px; font-weight: 600; color: var(--muted); text-transform: uppercase;
    letter-spacing: 0.08em; margin: 28px 0 10px; }
  h3 { font-size: 15px; font-weight: 600; margin: 22px 0 6px; letter-spacing: -0.01em; }
  .caption { color: var(--muted); font-size: 12px; margin: 0 0 10px; max-width: 110ch; }
  .footnote { color: var(--warn); font-size: 12px; margin: 8px 0 0; max-width: 110ch; }
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
  table.paper tr.block-head td { color: var(--muted); font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600;
    background: var(--bg); padding-top: 10px; }
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
  .pill.cancelled { color: var(--dim); }
  .pill.optional, .pill.info { color: var(--accent); border-color: #2a3a5a; }
  .plan-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; margin-top: 10px; }
  .plan-card { background: var(--bg2); border: 1px solid var(--line); padding: 10px 12px; }
  .plan-card h4 { margin: 0 0 6px; font-size: 13px; font-weight: 600; }
  .plan-card p { margin: 0; color: var(--muted); font-size: 12px; }
  .call.plan { border-color: #2a4a5a; }
  .plat-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 10px; }
  .plat { background: var(--bg2); border: 1px solid var(--line); padding: 12px 14px; }
  .plat h3 { margin: 0; font-size: 15px; font-weight: 600; }
  .plat .spec { color: var(--dim); font-size: 12px; margin: 2px 0 6px; }
  .plat .role { font-size: 13px; margin-bottom: 8px; }
  .plat ul { margin: 0; padding-left: 16px; }
  .plat li { margin: 4px 0; color: var(--muted); font-size: 12px; }
  .run-block { margin-top: 16px; }
  .run-block h3 { margin: 0 0 4px; font-size: 14px; }
  @media (max-width: 900px) { .plat-grid { grid-template-columns: 1fr; } }
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
  #toasts {
    position: fixed; top: 16px; right: 16px; z-index: 50;
    display: flex; flex-direction: column; gap: 8px;
    width: min(400px, calc(100vw - 32px));
    pointer-events: none;
  }
  .toast {
    pointer-events: auto;
    background: var(--bg2);
    border: 1px solid #3d5a2c;
    padding: 10px 12px;
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 4px 10px;
    align-items: start;
  }
  .toast .k { color: var(--run); font-size: 11px; letter-spacing: 0.06em;
    text-transform: uppercase; }
  .toast .t { font-weight: 600; }
  .toast .d { color: var(--muted); font-size: 12px; grid-column: 1 / 2; }
  .toast-x {
    background: transparent; color: var(--muted); border: 1px solid var(--line);
    width: 22px; height: 22px; cursor: pointer; font-size: 16px; line-height: 1;
    padding: 0;
  }
  .toast-x:hover { color: var(--text); border-color: var(--text); }
</style>
</head>
<body>
<div id="toasts" aria-live="polite"></div>
<div class="wrap">
  <h1>Path-Consistency 实验控制台</h1>
  <div class="host" id="hostline"></div>
  <div class="sub">ICLR 2027 改稿 · 数据每 8 秒从磁盘与 <span class="mono">nvidia-smi</span> 刷新 · <span id="stamp"></span></div>

  <h2>审稿补实验 · 平台与任务</h2>
  <div class="call plan" id="rebuttal-headline"></div>
  <div class="plat-grid" id="rebuttal-platforms"></div>
  <div id="rebuttal-experiments"></div>
  <div class="note" id="rebuttal-skip" style="margin-top:8px"></div>

  <div class="row" id="kpis"></div>
  <div class="call" id="headline"></div>

  <h2>GPU</h2>
  <div class="row" id="gpus"></div>

  <h2>正在进行</h2>
  <div id="running"></div>

  <h2>冻结协议（2026-09-09，仍有效）</h2>
  <div class="call plan" id="plan-headline"></div>
  <div class="plan-grid" id="plan-decisions"></div>
  <h3>Go / no-go</h3>
  <div id="plan-go"></div>
  <h3>排期</h3>
  <div id="plan-schedule"></div>
  <div class="note" id="plan-cancelled" style="margin-top:8px"></div>

  <h2>论文表格</h2>
  <div class="note" id="paper-notes"></div>
  <div id="table1"></div>
  <div id="table2"></div>
  <div id="table3"></div>
  <h2>试跑（不进 Table 1）</h2>
  <div id="pilots"></div>
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
const FILTERS = ["全部","进行中","已完成","未开始","部分/卡住","已取消"];
let filter = "全部";
let state = null;

function pct(x, d=1){ if(x==null||Number.isNaN(x)) return "—"; return (100*x).toFixed(d)+"%"; }
function num(x, d=1){ if(x==null||Number.isNaN(x)) return "—"; return Number(x).toFixed(d); }
function pill(status){
  const map = {complete:"完成", running:"进行中", queued:"未开始", partial:"部分", stale:"进程丢失", failed:"失败", paused:"已暂停", cancelled:"已取消", terminated_protocol_invalid:"已作废", missing:"缺失", ready:"就绪", loading_engine:"加载中", optional:"非必需"};
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

function planPill(status){
  const map = {frozen:"已冻结", ablation:"消融", optional:"可选", cancelled:"已取消", done:"完成", running:"进行中", queued:"未开始", partial:"部分"};
  const cls = {frozen:"complete", done:"complete", ablation:"partial", optional:"optional", cancelled:"cancelled", running:"run", queued:"queued", partial:"partial"};
  return `<span class="pill ${cls[status]||status}">${map[status]||status}</span>`;
}
function renderRebuttal(){
  const r = state.rebuttal;
  const head = document.getElementById("rebuttal-headline");
  if(!r){
    head.textContent = "补实验计划未加载。";
    return;
  }
  head.innerHTML =
    `<b>${r.updated}</b> · 交接文件 <span class="mono">${r.source}</span>` +
    `<div class="note" style="margin-top:6px;max-width:none">${r.headline}</div>` +
    `<div class="dim" style="margin-top:4px">${r.protocol}</div>`;
  document.getElementById("rebuttal-platforms").innerHTML = (r.platforms||[]).map(p => `
    <div class="plat">
      <h3>${p.name}</h3>
      <div class="spec">${p.spec}</div>
      <div class="role">${p.role}</div>
      <ul>${(p.lines||[]).map(line => `<li>${line}</li>`).join("")}</ul>
    </div>`).join("");
  document.getElementById("rebuttal-experiments").innerHTML = (r.groups||[]).map(g => `
    <div class="run-block">
      <h3>${g.title}</h3>
      <div class="note" style="max-width:none">${g.reviews}</div>
      <div class="note" style="max-width:none">${g.platform} · ${g.model}</div>
      <div class="note" style="max-width:none">${g.data}</div>
      <div class="note" style="max-width:none">${g.note}</div>
      <table>
        <thead><tr><th>系统</th><th>tag</th></tr></thead>
        <tbody>${(g.runs||[]).map(run => `
          <tr><td>${run.what}</td><td class="mono">${run.tag}</td></tr>`).join("")}
        </tbody>
      </table>
    </div>`).join("");
  document.getElementById("rebuttal-skip").innerHTML =
    "<b>不新跑</b> · " + (r.skip||[]).join("<br>");
}
function renderPlan(){
  const p = state.plan;
  if(!p){
    document.getElementById("plan-headline").textContent = "计划未加载。";
    return;
  }
  document.getElementById("plan-headline").innerHTML =
    `<b>计划 ${p.version}</b> · 冻结 ${p.frozen_at}` +
    `<div class="note" style="margin-top:6px;max-width:none">${p.headline}</div>` +
    `<div class="dim" style="margin-top:4px">${p.source}</div>`;
  document.getElementById("plan-decisions").innerHTML = (p.decisions||[]).map(d => `
    <div class="plan-card">
      <h4>${d.title} ${planPill(d.status)}</h4>
      <p>${d.text}</p>
    </div>`).join("");
  document.getElementById("plan-go").innerHTML = `<table>
    <thead><tr><th>门槛</th><th>条件</th></tr></thead>
    <tbody>${(p.go_nogo||[]).map(g => `
      <tr><td>${g.must ? planPill("frozen") : planPill("optional")}</td>
          <td>${g.text}</td></tr>`).join("")}
    </tbody></table>`;
  document.getElementById("plan-schedule").innerHTML = `<table>
    <thead><tr><th>状态</th><th>窗口</th><th>事项</th></tr></thead>
    <tbody>${(p.schedule||[]).map(s => `
      <tr>
        <td>${planPill(s.status)}</td>
        <td>${s.when}<div class="note">${s.title}</div></td>
        <td class="note" style="max-width:none">${s.items}</td>
      </tr>`).join("")}
    </tbody></table>`;
  document.getElementById("plan-cancelled").textContent =
    "已取消：" + ((p.cancelled||[]).join(" · ") || "—");
}

function acc(run, which){
  const p = which==="sc" ? run.sc_accuracy : which==="polr" ? run.polr_accuracy : run.pc_accuracy;
  const c = which==="sc" ? run.sc_correct : which==="polr" ? run.polr_correct : run.pc_correct;
  const n = which==="sc" ? run.sc_questions : which==="polr" ? run.polr_questions : run.pc_questions;
  if(p==null) return "—";
  if(c==null || n==null) return pct(p,0);
  return pct(p,0) + " ("+c+"/"+n+")";
}
function isPolr(run){
  return run.family==="polr" || (run.protocol||"").indexOf("polr")>=0 || (run.tag||"").indexOf("polr")>=0;
}
function isCpt(run){
  return run.family==="cpt" || (run.tag||"").indexOf("cpt_")>=0;
}
function accSummary(run){
  if(run.accuracy_summary) return run.accuracy_summary;
  if(isCpt(run)) return "CPT —";
  if(isPolr(run)) return "PoLR " + acc(run, "polr");
  if(run.family==="acesc"){
    const bits = [];
    if(run.ac_accuracy!=null) bits.push("AC " + pct(run.ac_accuracy,0) + (run.ac_correct!=null && run.ac_questions!=null ? " ("+run.ac_correct+"/"+run.ac_questions+")" : ""));
    if(run.ac_pc_accuracy!=null) bits.push("AC+PC " + pct(run.ac_pc_accuracy,0));
    if(run.esc_accuracy!=null) bits.push("ESC " + pct(run.esc_accuracy,0) + (run.esc_correct!=null && run.esc_questions!=null ? " ("+run.esc_correct+"/"+run.esc_questions+")" : ""));
    if(run.esc_pc_accuracy!=null) bits.push("ESC+PC " + pct(run.esc_pc_accuracy,0));
    return bits.join(" / ") || "—";
  }
  if(run.family==="pc") return "PC " + acc(run, "pc");
  return "SC " + acc(run,"sc") + " / PC " + acc(run,"pc");
}
function progressLabel(run){
  if(run.progress_label) return run.progress_label;
  return (isPolr(run) || isCpt(run) || run.family==="pc") ? "进度" : "配对进度";
}
function runMeta(run){
  if(run.display_meta) return run.display_meta;
  if(isCpt(run)){
    const n = run.n_prefixes != null ? run.n_prefixes : run.max_branches;
    return (run.model_type||"") + " · GPU " + (run.gpus||"—") + " · CPT N=" + n + " max_tok=" + run.max_tokens;
  }
  if(isPolr(run)){
    const n = run.n_prefixes != null ? run.n_prefixes : run.max_branches;
    const lp = run.prefix_len != null ? run.prefix_len : "—";
    return (run.model_type||"") + " · GPU " + (run.gpus||"—") + " · PoLR N=" + n + " L_p=" + lp + " max_tok=" + run.max_tokens;
  }
  const tau = run.early_stop_threshold != null ? " τ=" + run.early_stop_threshold : "";
  return (run.model_type||"") + " · GPU " + (run.gpus||"—") + " · B=" + run.max_branches + " C=" + run.beta_threshold + tau + " max_tok=" + run.max_tokens;
}
function where(run){
  if(run.gpus==="api" || (run.model_type||"").indexOf("deepseek")>=0) return "API";
  return "GPU " + (run.gpus==null ? "—" : run.gpus);
}
function fmtAcc(cell){
  if(!cell || cell.acc==null) return "";
  const pctv = (100*cell.acc).toFixed(2);
  if(cell.correct==null || cell.n==null) return pctv;
  const extra = cell.partial ? " · 部分" : "";
  return pctv + '<div class="stub">' + cell.correct + "/" + cell.n + extra + "</div>";
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
  const blockLabel = (id) => ((t.blocks||[]).find(b => b.id===id)||{}).label || id;
  let lastBlock = null;
  const body = t.rows.map(row => {
    let lead = "";
    if (row.block && row.block !== lastBlock) {
      lastBlock = row.block;
      lead = `<tr class="block-head"><td class="left" colspan="${1+t.columns.length}">${blockLabel(row.block)}</td></tr>`;
    }
    const cells = t.columns.map(c => {
      const v = (row.cells||{})[c.id];
      if(c.kind==="acc") return tdCell(fmtAcc(v), !!v);
      if(c.kind==="tok") return tdCell(fmtTok(v), !!(v && v.tok!=null));
      if(c.kind==="lat") return tdCell(
        (v && v.latency_s!=null) ? fmtNum(v.latency_s, 1, " s") : "",
        !!(v && v.latency_s!=null)
      );
      return tdCell("", false);
    }).join("");
    const pc = row.method_id==="pc" ? " method-pc" : "";
    return lead + `<tr class="${pc}"><td class="left">${row.method}</td>${cells}</tr>`;
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
    const n = row.n ? `<div class="stub">n=${row.n}${row.partial?" · 部分":row.planned?"":" · 已完成配对"}</div>` : "";
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
    const cells = (row.cells||[]).map(c => tdCell(c||"", !!c)).join("");
    return `<tr><td class="left">${row.label}</td>${detail}${cells}</tr>`;
  }).join("");
  return `
    <h3>${t.title}</h3>
    <div class="fill-meta">已填 ${t.filled} / ${t.total} 格</div>
    <div class="caption">${t.caption}</div>
    <div class="paper-scroll"><table class="paper">
      <thead><tr>${head}</tr></thead>
      <tbody>${body}</tbody>
    </table></div>
    ${t.note ? `<div class="footnote">${t.note}</div>` : ""}`;
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
  document.getElementById("pilots").innerHTML =
    (paper.pilots||[]).map(renderAblation).join("");
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
    <div class="stat"><div class="v live">${state.running.length}</div><div class="l">正在跑的实验</div></div>
    <div class="stat"><div class="v">${state.counts.complete}</div><div class="l">清单已完成</div></div>
    <div class="stat"><div class="v warn">${state.counts.queued}</div><div class="l">尚未开始</div></div>`;
}
function render(){
  if(!state) return;
  document.getElementById("stamp").textContent = state.generated_at;
  const host = state.host || {};
  const hostBits = [`<span class="mono">${host.hostname||"—"}</span>`];
  if(host.gpu_summary) hostBits.push(`<span class="chip">${host.gpu_summary}</span>`);
  if(host.label) hostBits.push(host.label);
  document.getElementById("hostline").innerHTML = hostBits.join(" · ");
  try { renderPaper(state.paper); } catch (err) { console.error(err); }
  try { renderRebuttal(); } catch (err) { console.error(err); }
  try { renderPlan(); } catch (err) { console.error(err); }
  renderKpis();

  const stale = state.stale[0];
  let head = "没有正在跑的实验。";
  if(state.running.length){
    head = state.running.map(run => `
      <div style="margin:0 0 12px">
        <b class="live">进行中</b> ${run.method_label || (isCpt(run)?"CPT":(isPolr(run)?"PoLR":"SC / PC"))} · ${run.model} · ${where(run)} · ${progressLabel(run)} ${run.paired}/${run.planned}
        ${bar(run.paired, run.planned)}
        <div class="note">PID ${run.pid} · 已用 ${eta(run.elapsed_s)} · 预计剩余 ${eta(run.eta_s)}
        · 已完成题准确率 ${accSummary(run)}
        ${run.last_question ? " · 最近题 "+run.last_question.id : ""}</div>
      </div>`).join("");
  } else if(stale){
    head = `<b class="bad">进程丢失</b> ${stale.tag} 标记为 running 但 PID ${stale.pid} 已不在。检查 orphan EngineCore 后决定是否续跑。`;
  }
  document.getElementById("headline").innerHTML = head;

  document.getElementById("gpus").innerHTML = state.gpus.map(g => `
    <div class="gpu" title="${g.name||""}">
      <div>GPU ${g.index} · ${g.short||g.name||"—"} ${g.busy ? '<span class="pill run">占用</span>' : '<span class="pill queued">空闲</span>'}</div>
      <div class="mem">${num(g.used_mib,0)} / ${num(g.total_mib,0)} MiB · util ${num(g.util,0)}%</div>
      ${bar(g.used_mib, g.total_mib)}
    </div>`).join("") || '<div class="empty">无法读取 nvidia-smi</div>';

  if(!state.running.length && !state.processes.length){
    document.getElementById("running").innerHTML = '<div class="empty">无 run_math / run_polr / run_cpt / run_math_deepseek / run_gsm8k / EngineCore 进程。</div>';
  } else {
    const procRows = state.processes.map(p =>
      `<tr><td class="mono">${p.pid}</td><td>${p.etime}</td><td>${p.cpu}%</td><td>${p.rss_mb} MB</td><td class="mono">${p.cmd}</td></tr>`
    ).join("");
    const runRows = state.running.map(r =>
      `<tr><td>${pill(r.status)}</td><td class="mono">${r.method_label || (isCpt(r)?"CPT":(isPolr(r)?"PoLR":"SC / PC"))} · ${r.model}<div class="note">${r.tag}</div></td><td>${where(r)}</td>
       <td>${r.paired}/${r.planned}</td><td>${accSummary(r)}</td>
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
    "已取消": x => x.status==="cancelled",
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
      <td class="mono">${r.method_label || "—"} · ${r.fingerprint}<div class="note">${r.tag || ""} · ${runMeta(r)}</div></td>
      <td>${r.paired}/${r.planned}</td>
      <td>${accSummary(r)}</td>
      <td>${r.decode_saving==null?"—":pct(r.decode_saving)}</td>
      <td>${r.truncations_sc??"—"}</td>
    </tr>`).join("");
  document.getElementById("math").innerHTML = `<table>
    <thead><tr><th>状态</th><th>类型</th><th>Run</th><th>进度</th><th>已完成准确率</th><th>Decode 节省</th><th>SC 截断</th></tr></thead>
    <tbody>${mathRows}</tbody></table>`;

  document.getElementById("datasets").innerHTML = `<table>
    <thead><tr><th>数据集</th><th>状态</th><th>行数</th><th>路径</th></tr></thead>
    <tbody>${state.datasets.map(r => `
      <tr><td>${r.name}${r.required?"":"<div class=\"note\">不阻塞</div>"}</td><td>${pill(r.status==="ready"?"complete":r.status==="optional"?"optional":"queued")}</td>
      <td>${r.rows||0}</td><td class="mono">${r.path}</td></tr>`).join("")}</tbody></table>`;
}

const DISMISS_KEY = "pc.console.dismissedNotifs.v2";
const SEEN_KEY = "pc.console.seenTaskStatus.v2";
const PENDING_KEY = "pc.console.pendingNotifs.v2";
const RECENT_MS = 12 * 3600 * 1000;

function loadMap(key){
  try { return JSON.parse(localStorage.getItem(key) || "{}") || {}; }
  catch(e){ return {}; }
}
function saveMap(key, obj){
  localStorage.setItem(key, JSON.stringify(obj));
}
function dismissToast(id){
  const dismissed = loadMap(DISMISS_KEY);
  dismissed[id] = Date.now();
  saveMap(DISMISS_KEY, dismissed);
  const pending = loadMap(PENDING_KEY);
  delete pending[id];
  saveMap(PENDING_KEY, pending);
  const el = document.querySelector('#toasts .toast[data-id="'+id+'"]');
  if(el) el.remove();
}
function makeToast(task){
  const el = document.createElement("div");
  el.className = "toast";
  el.dataset.id = task.id;
  const body = document.createElement("div");
  const k = document.createElement("div");
  k.className = "k";
  k.textContent = "任务完成";
  const title = document.createElement("div");
  title.className = "t";
  title.textContent = task.title;
  body.appendChild(k);
  body.appendChild(title);
  if(task.detail){
    const d = document.createElement("div");
    d.className = "d";
    d.textContent = task.detail;
    body.appendChild(d);
  }
  const x = document.createElement("button");
  x.type = "button";
  x.className = "toast-x";
  x.setAttribute("aria-label", "关闭通知");
  x.textContent = "×";
  x.addEventListener("click", function(ev){
    ev.preventDefault();
    ev.stopPropagation();
    dismissToast(task.id);
  });
  el.appendChild(body);
  el.appendChild(x);
  return el;
}
function syncNotifications(st){
  const dismissed = loadMap(DISMISS_KEY);
  const seen = loadMap(SEEN_KEY);
  const pending = loadMap(PENDING_KEY);
  const now = Date.now();
  const live = new Set();
  for(const it of (st.catalog || [])){
    const prev = seen[it.id];
    seen[it.id] = it.status;
    live.add(it.id);
    if(it.status !== "complete") continue;
    if(dismissed[it.id]) continue;
    const became = prev && prev !== "complete" &&
      ["running","partial","loading_engine"].includes(prev);
    let recent = false;
    if(it.finished){
      const ts = Date.parse(it.finished);
      recent = Number.isFinite(ts) && (now - ts) < RECENT_MS && !prev;
    }
    if(became || recent || pending[it.id]){
      pending[it.id] = {
        id: it.id,
        title: it.title,
        detail: it.detail || "",
        at: pending[it.id] && pending[it.id].at ? pending[it.id].at : now
      };
    }
  }
  for(const id of Object.keys(pending)){
    if(dismissed[id] || !live.has(id)) delete pending[id];
  }
  saveMap(SEEN_KEY, seen);
  saveMap(PENDING_KEY, pending);
  const root = document.getElementById("toasts");
  const have = new Set(Array.from(root.children).map(el => el.dataset.id));
  const want = new Set(Object.keys(pending));
  for(const el of Array.from(root.children)){
    if(!want.has(el.dataset.id)) el.remove();
  }
  Object.keys(pending).forEach(id => {
    if(have.has(id)) return;
    root.appendChild(makeToast(pending[id]));
  });
}

async function tick(){
  try {
    const res = await fetch("/api/state", {cache:"no-store"});
    state = await res.json();
    render();
    syncNotifications(state);
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
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=28787)
    args = parser.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"Experiment console at http://{args.host}:{args.port}/  ({iso_cst(now_cst())})",
        flush=True,
    )
    if args.host in {"0.0.0.0", "::"}:
        print(
            "LAN: http://10.10.60.106:%s/  (no IDE tunnel). "
            "Off-campus: ssh -N -L %s:127.0.0.1:%s USER@10.10.60.106"
            % (args.port, args.port, args.port),
            flush=True,
        )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)


if __name__ == "__main__":
    main()
