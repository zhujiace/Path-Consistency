#!/usr/bin/env python3
"""Merge DeepSeek-V4-Flash GSM8K shard JSONL into one 1319-question summary.

Reads first100 / next500 / rest719 (or any gsm8k_deepseek_* SC/PC jsonl) under
~/data/path-consistency-results/gsm8k/. Does not belong in git output.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from run_math import load_rows, summarize


OUT = Path("/home/26xyw/data/path-consistency-results/gsm8k")
EXPECTED = 1319
SHARD_SUBSTR = ("first100", "next500", "rest719")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=OUT)
    parser.add_argument("--tag", default="gsm8k_deepseek_v4_flash_b32_c08_full1319")
    return parser.parse_args()


def shard_runs(outdir: Path) -> list[Path]:
    runs = []
    for path in sorted(outdir.glob("gsm8k_deepseek_v4_flash_b32_c08_*.run.json")):
        name = path.name.lower()
        if "full1319" in name:
            continue
        if any(part in name for part in SHARD_SUBSTR):
            runs.append(path)
    return runs


def main() -> None:
    args = parse_args()
    outdir = args.outdir.expanduser().resolve()
    sc_by: dict[str, dict] = {}
    pc_by: dict[str, dict] = {}
    sources = []
    engine_times = {}
    protocol = None
    model = None
    for manifest in shard_runs(outdir):
        rec = json.loads(manifest.read_text())
        cfg = rec.get("config") or {}
        protocol = protocol or cfg.get("protocol_version")
        model = model or cfg.get("model")
        engine_times = rec.get("engine_times") or engine_times
        rps = rec.get("result_paths") or {}
        sc_rows = load_rows(Path(rps["sc"])) if rps.get("sc") else []
        pc_rows = load_rows(Path(rps["pc"])) if rps.get("pc") else []
        for row in sc_rows:
            sc_by.setdefault(row["question_id"], row)
        for row in pc_rows:
            pc_by.setdefault(row["question_id"], row)
        sources.append(
            {
                "manifest": str(manifest),
                "status": rec.get("status"),
                "fingerprint": rec.get("fingerprint"),
                "sc": len(sc_rows),
                "pc": len(pc_rows),
            }
        )
    expected_ids = [f"gsm8k_{i:04d}" for i in range(EXPECTED)]
    missing_sc = [qid for qid in expected_ids if qid not in sc_by]
    missing_pc = [qid for qid in expected_ids if qid not in pc_by]
    paired = [
        qid for qid in expected_ids if qid in sc_by and qid in pc_by
    ]
    sc_path = outdir / f"{args.tag}_sc.jsonl"
    pc_path = outdir / f"{args.tag}_pc.jsonl"
    with sc_path.open("w") as handle:
        for qid in expected_ids:
            if qid in sc_by:
                handle.write(json.dumps(sc_by[qid], ensure_ascii=False) + "\n")
    with pc_path.open("w") as handle:
        for qid in expected_ids:
            if qid in pc_by:
                handle.write(json.dumps(pc_by[qid], ensure_ascii=False) + "\n")
    rows = load_rows(sc_path) + load_rows(pc_path)
    config = {
        "protocol_version": protocol,
        "model": model,
        "dataset": "/home/26xyw/data/Datasets/gsm8k_test.jsonl",
        "dataset_task": "gsm8k",
        "selected_question_ids": expected_ids,
        "methods": ["sc", "pc"],
        "max_branches": 32,
        "window_size": 4,
        "beta_threshold": 0.8,
        "note": "Merged DeepSeek GSM8K shards; C=0.8 default operating point.",
        "shard_manifests": sources,
    }
    summary = summarize(rows, config, engine_times)
    summary["paired_questions"] = len(paired)
    summary["missing_sc"] = missing_sc
    summary["missing_pc"] = missing_pc
    summary["complete_1319"] = not missing_sc and not missing_pc
    summary_path = outdir / f"{args.tag}.summary.json"
    manifest_path = outdir / f"{args.tag}.run.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    manifest_path.write_text(
        json.dumps(
            {
                "status": "complete" if summary["complete_1319"] else "partial",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": config,
                "result_paths": {"sc": str(sc_path), "pc": str(pc_path)},
                "summary_path": str(summary_path),
                "paired": len(paired),
                "planned": EXPECTED,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(
        json.dumps(
            {
                "paired": len(paired),
                "planned": EXPECTED,
                "complete": summary["complete_1319"],
                "missing_sc": missing_sc,
                "missing_pc": missing_pc,
                "summary": str(summary_path),
                "sc_jsonl": str(sc_path),
                "pc_jsonl": str(pc_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
