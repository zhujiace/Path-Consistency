#!/usr/bin/env python3
"""Matched-history prefix-length intervention via DeepSeek-V4-Flash.

Freezes MATH-500 window-0 SC roots (fingerprint 4856d4ef5673), picks one BLC
source, and resamples one window at r in {0, 0.2, 0.4, 0.6, 0.8}.
Does not overwrite the main-table SC/PC JSONL.

    /home/26xyw/.conda/envs/PC/bin/python -u run_math_deepseek_lenint.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from path_consistency.deepseek_api import DeepSeekClient, load_deepseek_env
from path_consistency.deepseek_inference import DeepSeekMathInference
from path_consistency.length_intervention import (
    LENGTH_RATIOS,
    NONZERO_RATIOS,
    build_question_snapshot,
    compact_trace,
    final_vote_answers,
    prefix_ids_for_ratio,
    window_traces,
)
from path_consistency.math_eval import grade_answer, majority_vote, recanonicalize_trace_answers
from run_math import combined_sha256, file_sha256, load_eval_records, stable_question_seed
from run_math_deepseek import (
    ENGINEERING_ROWS,
    index_root_caches,
    load_root_cache,
    recanonicalize_root_windows,
)

DEFAULT_CACHE = (
    "/home/26xyw/data/path-consistency-results/math500/"
    "math500_deepseek_v4_flash_b32_c08_rep0_4856d4ef5673_root_cache"
)
DEFAULT_SMOKE = (
    "/home/26xyw/data/path-consistency-results/math500/reports/deepseek_smoke.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="/home/26xyw/data/Datasets/math500_test.jsonl"
    )
    parser.add_argument(
        "--outdir",
        default="/home/26xyw/data/path-consistency-results/math500",
    )
    parser.add_argument("--tag", default="math500_deepseek_lenint_r8_seed0")
    parser.add_argument("--reuse-root-cache", action="append", default=[])
    parser.add_argument("--smoke-path", default=DEFAULT_SMOKE)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--source-rows", default=None)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--beta-threshold", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--logprobs", type=int, default=20)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--question-concurrency", type=int, default=2)
    parser.add_argument("--max-inflight", type=int, default=8)
    parser.add_argument("--user-id", default="pc-math500-lenint")
    parser.add_argument(
        "--common-support-only",
        action="store_true",
        default=True,
        help="Only sample questions whose frozen source has 20/40/60/80%% endpoints",
    )
    parser.add_argument("--include-partial-support", action="store_true")
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--sample-seed", type=int, default=2027)
    return parser.parse_args()


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def append_jsonl(path: Path, row: dict, lock: threading.Lock) -> None:
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with lock:
        with path.open("a") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def unit_key(qid: str, ratio: float, replicate: int) -> tuple[str, str, int]:
    return (qid, f"{ratio:.1f}", int(replicate))


def load_done(path: Path) -> set[tuple[str, str, int]]:
    done = set()
    if not path.exists():
        return done
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            done.add(unit_key(row["question_id"], float(row["ratio"]), int(row["replicate"])))
    return done


def prefix_variant_from_smoke(path: str) -> str:
    smoke_path = Path(path)
    if not smoke_path.exists():
        return "reasoning_content"
    try:
        smoke = json.loads(smoke_path.read_text())
    except json.JSONDecodeError:
        return "reasoning_content"
    return (smoke.get("protocol") or {}).get("prefix_variant") or "reasoning_content"


def find_source_trace(windows: list[dict], source_id: str) -> dict | None:
    for trace in window_traces(windows, 0):
        if trace.get("branch_id") == source_id:
            return trace
    return None


def planned_units(snapshots: list[dict], replicates: int, common_only: bool) -> list[tuple[str, float, int]]:
    units = []
    for snap in snapshots:
        qid = snap["question_id"]
        if common_only and not snap.get("common_support"):
            continue
        ratios = [0.0]
        for ratio in NONZERO_RATIOS:
            if str(ratio) in (snap.get("endpoints") or {}):
                ratios.append(ratio)
        if common_only:
            ratios = list(LENGTH_RATIOS)
        for ratio in ratios:
            if ratio > 0 and str(ratio) not in (snap.get("endpoints") or {}):
                continue
            for rep in range(replicates):
                units.append((qid, ratio, rep))
    return units


def main() -> None:
    args = parse_args()
    if args.include_partial_support:
        args.common_support_only = False
    output_dir = Path(args.outdir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    caches = args.reuse_root_cache or [DEFAULT_CACHE]
    reused = index_root_caches(caches)
    dataset_path = Path(args.dataset).expanduser().resolve()
    requested_rows = (
        [int(value) for value in args.source_rows.split(",")]
        if args.source_rows
        else None
    )
    data, dataset_task = load_eval_records(
        dataset_path, args.max_questions, args.sample_seed, requested_rows
    )
    if dataset_task == "math":
        data = [item for item in data if item["source_row"] not in ENGINEERING_ROWS]
        if args.max_questions is not None and requested_rows is None:
            data = data[: args.max_questions]

    snapshots = []
    skipped = []
    for item in data:
        qid = item["question_id"]
        stem = qid.replace("/", "_")
        cache_path = reused.get(stem)
        windows = recanonicalize_root_windows(load_root_cache(cache_path), dataset_task) if cache_path else None
        if not windows:
            skipped.append({"question_id": qid, "reason": "missing_root_cache"})
            continue
        recanonicalize_trace_answers(
            [trace for window in windows for trace in (window.get("traces") or [])],
            dataset_task,
        )
        snap = build_question_snapshot(
            question_id=qid,
            windows=windows,
            gold=item["answer"],
            task=dataset_task,
            beta_threshold=args.beta_threshold,
            selection_seed=args.selection_seed,
        )
        snap["gold"] = item["answer"]
        snap["source_row"] = item["source_row"]
        snap["subject"] = item.get("subject")
        snapshots.append(snap)

    n_common = sum(1 for snap in snapshots if snap.get("common_support"))
    reason_counts: dict[str, int] = {}
    for snap in snapshots:
        reason = snap.get("reason") or ("common_support" if snap.get("common_support") else "other")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    frozen = {
        "protocol": "matched-history-length-v1",
        "root_cache": [str(Path(path).expanduser().resolve()) for path in caches],
        "beta_threshold": args.beta_threshold,
        "selection_seed": args.selection_seed,
        "skipped": skipped,
        "snapshots": [
            {
                key: snap[key]
                for key in (
                    "question_id",
                    "source_row",
                    "source_id",
                    "source_reasoning_tokens",
                    "endpoints",
                    "missing_ratios",
                    "common_support",
                    "gate_passed",
                    "reason",
                    "n_root_traces",
                    "n_tail_traces",
                    "history_answers",
                    "tail_answers",
                    "history_vote",
                    "history_correct",
                )
                if key in snap
            }
            for snap in snapshots
        ],
    }
    snapshot_sha = hashlib.sha256(
        json.dumps(frozen, sort_keys=True, default=list).encode()
    ).hexdigest()[:12]
    snapshot_path = output_dir / f"{args.tag}_{snapshot_sha}.snapshot.json"
    snapshot_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_sha256": snapshot_sha,
        "n_questions": len(snapshots),
        "n_common_support": n_common,
        "n_missing_cache": len(skipped),
        "reason_counts": reason_counts,
        **frozen,
        "snapshots": snapshots,
    }
    if not snapshot_path.exists():
        dump_json(snapshot_path, snapshot_payload)
    e0_path = output_dir / "reports" / f"{args.tag}_{snapshot_sha}_e0.json"
    dump_json(
        e0_path,
        {
            "snapshot": str(snapshot_path),
            "snapshot_sha256": snapshot_sha,
            "n_questions": len(snapshots),
            "n_common_support": n_common,
            "n_missing_cache": len(skipped),
            "reason_counts": reason_counts,
            "first_common_source_rows": [
                snap.get("source_row")
                for snap in snapshots
                if snap.get("common_support")
            ][:5],
        },
    )
    print(
        json.dumps(
            {
                "snapshot": str(snapshot_path),
                "n_questions": len(snapshots),
                "n_common_support": n_common,
                "n_missing_cache": len(skipped),
                "reason_counts": reason_counts,
                "e0": str(e0_path),
            },
            ensure_ascii=False,
        )
    )
    if args.snapshot_only:
        return

    prefix_variant = prefix_variant_from_smoke(args.smoke_path)
    implementation_paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().parent / "path_consistency" / "length_intervention.py",
        Path(__file__).resolve().parent / "path_consistency" / "deepseek_inference.py",
        Path(__file__).resolve().parent / "path_consistency" / "math_protocol.py",
    ]
    sampled_ids = [
        snap["question_id"]
        for snap in snapshots
        if (snap.get("common_support") or not args.common_support_only)
    ]
    config = {
        "protocol_version": "matched-history-length-v1-deepseek-api",
        "implementation_sha256": combined_sha256(implementation_paths),
        "snapshot_sha256": snapshot_sha,
        "snapshot_path": str(snapshot_path),
        "model": "deepseek-v4-flash",
        "dataset": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "dataset_task": dataset_task,
        "methods": ["lenint"],
        "selected_question_ids": sampled_ids,
        "excluded_engineering_source_rows": list(ENGINEERING_ROWS),
        "replicates": args.replicates,
        "ratios": list(LENGTH_RATIOS),
        "window_size": args.window_size,
        "max_branches": 32,
        "beta_threshold": args.beta_threshold,
        "max_tokens": args.max_tokens,
        "logprobs": args.logprobs,
        "reasoning_effort": args.reasoning_effort,
        "thinking": True,
        "selector": "blc",
        "selection_seed": args.selection_seed,
        "seed": args.seed,
        "prefix_variant": prefix_variant,
        "common_support_only": args.common_support_only,
        "question_concurrency": args.question_concurrency,
        "max_inflight": args.max_inflight,
        "user_id": args.user_id,
        "reuse_root_cache": [str(Path(path).expanduser().resolve()) for path in caches],
        "pc_length_policy": "matched_history_intervention",
        "note": "Not a test-time PVC controller. C=0.8 is not Dev-Mix C*.",
    }
    fingerprint_payload = {
        key: config[key]
        for key in (
            "protocol_version",
            "implementation_sha256",
            "snapshot_sha256",
            "dataset_sha256",
            "selected_question_ids",
            "excluded_engineering_source_rows",
            "ratios",
            "window_size",
            "beta_threshold",
            "max_tokens",
            "logprobs",
            "reasoning_effort",
            "thinking",
            "selector",
            "selection_seed",
            "seed",
            "prefix_variant",
            "common_support_only",
            "pc_length_policy",
        )
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, default=list).encode()
    ).hexdigest()[:12]
    base_name = f"{args.tag}_{fingerprint}"
    jsonl_path = output_dir / f"{base_name}.jsonl"
    manifest_path = output_dir / f"{base_name}.run.json"
    summary_path = output_dir / f"{base_name}.summary.json"

    units = planned_units(snapshots, args.replicates, args.common_support_only)
    snap_by_id = {snap["question_id"]: snap for snap in snapshots}
    item_by_id = {item["question_id"]: item for item in data}
    done = load_done(jsonl_path)
    remaining = [unit for unit in units if unit_key(*unit) not in done]

    config["planned_units"] = len(units)
    run_record = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "config": config,
        "fingerprint": fingerprint,
        "engine_times": {"model_type": "deepseek-v4-flash"},
        "result_paths": {"lenint": str(jsonl_path)},
        "planned": len(units),
        "done": len(done),
    }
    dump_json(manifest_path, run_record)

    write_lock = threading.Lock()
    done_lock = threading.Lock()
    started = time.perf_counter()
    client = None
    if remaining:
        env = load_deepseek_env()
        client = DeepSeekClient(
            env=env,
            max_inflight=args.max_inflight,
            user_id=args.user_id[:64],
        )
        engine = DeepSeekMathInference(
            client,
            selector="blc",
            prefix_variant=prefix_variant,
            reasoning_effort=args.reasoning_effort,
            want_logprobs=True,
            logprob_top_k=args.logprobs,
            task=dataset_task,
            selection_seed=args.selection_seed,
        )

    def process_unit(unit: tuple[str, float, int]) -> dict:
        qid, ratio, replicate = unit
        snap = snap_by_id[qid]
        item = item_by_id[qid]
        stem = qid.replace("/", "_")
        windows = recanonicalize_root_windows(
            load_root_cache(reused[stem]), dataset_task
        )
        if not windows:
            raise RuntimeError(f"{qid}: root cache missing at sample time")
        recanonicalize_trace_answers(
            [trace for window in windows for trace in (window.get("traces") or [])],
            dataset_task,
        )
        prefix_ids: list = []
        if ratio > 0:
            source = find_source_trace(windows, snap["source_id"])
            if source is None:
                raise RuntimeError(f"{qid}: frozen source {snap['source_id']} missing")
            prefix_ids = prefix_ids_for_ratio(source, snap, ratio)
            if not prefix_ids:
                raise RuntimeError(f"{qid}: no prefix ids for r={ratio}")
        prompt, base_ids = engine.build_prompt(item["problem"])
        sampling_seed = stable_question_seed(
            args.seed, f"{qid}:r{ratio:.1f}:rep{replicate}"
        )
        traces, elapsed = engine._generate_window(
            base_ids,
            prefix_ids,
            args.window_size,
            sampling_seed,
            0.6,
            0.95,
            20,
            args.max_tokens,
            args.logprobs,
            0.0,
        )
        recanonicalize_trace_answers(traces, dataset_task)
        for offset, trace in enumerate(traces):
            trace["branch_id"] = f"{qid}:lenint:r{ratio:.1f}:rep{replicate}:b{offset}"
            trace["window"] = 1
            trace["prefix_ratio"] = ratio
        answers = [trace.get("answer") for trace in traces]
        window_vote = majority_vote(answers)
        gold = item["answer"]
        final_answers = final_vote_answers(snap, answers)
        final_vote = majority_vote(final_answers)
        row = {
            "question_id": qid,
            "ratio": ratio,
            "replicate": replicate,
            "gold": gold,
            "source_row": item.get("source_row"),
            "subject": item.get("subject"),
            "source_id": snap.get("source_id"),
            "prefix_tokens": len(prefix_ids),
            "endpoint": (snap.get("endpoints") or {}).get(str(ratio), {}).get("endpoint")
            if ratio > 0
            else 0,
            "window_vote": window_vote,
            "window_correct": grade_answer(dataset_task, window_vote, gold),
            "final_vote": final_vote,
            "final_correct": grade_answer(dataset_task, final_vote, gold),
            "n_final_votes": len(final_answers),
            "traces": [compact_trace(trace) for trace in traces],
            "total_generated_tokens": sum(
                int(trace.get("generated_tokens") or 0) for trace in traces
            ),
            "generation_seconds": elapsed,
            "actual_generation_seconds": elapsed,
            "truncations": sum(1 for trace in traces if trace.get("truncated")),
            "sampling_seed": sampling_seed,
            "config_fingerprint": fingerprint,
        }
        append_jsonl(jsonl_path, row, write_lock)
        with done_lock:
            done.add(unit_key(qid, ratio, replicate))
        return {
            "question_id": qid,
            "ratio": ratio,
            "replicate": replicate,
            "window_correct": row["window_correct"],
            "final_correct": row["final_correct"],
            "generated_tokens": row["total_generated_tokens"],
        }

    workers = max(1, args.question_concurrency)
    completed_now = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_unit, unit): unit for unit in remaining}
        for future in as_completed(futures):
            unit = futures[future]
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001 - keep run alive, record later
                print(
                    json.dumps(
                        {
                            "error": error.__class__.__name__,
                            "message": str(error),
                            "unit": {"question_id": unit[0], "ratio": unit[1], "replicate": unit[2]},
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            completed_now += 1
            if completed_now == 1 or completed_now % 10 == 0:
                print(
                    json.dumps(
                        {
                            "progress": f"{len(done)}/{len(units)}",
                            "last": result,
                            "elapsed_s": round(time.perf_counter() - started, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                run_record["done"] = len(done)
                run_record["run_elapsed_seconds"] = time.perf_counter() - started
                dump_json(manifest_path, run_record)

    run_record["status"] = "complete" if len(done) >= len(units) else "partial"
    run_record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_record["done"] = len(done)
    run_record["run_elapsed_seconds"] = time.perf_counter() - started
    dump_json(manifest_path, run_record)

    by_ratio: dict[str, dict[str, float]] = {}
    if jsonl_path.exists():
        with jsonl_path.open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        for ratio in LENGTH_RATIOS:
            subset = [row for row in rows if abs(float(row["ratio"]) - ratio) < 1e-9]
            if not subset:
                continue
            by_ratio[f"{ratio:.1f}"] = {
                "n": len(subset),
                "window_acc": sum(bool(row["window_correct"]) for row in subset) / len(subset),
                "final_acc": sum(bool(row["final_correct"]) for row in subset) / len(subset),
                "avg_generated_tokens": sum(row["total_generated_tokens"] for row in subset)
                / len(subset),
            }
    summary = {
        "fingerprint": fingerprint,
        "snapshot_sha256": snapshot_sha,
        "planned": len(units),
        "done": len(done),
        "n_common_support": n_common,
        "by_ratio": by_ratio,
        "status": run_record["status"],
    }
    dump_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if client is not None:
        client.close()


if __name__ == "__main__":
    main()
