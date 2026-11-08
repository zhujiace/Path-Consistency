#!/usr/bin/env python3
"""Regrade DeepSeek API experiment wrongs with format-tolerant equality.

Official JSONL ``predictions.*.correct`` flags are left unchanged. This script:

1. Re-runs the current grader (math-verify / AIME integer / GPQA letter).
2. For GSM8K, also accepts a single number wrapped in units, ``\\$``, or commas.
3. Remaining GSM8K / MATH / AIME mismatches (non-empty prediction) are sent to
   DeepSeek-V4-Flash as a format judge: same value, ignore presentation.
4. Writes a sidecar ``*.numeric_regrade.json`` next to each ``*.run.json`` so
   the console Table 1 can overlay accuracy without rewriting traces.

GPQA is not sent to the numeric/format judge.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from path_consistency.deepseek_api import DeepSeekClient, load_deepseek_env
from path_consistency.math_eval import extract_last_boxed, grade_answer


RESULTS_ROOT = Path("/home/26xyw/data/path-consistency-results")
PROTOCOL_DIRS = [
    RESULTS_ROOT / "math500",
    RESULTS_ROOT / "gsm8k",
    RESULTS_ROOT / "aime",
    RESULTS_ROOT / "gpqa",
]
MAIN_PRED_KEYS = ("SC", "PC", "Pass@1(SC)", "Pass@1(PC)", "Pass@1")
API_TASKS = {"gsm8k", "math500", "aime"}
JUDGE_PROMPT = """You are checking whether a model prediction matches a gold answer.
Ignore formatting only: units, currency symbols, thousands separators, LaTeX
wrappers, degree/percent signs, a leading variable like x=, list order, and
capitalization. The mathematical VALUE must be the same.
Examples of YES: gold 70000 vs $70,000; gold 160 vs 160 minutes;
gold 95 vs 1 hour 35 minutes; gold 90 degrees vs 90; gold 4/3 vs 4 over 3.
Examples of NO: gold 8 vs 12; gold 175 vs 765; a missing answer.
Reply with only \\boxed{yes} or \\boxed{no}.

GOLD:
<<GOLD>>

PREDICTION:
<<PRED>>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=RESULTS_ROOT / "reports",
        help="Aggregate report directory",
    )
    parser.add_argument("--no-api", action="store_true", help="Skip DeepSeek judge")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-api", type=int, default=None)
    return parser.parse_args()


def dataset_task(row: dict, fallback: str = "") -> str:
    return str(row.get("dataset_task") or row.get("subject") or fallback or "")


def prediction_keys(row: dict) -> list[str]:
    names = list((row.get("predictions") or {}).keys())
    ordered = [name for name in MAIN_PRED_KEYS if name in names]
    ordered.extend(name for name in names if name not in ordered)
    return ordered


def parse_judge_yes(text: str) -> bool | None:
    boxed = extract_last_boxed(text or "")
    blob = (boxed or text or "").strip().lower()
    blob = re.sub(r"[^a-z]", "", blob)
    if blob.startswith("yes"):
        return True
    if blob.startswith("no"):
        return False
    if blob in {"true", "equivalent", "match", "same"}:
        return True
    return None


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def discover_runs() -> list[dict]:
    found = []
    for directory in PROTOCOL_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.run.json")):
            if "deepseek" not in path.name.lower():
                continue
            try:
                record = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            found.append({"manifest": path, "record": record})
    return found


def collect_official_wrongs(rows: list[dict], jsonl_path: Path) -> list[dict]:
    items = []
    for row in rows:
        task = dataset_task(row)
        gold = row.get("gold")
        method_pred = row.get("answer")
        for name in prediction_keys(row):
            payload = row["predictions"][name]
            if payload.get("correct"):
                continue
            pred = payload.get("answer")
            if pred is None and name in {"SC", "PC"}:
                pred = method_pred
            items.append(
                {
                    "question_id": row.get("question_id"),
                    "method": row.get("method"),
                    "pred_key": name,
                    "task": task,
                    "gold": gold,
                    "pred": pred,
                    "jsonl": str(jsonl_path),
                }
            )
    return items


def local_rescue(task: str, pred, gold) -> str | None:
    if pred is None or gold is None:
        return None
    if grade_answer(task, pred, gold):
        if task == "gsm8k":
            return "gsm8k_numeric_or_math_equal"
        return "current_grader"
    return None


def judge_one(client: DeepSeekClient, gold, pred) -> dict:
    gold_text = "" if gold is None else str(gold)
    pred_text = "" if pred is None else str(pred)
    prompt = JUDGE_PROMPT.replace("<<GOLD>>", gold_text).replace("<<PRED>>", pred_text)
    response = client.complete(
        [{"role": "user", "content": prompt}],
        max_tokens=256,
        reasoning_effort="low",
        thinking=False,
        logprobs=False,
        stream=False,
        request_tag="numeric-regrade",
    )
    content = (response.get("content") or "") + "\n" + (response.get("reasoning_content") or "")
    verdict = parse_judge_yes(content)
    return {
        "verdict": verdict,
        "content": (response.get("content") or "")[:500],
        "api_id": response.get("id"),
        "finish_reason": response.get("finish_reason"),
    }


def overlay_for_rows(
    rows: list[dict],
    jsonl_path: Path,
    verdicts: dict[tuple[str, str, str], dict],
) -> tuple[dict[str, dict], list[dict], list[dict]]:
    methods: dict[str, dict] = {}
    rescues: list[dict] = []
    still: list[dict] = []
    for row in rows:
        task = dataset_task(row)
        gold = row.get("gold")
        method_pred = row.get("answer")
        for name in prediction_keys(row):
            payload = row["predictions"][name]
            bucket = methods.setdefault(
                name,
                {
                    "questions": 0,
                    "official_correct": 0,
                    "correct": 0,
                    "rescued": 0,
                    "accuracy": None,
                },
            )
            bucket["questions"] += 1
            official = bool(payload.get("correct"))
            if official:
                bucket["official_correct"] += 1
                bucket["correct"] += 1
                continue
            pred = payload.get("answer")
            if pred is None and name in {"SC", "PC"}:
                pred = method_pred
            via = local_rescue(task, pred, gold)
            if via is None and task in API_TASKS and pred not in (None, ""):
                key = (task, str(gold), str(pred))
                hit = verdicts.get(key) or {}
                if hit.get("verdict") is True:
                    via = "api_format_judge"
            if via:
                bucket["correct"] += 1
                bucket["rescued"] += 1
                rescues.append(
                    {
                        "question_id": row.get("question_id"),
                        "pred_key": name,
                        "task": task,
                        "gold": gold,
                        "pred": pred,
                        "via": via,
                        "jsonl": str(jsonl_path),
                    }
                )
            else:
                still.append(
                    {
                        "question_id": row.get("question_id"),
                        "pred_key": name,
                        "task": task,
                        "gold": gold,
                        "pred": pred,
                        "jsonl": str(jsonl_path),
                    }
                )
    for bucket in methods.values():
        n = bucket["questions"]
        bucket["accuracy"] = (bucket["correct"] / n) if n else None
    return methods, rescues, still


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs()
    pending: dict[tuple[str, str, str], dict] = {}
    per_run_rows: list[dict] = []

    for item in runs:
        record = item["record"]
        paths = record.get("result_paths") or {}
        sc_path = Path(paths.get("sc") or "")
        pc_path = Path(paths.get("pc") or "")
        sc_rows = load_jsonl(sc_path) if sc_path.name else []
        pc_rows = load_jsonl(pc_path) if pc_path.name else []
        bundle = {
            "manifest": item["manifest"],
            "record": record,
            "sc_path": sc_path,
            "pc_path": pc_path,
            "sc_rows": sc_rows,
            "pc_rows": pc_rows,
        }
        per_run_rows.append(bundle)
        for path, rows in ((sc_path, sc_rows), (pc_path, pc_rows)):
            for wrong in collect_official_wrongs(rows, path):
                if wrong["pred_key"] not in {"SC", "PC", "Pass@1(SC)", "Pass@1(PC)", "Pass@1"}:
                    continue
                if local_rescue(wrong["task"], wrong["pred"], wrong["gold"]):
                    continue
                if wrong["task"] not in API_TASKS or wrong["pred"] in (None, ""):
                    continue
                key = (wrong["task"], str(wrong["gold"]), str(wrong["pred"]))
                pending.setdefault(
                    key,
                    {
                        "task": wrong["task"],
                        "gold": wrong["gold"],
                        "pred": wrong["pred"],
                        "n": 0,
                    },
                )
                pending[key]["n"] += 1

    verdicts: dict[tuple[str, str, str], dict] = {}
    api_error = None
    keys = list(pending.keys())
    if args.max_api is not None:
        keys = keys[: args.max_api]
    if not args.no_api and keys:
        print(
            json.dumps(
                {
                    "event": "api_judge_start",
                    "unique_pairs": len(keys),
                    "concurrency": args.concurrency,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        client = DeepSeekClient(
            load_deepseek_env(),
            timeout_s=120.0,
            max_retries=4,
            max_inflight=args.concurrency,
            user_id="pc-numeric-regrade",
        )
        try:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {
                    pool.submit(judge_one, client, pending[key]["gold"], pending[key]["pred"]): key
                    for key in keys
                }
                done = 0
                for future in as_completed(futures):
                    key = futures[future]
                    done += 1
                    try:
                        verdicts[key] = future.result()
                    except Exception as exc:  # noqa: BLE001
                        verdicts[key] = {"verdict": None, "error": str(exc)}
                        api_error = str(exc)
                    if done % 20 == 0 or done == len(futures):
                        print(
                            json.dumps(
                                {
                                    "event": "api_judge_progress",
                                    "done": done,
                                    "total": len(futures),
                                }
                            ),
                            flush=True,
                        )
        finally:
            client.close()

    overlays = []
    all_rescues = []
    all_still = []
    for bundle in per_run_rows:
        methods: dict[str, dict] = {}
        rescues: list[dict] = []
        still: list[dict] = []
        for path, rows in (
            (bundle["sc_path"], bundle["sc_rows"]),
            (bundle["pc_path"], bundle["pc_rows"]),
        ):
            if not rows:
                continue
            part_methods, part_rescues, part_still = overlay_for_rows(
                rows, path, verdicts
            )
            for name, stats in part_methods.items():
                acc = methods.setdefault(
                    name,
                    {
                        "questions": 0,
                        "official_correct": 0,
                        "correct": 0,
                        "rescued": 0,
                        "accuracy": None,
                    },
                )
                acc["questions"] += stats["questions"]
                acc["official_correct"] += stats["official_correct"]
                acc["correct"] += stats["correct"]
                acc["rescued"] += stats["rescued"]
            rescues.extend(part_rescues)
            still.extend(part_still)
        for stats in methods.values():
            n = stats["questions"]
            stats["accuracy"] = (stats["correct"] / n) if n else None
        sidecar = {
            "schema": "deepseek-numeric-regrade-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "manifest": str(bundle["manifest"]),
            "note": (
                "Format-tolerant overlay. Original JSONL correct flags unchanged. "
                "GSM8K accepts a single number plus units/currency; remaining "
                "GSM8K/MATH/AIME mismatches used a DeepSeek format judge."
            ),
            "methods": methods,
            "rescued": len(rescues),
            "still_wrong": len(still),
            "api_pairs_used": len(verdicts),
        }
        sidecar_path = bundle["manifest"].with_name(
            bundle["manifest"].name.replace(".run.json", ".numeric_regrade.json")
        )
        sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n")
        overlays.append(
            {
                "sidecar": str(sidecar_path),
                "tag": bundle["manifest"].name.replace(".run.json", ""),
                "methods": {
                    name: {
                        "official": stats["official_correct"],
                        "regraded": stats["correct"],
                        "n": stats["questions"],
                    }
                    for name, stats in methods.items()
                    if name in {"SC", "PC", "Pass@1(SC)"}
                },
            }
        )
        all_rescues.extend(rescues)
        all_still.extend(still)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "no_api": args.no_api,
        "unique_api_pairs": len(keys) if not args.no_api else 0,
        "api_error": api_error,
        "verdicts_yes": sum(1 for item in verdicts.values() if item.get("verdict") is True),
        "verdicts_no": sum(1 for item in verdicts.values() if item.get("verdict") is False),
        "verdicts_unknown": sum(
            1 for item in verdicts.values() if item.get("verdict") not in {True, False}
        ),
        "overlays": overlays,
        "rescues_sample": all_rescues[:40],
        "still_wrong_sample": all_still[:40],
        "n_rescues": len(all_rescues),
        "n_still_wrong": len(all_still),
        "api_verdicts": [
            {
                "task": key[0],
                "gold": key[1],
                "pred": key[2],
                **value,
            }
            for key, value in verdicts.items()
        ],
    }
    report_path = args.outdir / "deepseek_numeric_regrade.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    summary = {
        "report": str(report_path),
        "overlays": len(overlays),
        "rescues": len(all_rescues),
        "still_wrong": len(all_still),
        "api_yes": report["verdicts_yes"],
        "api_no": report["verdicts_no"],
        "per_run": overlays,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
