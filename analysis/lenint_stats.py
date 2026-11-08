#!/usr/bin/env python3
"""Paired stats for the matched-history length intervention JSONL.

Question-clustered. Raw JSONL grades. Not a PVC controller.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from path_consistency.math_eval import grade_answer

RATIOS = (0.0, 0.2, 0.4, 0.6, 0.8)
JSONL = Path(
    "/home/26xyw/data/path-consistency-results/math500/"
    "math500_deepseek_lenint_r8_seed0_2ebec2c2eced.jsonl"
)
SNAPSHOT = Path(
    "/home/26xyw/data/path-consistency-results/math500/"
    "math500_deepseek_lenint_r8_seed0_75d4941afc34.snapshot.json"
)
OUT = Path(
    "/home/26xyw/data/path-consistency-results/math500/reports/"
    "lenint_2ebec2c2eced_stats.json"
)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def se_mean(xs: list[float]) -> float:
    return sd(xs) / math.sqrt(len(xs)) if xs else float("nan")


def ci95(xs: list[float]) -> tuple[float, float]:
    m, e = mean(xs), 1.96 * se_mean(xs)
    return m - e, m + e


def unique_answers(row: dict) -> list:
    return [t.get("answer") for t in (row.get("traces") or [])]


def modal_share(answers: list) -> float:
    valid = [a for a in answers if a]
    if not valid:
        return 0.0
    return Counter(valid).most_common(1)[0][1] / len(valid)


def tv(p: Counter, q: Counter) -> float:
    keys = set(p) | set(q)
    n_p, n_q = sum(p.values()), sum(q.values())
    if not n_p or not n_q:
        return float("nan")
    return 0.5 * sum(abs(p[k] / n_p - q[k] / n_q) for k in keys)


def main() -> None:
    rows = [json.loads(line) for line in JSONL.open() if line.strip()]
    keys = [(r["question_id"], float(r["ratio"]), int(r["replicate"])) for r in rows]
    assert len(keys) == 18640, len(keys)
    assert len(set(keys)) == 18640
    snap = json.loads(SNAPSHOT.read_text())
    hist = {s["question_id"]: s for s in snap["snapshots"] if s.get("common_support")}
    qids = sorted({r["question_id"] for r in rows})
    assert len(qids) == 466
    assert set(qids) == set(hist)

    by_qr: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        by_qr[(row["question_id"], float(row["ratio"]))].append(row)
    for qid in qids:
        for ratio in RATIOS:
            assert len(by_qr[(qid, ratio)]) == 8, (qid, ratio)

    q_hist_correct = {qid: bool(hist[qid].get("history_correct")) for qid in qids}
    n_hist_ok = sum(q_hist_correct.values())
    n_hist_bad = 466 - n_hist_ok

    per_q: dict[str, dict[float, dict]] = {}
    for qid in qids:
        per_q[qid] = {}
        for ratio in RATIOS:
            chunk = by_qr[(qid, ratio)]
            answers = [a for row in chunk for a in unique_answers(row)]
            branch_flags = []
            for r in chunk:
                wv, wc, gold = r.get("window_vote"), bool(r["window_correct"]), r["gold"]
                for a in unique_answers(r):
                    if a == wv:
                        branch_flags.append(float(wc))
                    else:
                        branch_flags.append(float(bool(grade_answer("math", a, gold))))
            per_q[qid][ratio] = {
                "window_acc": mean([float(bool(r["window_correct"])) for r in chunk]),
                "final_acc": mean([float(bool(r["final_correct"])) for r in chunk]),
                "branch_acc": mean(branch_flags),
                "tok": mean([r["total_generated_tokens"] for r in chunk]),
                "prefix": mean([r["prefix_tokens"] for r in chunk]),
                "seconds": mean([r["generation_seconds"] for r in chunk]),
                "unique": mean([len(set(unique_answers(r))) for r in chunk]),
                "modal_share": mean([modal_share(unique_answers(r)) for r in chunk]),
                "trunc": sum(int(r.get("truncations") or 0) for r in chunk),
                "n_final": mean([r["n_final_votes"] for r in chunk]),
                "answer_counter": Counter(answers),
                "n_window_correct_reps": sum(bool(r["window_correct"]) for r in chunk),
                "n_final_correct_reps": sum(bool(r["final_correct"]) for r in chunk),
            }

    def collect(ratio: float, field: str, subset: list[str] | None = None) -> list[float]:
        ids = subset or qids
        return [per_q[qid][ratio][field] for qid in ids]

    def pack(ratio: float, subset: list[str] | None = None) -> dict:
        ids = subset or qids
        n = len(ids)
        out = {"n_questions": n, "n_units": n * 8}
        for field in (
            "window_acc",
            "final_acc",
            "branch_acc",
            "tok",
            "prefix",
            "seconds",
            "unique",
            "modal_share",
        ):
            xs = collect(ratio, field, ids)
            lo, hi = ci95(xs)
            out[field] = {
                "mean": mean(xs),
                "sd": sd(xs),
                "se": se_mean(xs),
                "ci95": [lo, hi],
            }
        out["truncations"] = sum(per_q[qid][ratio]["trunc"] for qid in ids)
        if ratio > 0:
            d_win = [per_q[qid][ratio]["window_acc"] - per_q[qid][0.0]["window_acc"] for qid in ids]
            d_fin = [per_q[qid][ratio]["final_acc"] - per_q[qid][0.0]["final_acc"] for qid in ids]
            d_tok = [per_q[qid][ratio]["tok"] - per_q[qid][0.0]["tok"] for qid in ids]
            d_br = [per_q[qid][ratio]["branch_acc"] - per_q[qid][0.0]["branch_acc"] for qid in ids]
            tvs = [
                tv(per_q[qid][ratio]["answer_counter"], per_q[qid][0.0]["answer_counter"])
                for qid in ids
            ]
            tok0 = mean(collect(0.0, "tok", ids))
            out["delta_vs_r0"] = {
                "window_acc_pp": 100 * mean(d_win),
                "window_acc_se_pp": 100 * se_mean(d_win),
                "window_acc_ci95_pp": [100 * x for x in ci95(d_win)],
                "n_q_window_down": sum(d < -1e-12 for d in d_win),
                "n_q_window_up": sum(d > 1e-12 for d in d_win),
                "n_q_window_tie": sum(abs(d) <= 1e-12 for d in d_win),
                "final_acc_pp": 100 * mean(d_fin),
                "final_acc_se_pp": 100 * se_mean(d_fin),
                "final_acc_ci95_pp": [100 * x for x in ci95(d_fin)],
                "n_q_final_down": sum(d < -1e-12 for d in d_fin),
                "n_q_final_up": sum(d > 1e-12 for d in d_fin),
                "n_q_final_tie": sum(abs(d) <= 1e-12 for d in d_fin),
                "branch_acc_pp": 100 * mean(d_br),
                "tok": mean(d_tok),
                "tok_se": se_mean(d_tok),
                "tok_save_pct": 100 * (1 - mean(collect(ratio, "tok", ids)) / tok0),
                "tv_answer": mean(tvs),
                "tv_answer_se": se_mean(tvs),
            }
        return out

    locked_final = []
    for qid in qids:
        means = [per_q[qid][r]["final_acc"] for r in RATIOS]
        locked_final.append(max(means) - min(means) < 1e-12)

    payload = {
        "fingerprint": "2ebec2c2eced",
        "snapshot": "75d4941afc34",
        "n_questions": 466,
        "n_hist_correct": n_hist_ok,
        "n_hist_incorrect": n_hist_bad,
        "replicates": 8,
        "note": (
            "Question-clustered means of 8 window resamples. "
            "Final vote conditions on cached SC windows 2-7; not full d_j(h). "
            "Raw JSONL grades. Not PVC / not Table 1."
        ),
        "by_ratio": {f"{r:.1f}": pack(r) for r in RATIOS},
        "by_ratio_hist_correct": {f"{r:.1f}": pack(r, [q for q in qids if q_hist_correct[q]]) for r in RATIOS},
        "by_ratio_hist_incorrect": {
            f"{r:.1f}": pack(r, [q for q in qids if not q_hist_correct[q]]) for r in RATIOS
        },
        "n_questions_final_identical_across_r": sum(locked_final),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps({"wrote": str(OUT), "n_hist_ok": n_hist_ok, "n_hist_bad": n_hist_bad}, ensure_ascii=False))
    for r in RATIOS:
        block = payload["by_ratio"][f"{r:.1f}"]
        extra = ""
        if r > 0:
            d = block["delta_vs_r0"]
            extra = (
                f"  Δwin {d['window_acc_pp']:+.3f}pp (SE {d['window_acc_se_pp']:.3f}; "
                f"up {d['n_q_window_up']} down {d['n_q_window_down']} tie {d['n_q_window_tie']}) "
                f"Δfinal {d['final_acc_pp']:+.3f}pp (SE {d['final_acc_se_pp']:.3f}) "
                f"tok {d['tok_save_pct']:.1f}% TV {d['tv_answer']:.3f}"
            )
        print(
            f"r={r:.1f}  win {100*block['window_acc']['mean']:.2f}%  "
            f"final {100*block['final_acc']['mean']:.2f}%  "
            f"branch {100*block['branch_acc']['mean']:.2f}%  "
            f"tok {block['tok']['mean']:.1f}  pref {block['prefix']['mean']:.1f}  "
            f"uniq {block['unique']['mean']:.2f}  modal {block['modal_share']['mean']:.3f}"
            + extra
        )
    print("locked_final", payload["n_questions_final_identical_across_r"])
    print("--- hist correct ---")
    for r in RATIOS:
        if r == 0:
            continue
        d = payload["by_ratio_hist_correct"][f"{r:.1f}"]["delta_vs_r0"]
        print(f"ok r={r:.1f} Δwin {d['window_acc_pp']:+.3f} Δfinal {d['final_acc_pp']:+.3f} save {d['tok_save_pct']:.1f}%")
    print("--- hist incorrect ---")
    for r in RATIOS:
        if r == 0:
            continue
        d = payload["by_ratio_hist_incorrect"][f"{r:.1f}"]["delta_vs_r0"]
        print(f"bad r={r:.1f} Δwin {d['window_acc_pp']:+.3f} Δfinal {d['final_acc_pp']:+.3f} save {d['tok_save_pct']:.1f}%")


if __name__ == "__main__":
    main()
