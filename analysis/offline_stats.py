"""Offline analyses over stored GSM8K SC/PC results (no GPU needed).

Answers reviewer questions:
  A2 proxy : gate outcome & prefix survival, difficulty-grouped
  C1       : multi-level motivation -> prefix value curve V(t) [needs --store-text]
  C2       : diversity / answer collapse per window
"""

import json
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/home/26xyw/projects/path-consistency")
from path_consistency.answers import is_correct
from path_consistency.prefix import split_sentences

RESULTS = "/home/26xyw/projects/path-consistency/results"
WIN = 5  # branches per window


def load(tag):
    rows = []
    with open(f"{RESULTS}/{tag}.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def majority(answers):
    return Counter(answers).most_common(1)[0][0]


def prefix_value_curve(sc):
    """V(t): among branches sharing the majority branch's first-t-sentence
    prefix, P(that group's majority answer is correct) and agreement with the
    full-run majority answer."""
    stats = defaultdict(lambda: {"n": 0, "agree": 0, "correct": 0,
                                 "agree_and_correct": 0})
    for q in sc:
        gens = q["generations"]
        answers = q["answers"]
        gold = q["target"]
        mv = majority(answers)
        mv_gold = is_correct(mv, gold)
        sents = split_sentences(gens[answers.index(mv)])

        for t in range(min(len(sents), 8) + 1):
            if t == 0:
                prefix = ""
                grp = list(range(len(gens)))
            else:
                prefix = "".join(sents[:t])
                grp = [i for i, g in enumerate(gens)
                       if g.startswith(prefix)]
            if not grp:
                continue
            s = stats[t]
            s["n"] += 1
            grp_mv = majority([answers[i] for i in grp])
            s["agree"] += int(grp_mv == mv)
            s["correct"] += int(is_correct(grp_mv, gold))
            s["agree_and_correct"] += int(mv_gold and grp_mv == mv)

    print("\n=== V(t): prefix value curve over SC (n=%d) ==="
          % stats[0]["n"])
    print(f"{'t':>2} {'avg grp':>8} {'keepMV%':>9} {'grpMV correct%':>15} "
          f"{'grpMV correct & MV ok%':>22}")
    for t in sorted(stats):
        s = stats[t]
        avg = (1 + s["agree"] / s["n"])
        print(f"{t:>2} {avg:>8.2f} "
              f"{100*s['agree']/s['n']:>9.1f} "
              f"{100*s['correct']/s['n']:>15.1f} "
              f"{100*s['agree_and_correct']/max(1,s['agree']):>22.1f}")


def gate_survival(pc_tag, sc_tag=None):
    rows = load(pc_tag)
    sc_rows = load(sc_tag) if sc_tag else None

    buckets = {"easy": {"q": 0, "ok": 0}, "hard": {"q": 0, "ok": 0}}
    gate = Counter()
    # per boundary k (after window k+1): prefix promoted into next window?
    for qi, q in enumerate(rows):
        gold = q["target"]
        answers = q["answers"]
        lvls = q["prefix_level"]
        n_corr = sum(is_correct(a, gold) for a in answers)
        easy = n_corr >= 14
        buckets["easy" if easy else "hard"]["q"] += 1
        if sc_rows is not None:
            ok_sc = is_correct(majority(sc_rows[qi]["answers"]), gold)
            buckets["easy" if easy else "hard"]["ok"] += int(ok_sc)
        for k in range(3):
            passed = lvls[WIN * (k + 1)] == k + 1
            if passed:
                gate["pass"] += 1
                win_mv = majority(answers[WIN * k:WIN * (k + 1)])
                gate["pass_correct_winmv"] += int(is_correct(win_mv, gold))
            else:
                gate["fail"] += 1

    print(f"\n=== Gate stats ({pc_tag}, n={len(rows)}) ===")
    print(f"boundaries: pass={gate['pass']} fail={gate['fail']} "
          f"| pass with correct window-majority="
          f"{gate['pass_correct_winmv']} "
          f"({100*gate['pass_correct_winmv']/max(1,gate['pass']):.1f}%)")
    if sc_rows:
        for d in buckets:
            b = buckets[d]
            acc = 100 * b["ok"] / max(1, b["q"])
            print(f"difficulty={d}: n={b['q']} (SC-majority correct {acc:.1f}%)")


def diversity_per_window(tags):
    print(f"\n=== avg distinct answers per window ===")
    header = "  ".join(f"w{i}" for i in range(4))
    for tag in tags:
        rows = load(tag)
        per_win = defaultdict(list)
        for q in rows:
            for w in range(4):
                per_win[w].append(len(set(q["answers"][WIN * w:WIN * (w + 1)])))
        line = "  ".join(f"{sum(v)/len(v):.2f}" for _, v in sorted(per_win.items()))
        print(f"{tag:>14}: {header}\n{'':>14}  {line}")


if __name__ == "__main__":
    tags = sys.argv[1:] or ["SC_seed0", "PC_C0_seed0", "PC_C0.8_seed0"]
    sc = load("SC_seed0")
    if "generations" in sc[0]:
        prefix_value_curve(sc)
    else:
        print("(SC results lack branch texts; rerun with --store-text for V(t))")
    gate_survival("PC_C0.8_seed0", "SC_seed0")
    diversity_per_window(tags)
