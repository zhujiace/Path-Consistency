"""Accuracy vs tokens scatter/frontier for GSM8K (paper main figure prototype).

x = avg tokens per problem, y = accuracy (%).
Points: SC@20, AC, ESC, margin, PC(C) for C in {0, 0.7, 0.8, 0.9} (extendable
to SC@N sweep and per-dataset subplots).
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "/home/26xyw/projects/path-consistency/results"


def load_stats(tag):
    path = os.path.join(RESULTS, f"{tag}.jsonl")
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    acc = 100 * sum(r["result"] for r in rows) / n
    tok = sum(sum(r["tokens_gen"]) for r in rows) / n
    return acc, tok


def main():
    order = ["SC_seed0", "AC_seed0", "ESC_seed0", "MARGIN_seed0",
             "PC_C0_seed0", "PC_C0.7_seed0", "PC_C0.8_seed0", "PC_C0.9_seed0"]
    labels = {"SC_seed0": "SC (N=20)",
              "AC_seed0": "AC",
              "ESC_seed0": "ESC",
              "MARGIN_seed0": "Margin-stop",
              "PC_C0_seed0": "PC (C=0)",
              "PC_C0.7_seed0": "PC (C=0.7)",
              "PC_C0.8_seed0": "PC (C=0.8)",
              "PC_C0.9_seed0": "PC (C=0.9)"}
    pc = ["PC_C0_seed0", "PC_C0.7_seed0", "PC_C0.8_seed0", "PC_C0.9_seed0"]
    colors = {"SC_seed0": "#333333", "AC_seed0": "#c44e52",
              "ESC_seed0": "#dd8452", "MARGIN_seed0": "#55a868"}

    stats = {}
    for tag in order:
        if os.path.exists(os.path.join(RESULTS, f"{tag}.jsonl")):
            stats[tag] = load_stats(tag)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(*zip(*[stats[t] for t in pc]),
               s=70, marker="o", color="#4c72b0", zorder=3,
               label="PC (ours)")
    xs = [stats[t][1] for t in pc]
    ys = [stats[t][0] for t in pc]
    ax.plot(xs, ys, "--", color="#4c72b0", lw=1, zorder=2)
    for t in stats:
        if t in pc:
            continue
        ax.scatter(*stats[t], s=90, marker="^", color=colors[t], zorder=3,
                   label=labels[t])
    ax.set_xlabel("avg tokens per problem")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("GSM8K, Llama-3-8B, seed 0 (test n=1319)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = "/home/26xyw/projects/path-consistency/paper/review/accuracy_tokens_gsm8k_proto.pdf"
    fig.savefig(out)
    print("saved", out)
    for t, (a, tk) in sorted(stats.items(), key=lambda z: z[1][1]):
        print(f"{t:>16}: acc={a:5.1f} tok={tk:6.0f}")


if __name__ == "__main__":
    main()
