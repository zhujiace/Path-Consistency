"""Aggregate per-config result files into the paper's Table 1 layout.

Computes, relative to the SC baseline:
  - accuracy
  - total speedup and per-level speedup (%)
  - total token decrease and per-level decrease (%)
"""

import json
import os

from collections import defaultdict

RESULTS_DIR = "results"
SEED = 0


def summarize(tag):
    path = os.path.join(RESULTS_DIR, f"{tag}.jsonl")
    per_question = []
    with open(path) as f:
        for line in f:
            per_question.append(json.loads(line))

    n = len(per_question)
    slot = 5  # 20 branches / (3 levels + 1)
    agg = defaultdict(lambda: defaultdict(float))
    acc = 0
    tot_tok = 0.0
    tot_lat = 0.0
    for q in per_question:
        acc += int(q["result"])
        tot_tok += sum(q["tokens_gen"])
        tot_lat += sum(q["latency"])
        for i, (t, l) in enumerate(zip(q["tokens_gen"], q["latency"])):
            agg[i // slot]["tok"] += t
            agg[i // slot]["lat"] += l
            agg[i // slot]["cnt"] += 1

    return {
        "tag": tag,
        "n": n,
        "accuracy": acc / n if n else 0.0,
        "avg_tokens": tot_tok / n if n else 0.0,
        "avg_latency": tot_lat / n if n else 0.0,
        "slots": {k: dict(v) for k, v in sorted(agg.items())},
    }


def main():
    tags = ["SC_seed0", "PC_C0_seed0", "PC_C0.7_seed0", "PC_C0.8_seed0", "PC_C0.9_seed0"]
    data = {}
    for t in tags:
        if os.path.exists(os.path.join(RESULTS_DIR, f"{t}.jsonl")):
            data[t] = summarize(t)

    sc = data.get("SC_seed0")
    if sc is None:
        raise SystemExit("SC baseline not found")

    print(f"{'Method':<14}{'Acc(%)':>9}{'TotSpd(%)':>10}{'L1(%)':>8}{'L2(%)':>8}"
          f"{'L3(%)':>8}{'Tok#':>10}{'TotDec(%)':>10}{'L1Dec(%)':>9}"
          f"{'L2Dec(%)':>9}{'L3Dec(%)':>9}  (n={sc['n']})")
    print("-" * 113)

    for name, tag in [("SC", "SC_seed0"),
                      ("PC (C=0)", "PC_C0_seed0"),
                      ("PC (C=0.7)", "PC_C0.7_seed0"),
                      ("PC (C=0.8)", "PC_C0.8_seed0"),
                      ("PC (C=0.9)", "PC_C0.9_seed0")]:
        if tag not in data:
            continue
        s = data[tag]
        acc = s["accuracy"] * 100
        tok = s["avg_tokens"]
        lat = s["avg_latency"]

        tot_spd = (1 - lat / sc["avg_latency"]) * 100
        tot_dec = (1 - tok / sc["avg_tokens"]) * 100

        lv_spd = []
        lv_dec = []
        for k in range(len(sc["slots"])):
            s_lat = sc["slots"][k]["lat"]
            s_tok = sc["slots"][k]["tok"]
            lv_spd.append((1 - s["slots"][k]["lat"] / s_lat) * 100 if s_lat else 0.0)
            lv_dec.append((1 - s["slots"][k]["tok"] / s_tok) * 100 if s_tok else 0.0)

        # level k = window slot k+1 (the paper's Level-1..3)
        l1, l2, l3 = lv_spd[1], lv_spd[2], lv_spd[3]
        d1, d2, d3 = lv_dec[1], lv_dec[2], lv_dec[3]
        print(f"{name:<14}{acc:>9.1f}{tot_spd:>10.1f}{l1:>8.1f}{l2:>8.1f}"
              f"{l3:>8.1f}{tok:>10.1f}{tot_dec:>10.1f}{d1:>9.1f}"
              f"{d2:>9.1f}{d3:>9.1f}")


if __name__ == "__main__":
    main()
