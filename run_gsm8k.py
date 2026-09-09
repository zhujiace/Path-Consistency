"""Main experiment runner: GSM8K with self-consistency / path-consistency.

Usage:
    python run_gsm8k.py --method sc  --gpu 0 --seed 0
    python run_gsm8k.py --method pc --confidence 0.8 --gpu 1 --seed 0
"""

import argparse
import json
import os
import sys

# must be set before importing vllm/torch (CUDA context caching)
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]

from path_consistency.inference import PCInference
from path_consistency.prompts import build_prompt
from path_consistency.answers import extract_answer, is_correct


def load_gsm8k(max_questions=None, seed=0):
    """Load the pre-processed GSM8K JSONL (no pyarrow in the GPU process)."""
    import random

    path = os.path.expanduser("~/data/Datasets/gsm8k_test.jsonl")
    data = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            data.append({"question": r["question"], "target": r["target"]})
    if max_questions is not None:
        rng = random.Random(seed)
        rng.shuffle(data)
        data = data[:max_questions]
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["sc", "pc", "ac", "esc", "margin"],
                    required=True)
    ap.add_argument("--confidence", type=float, default=0.8,
                    help="beta confidence threshold (pc) / ac stopping thresh")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-branches", type=int, default=20)
    ap.add_argument("--max-level", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-questions", type=int, default=None,
                    help="limit dataset size for quick runs")
    ap.add_argument("--model", default=os.path.expanduser(
        "~/data/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/"
        "8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"))
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--store-text", action="store_true",
                    help="store branch texts (generations + prefixes) in jsonl")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    m = args.method
    es, es_thresh = None, None
    if m == "sc":
        threshold = 1.0
        tag = f"SC_seed{args.seed}"
    elif m == "pc":
        threshold = args.confidence
        tag = f"PC_C{args.confidence:g}_seed{args.seed}"
    else:
        threshold, es = 1.0, m
        es_thresh = args.confidence if m == "ac" else 0.9
        tag = f"{m.upper()}_seed{args.seed}"

    # load dataset BEFORE creating the vLLM engine: datasets uses fork(),
    # which is unsafe after a CUDA context exists
    data = load_gsm8k(max_questions=args.max_questions, seed=args.seed)

    engine = PCInference(
        model_path=args.model,
        max_branch=args.max_branches,
        max_level=args.max_level,
        confidence_thres=threshold,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        early_stop=es,
        early_stop_thresh=es_thresh or 0.9,
    )

    results = []
    n_correct = 0
    total_tokens = 0
    total_latency = 0.0
    n_slots = args.max_level + 1
    per_slot_tokens = {k: 0 for k in range(n_slots)}
    per_slot_latency = {k: 0.0 for k in range(n_slots)}
    per_slot_count = {k: 0 for k in range(n_slots)}
    slot = args.max_branches // n_slots

    out_file = os.path.join(args.outdir, f"{tag}.jsonl")
    done_ids = set()
    if os.path.exists(out_file):
        with open(out_file) as f:
            for line in f:
                done_ids.add(json.loads(line)["id"])

    fp_out = open(out_file, "a")
    for idx, item in enumerate(data):
        if idx in done_ids:
            continue
        prompt = build_prompt(item["question"])
        info = engine.inference(prompt)
        gold = item["target"]
        ok = is_correct(info["answer"], gold)

        per_q = {
            "id": idx,
            "question": item["question"],
            "target": gold,
            "result": ok,
            "answer": info["answer"],
            "answers": info["answers"],
            "tokens_gen": info["tokens_gen"],
            "prefix_level": info["prefix_level"],
            "latency": info["latency"],
        }
        if args.store_text:
            per_q["generations"] = info["generations"]
            per_q["prefixes_used"] = info["prefixes_used"]
        per_q["n_branches"] = info.get("n_branches", len(info["tokens_gen"]))
        results.append(per_q)
        fp_out.write(json.dumps(per_q) + "\n")
        fp_out.flush()

        n_correct += int(ok)
        total_tokens += sum(info["tokens_gen"])
        total_latency += sum(info["latency"])
        if args.method in ("sc", "pc"):
            for i, (t, lvl) in enumerate(zip(info["tokens_gen"], info["prefix_level"])):
                s = i // slot  # window slot (level in the paper's table)
                per_slot_tokens[s] += t
                per_slot_latency[s] += info["latency"][i]
                per_slot_count[s] += 1
        if (idx + 1) % 200 == 0:
            print(f"[{tag}] {idx+1}/{len(data)} acc so far: {n_correct/(idx+1):.4f}", flush=True)
    fp_out.close()

    summary = {
        "tag": tag,
        "method": args.method,
        "confidence": args.confidence,
        "seed": args.seed,
        "num_questions": len(data),
        "questions_done": len(results),
        "accuracy": n_correct / len(data) if data else 0.0,
        "total_tokens": total_tokens,
        "avg_tokens_per_question": total_tokens / len(data) if data else 0.0,
        "total_latency": total_latency,
        "avg_latency_per_question": total_latency / len(data) if data else 0.0,
        "avg_branches_used": sum(
            r.get("n_branches", 20) for r in results) / len(results) if results else 0,
        "per_slot_tokens": per_slot_tokens,
        "per_slot_latency": per_slot_latency,
        "per_slot_count": per_slot_count,
    }

    with open(os.path.join(args.outdir, f"{tag}.json"), "w") as f:
        json.dump({"summary": summary, "per_question": results}, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
