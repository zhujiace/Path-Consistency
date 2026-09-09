"""One-time dataset preparation.

Loads the HF datasets and stores them as plain JSONL under ~/data/Datasets
so that GPU inference processes never touch pyarrow (whose jemalloc
background thread races with CUDA and segfaults intermittently).
"""

import json
import os

import datasets as hf_datasets

DATA_DIR = os.path.expanduser("~/data/Datasets")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    ds = hf_datasets.load_dataset("openai/gsm8k", "main", split="test")
    out = os.path.join(DATA_DIR, "gsm8k_test.jsonl")
    with open(out, "w") as f:
        for r in ds:
            gold = r["answer"].split("####")[-1].strip().replace(",", "")
            f.write(json.dumps({"question": r["question"], "target": float(gold)}) + "\n")
    print(f"GSM8K test: {len(ds)} questions -> {out}")


if __name__ == "__main__":
    main()
