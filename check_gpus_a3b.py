#!/usr/bin/env python3
"""Print JSON: whether GPUs 1-4 have enough free memory for A3B TP=4."""
from __future__ import annotations

import json
import subprocess

MAX_USED_MIB = 3500
GPUS = (1, 2, 3, 4)


def main() -> None:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in out.strip().splitlines():
        idx, used, free, util = [part.strip() for part in line.split(",")]
        rows.append(
            {
                "index": int(idx),
                "used_mib": int(float(used)),
                "free_mib": int(float(free)),
                "util": int(float(util)),
            }
        )
    target = [row for row in rows if row["index"] in GPUS]
    free_enough = all(row["used_mib"] <= MAX_USED_MIB for row in target)
    print(
        json.dumps(
            {"free_enough": free_enough, "max_used_mib": MAX_USED_MIB, "gpus": target},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
