"""Answer extraction and aggregation for GSM8K-style numeric answers."""

import re
from collections import Counter

NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def extract_answer(generation: str):
    """Return the numeric answer from a CoT generation.

    Prefers the number following "answer is"; falls back to the last number
    in the generation. Returns None if no number is found.
    """
    text = generation
    m = re.search(r"answer is\s*:?\s*[-+]?\d+(?:\.\d+)?", text)
    if m:
        return float(NUM_RE.search(m.group()).group())

    matches = list(NUM_RE.finditer(text))
    if not matches:
        return None
    return float(matches[-1].group())


def integrate_answer(answers: list):
    """Majority vote over per-branch answers."""
    if not answers:
        return None
    counter = Counter(answers)
    return counter.most_common(1)[0][0]


def is_correct(pred, gold) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except (TypeError, ValueError):
        return False
