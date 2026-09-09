"""MATH-style answer extraction, vote canonicalization, and grading.

The primary grader is math-verify 0.9.0, installed outside the repository so
that its optional SymPy dependency does not alter the vLLM environment.
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter
from typing import Iterable, Optional


DEFAULT_MATH_VERIFY_PATH = "/data/26xyw/python_packages/math_verify_0_9"


def _balanced_braced_value(text: str, open_brace: int) -> Optional[str]:
    """Return the contents of the balanced group beginning at open_brace."""
    if open_brace >= len(text) or text[open_brace] != "{":
        return None
    depth = 0
    for pos in range(open_brace, len(text)):
        char = text[pos]
        if char == "{" and (pos == 0 or text[pos - 1] != "\\"):
            depth += 1
        elif char == "}" and (pos == 0 or text[pos - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : pos].strip()
    return None


def format_gsm8k_gold(target) -> str:
    """Canonicalize GSM8K JSONL targets (often JSON numbers like 18.0)."""
    if target is None:
        return ""
    if isinstance(target, bool):
        return str(target)
    if isinstance(target, int):
        return str(target)
    if isinstance(target, float):
        if target.is_integer():
            return str(int(target))
        return str(target)
    value = str(target).strip()
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return str(int(number))
    return value


def extract_gsm8k_numeric(text: Optional[str]) -> Optional[str]:
    """Fallback extractor when the model does not emit a boxed answer."""
    if not text:
        return None
    match = re.search(
        r"(?:the\s+)?answer\s+is\s*:?\s*([-+]?\d+(?:\.\d+)?)", text, flags=re.I
    )
    if match:
        return format_gsm8k_gold(match.group(1))
    matches = list(re.finditer(r"[-+]?\d+(?:\.\d+)?", text))
    if not matches:
        return None
    return format_gsm8k_gold(matches[-1].group())


def extract_last_boxed(text: str) -> Optional[str]:
    """Extract the last complete ``\boxed{...}`` expression."""
    if not text:
        return None
    starts = list(re.finditer(r"\\boxed\s*", text))
    for match in reversed(starts):
        pos = match.end()
        while pos < len(text) and text[pos].isspace():
            pos += 1
        value = _balanced_braced_value(text, pos)
        if value is not None and value != "":
            return value
    return None


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    """Cheap deterministic normalization used before answer-level voting."""
    if answer is None:
        return None
    value = str(answer).strip()
    if not value:
        return None
    value = value.replace("\n", " ")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = value.strip("$ ")
    value = re.sub(r"\s+", "", value)
    value = value.rstrip(".")
    return value or None


def majority_vote(answers: Iterable[Optional[str]]) -> Optional[str]:
    """Return a deterministic majority over normalized, valid answers."""
    valid = [normalize_answer(answer) for answer in answers]
    valid = [answer for answer in valid if answer is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def _load_math_verify():
    package_path = os.environ.get("MATH_VERIFY_PATH", DEFAULT_MATH_VERIFY_PATH)
    if package_path and package_path not in sys.path:
        # Append rather than prepend: torch requires the environment's pinned
        # SymPy 1.13.1, while the external target also contains SymPy 1.14.
        sys.path.append(package_path)
    try:
        from math_verify import parse, verify
    except ImportError as exc:
        raise RuntimeError(
            "math-verify is required for MATH500 grading. Install it under "
            f"{package_path} or set MATH_VERIFY_PATH."
        ) from exc
    return parse, verify


def math_equal(prediction: Optional[str], reference: Optional[str]) -> bool:
    """Judge a prediction with math-verify symbolic/numeric equivalence."""
    prediction = normalize_answer(prediction)
    reference = normalize_answer(reference)
    if prediction is None or reference is None:
        return False
    if prediction == reference:
        return True
    parse, verify = _load_math_verify()
    try:
        return bool(verify(parse(reference), parse(prediction)))
    except Exception:
        return False


def extract_aime_integer(answer: Optional[str]) -> Optional[str]:
    """Canonical AIME answer: integer string in 0–999."""
    if answer is None:
        return None
    text = str(answer).strip()
    if not text:
        return None
    compact = re.sub(r"[\s,]", "", text)
    compact = compact.strip("$")
    compact = re.sub(r"^\\?left\(", "", compact)
    compact = re.sub(r"\\?right\)$", "", compact)
    exact = re.fullmatch(r"0*(\d{1,3})", compact)
    if exact:
        return str(int(exact.group(1)))
    numbers = re.findall(r"\d+", text)
    if not numbers:
        return None
    value = int(numbers[-1])
    if 0 <= value <= 999:
        return str(value)
    return None


def aime_equal(prediction: Optional[str], reference: Optional[str]) -> bool:
    left = extract_aime_integer(prediction)
    right = extract_aime_integer(reference)
    return left is not None and left == right


def extract_gpqa_letter(answer: Optional[str]) -> Optional[str]:
    """Canonical GPQA vote key: a single A/B/C/D letter."""
    if answer is None:
        return None
    text = str(answer).strip()
    if not text:
        return None
    stripped = text.strip().strip("()[]{}.").strip()
    if len(stripped) == 1 and stripped.upper() in "ABCD":
        return stripped.upper()
    match = re.match(r"^\(?([ABCD])\)?(?:[.:)\s]|$)", text.strip(), flags=re.I)
    if match:
        return match.group(1).upper()
    match = re.search(
        r"\b(?:answer|option|choice)\s*(?:is\s*)?:?\s*\(?([ABCD])\b",
        text,
        flags=re.I,
    )
    if match:
        return match.group(1).upper()
    letters = re.findall(r"\b([ABCD])\b", text, flags=re.I)
    if len(set(letter.upper() for letter in letters)) == 1:
        return letters[0].upper()
    return None


def gpqa_equal(prediction: Optional[str], reference: Optional[str]) -> bool:
    left = extract_gpqa_letter(prediction)
    right = extract_gpqa_letter(reference)
    return left is not None and left == right


def canonicalize_vote_answer(task: str, raw: Optional[str]) -> Optional[str]:
    """Normalize a boxed payload so majority vote keys match the grader."""
    if task == "aime":
        return extract_aime_integer(raw)
    if task == "gpqa":
        return extract_gpqa_letter(raw)
    return normalize_answer(raw)


def grade_answer(task: str, prediction: Optional[str], reference: Optional[str]) -> bool:
    if task == "aime":
        return aime_equal(prediction, reference)
    if task == "gpqa":
        return gpqa_equal(prediction, reference)
    return math_equal(prediction, reference)

